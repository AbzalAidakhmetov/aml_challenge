import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from src.common import load_data


def _load_clip_model(
    device: torch.device,
    model_name: str = "ViT-B-32",
    pretrained: str = "laion2b_s34b_b79k",
):
    """
    Load an OpenCLIP model, its preprocessing transforms and tokenizer.

    Notes:
        - Uses LAION-pretrained weights (no explicit Flickr30k fine-tuning).
    """
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model.to(device)
    model.eval()
    return model, preprocess, tokenizer


@torch.inference_mode()
def _encode_images(
    image_names: np.ndarray,
    images_root: Path,
    model,
    preprocess,
    device: torch.device,
    batch_size: int = 64,
) -> torch.Tensor:
    """
    Encode all images listed in `image_names` into CLIP feature space.

    Returns:
        Tensor of shape [num_images, dim], L2-normalized.
    """
    num_images = len(image_names)
    features: list[torch.Tensor] = []

    for start in tqdm(range(0, num_images, batch_size), desc="Encoding images with CLIP"):
        end = min(start + batch_size, num_images)
        batch_names = image_names[start:end]
        batch_imgs = []
        for name in batch_names:
            img_path = images_root / name
            img = Image.open(img_path).convert("RGB")
            batch_imgs.append(preprocess(img).unsqueeze(0))

        if not batch_imgs:
            continue

        images_tensor = torch.cat(batch_imgs, dim=0).to(device)
        img_feats = model.encode_image(images_tensor)
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
        features.append(img_feats.cpu())

    if not features:
        raise RuntimeError("No image features were computed. Check image paths.")

    return torch.cat(features, dim=0)


@torch.inference_mode()
def _encode_captions(
    captions: np.ndarray,
    model,
    tokenizer,
    device: torch.device,
    batch_size: int = 128,
) -> torch.Tensor:
    """
    Encode all captions into CLIP text feature space (L2-normalized).
    """
    all_feats: list[torch.Tensor] = []
    captions_list = captions.tolist()
    num_captions = len(captions_list)

    for start in tqdm(range(0, num_captions, batch_size), desc="Encoding captions with CLIP"):
        end = min(start + batch_size, num_captions)
        batch_text = captions_list[start:end]
        tokens = tokenizer(batch_text).to(device)
        text_feats = model.encode_text(tokens)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
        all_feats.append(text_feats.cpu())

    if not all_feats:
        raise RuntimeError("No caption features were computed.")

    return torch.cat(all_feats, dim=0)


def compute_clip_pair_scores(
    data_path: Path,
    images_root: Path,
    device: str | None = None,
    image_batch_size: int = 64,
    text_batch_size: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute CLIP similarity scores for each (caption, ground-truth image) pair.

    Returns:
        scores: np.ndarray of shape [num_captions], cosine similarities.
        caption_image_ids: np.ndarray of shape [num_captions], image indices.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)

    data = load_data(str(data_path))
    captions = data["captions/text"]
    image_names = data["images/names"]
    label = data["captions/label"]

    if label.ndim != 2:
        raise ValueError(f"Expected label matrix of shape [N, M], got {label.shape}")

    caption_image_ids = np.argmax(label.astype(np.float32), axis=1).astype(np.int64)

    model, preprocess, tokenizer = _load_clip_model(torch_device)
    img_feats = _encode_images(image_names, images_root, model, preprocess, torch_device, image_batch_size)
    txt_feats = _encode_captions(captions, model, tokenizer, torch_device, text_batch_size)

    if txt_feats.size(0) != caption_image_ids.shape[0]:
        raise RuntimeError(
            f"Mismatch between captions ({txt_feats.size(0)}) and labels ({caption_image_ids.shape[0]})."
        )

    img_feats = img_feats.to(torch_device)
    txt_feats = txt_feats.to(torch_device)
    caption_image_ids_t = torch.from_numpy(caption_image_ids).long().to(torch_device)

    pair_img_feats = img_feats[caption_image_ids_t]
    scores = (txt_feats * pair_img_feats).sum(dim=-1)
    scores_np = scores.cpu().numpy().astype(np.float32)
    return scores_np, caption_image_ids


def export_filtered_dataset(
    data_path: Path,
    scores: np.ndarray,
    drop_fraction: float,
    output_path: Path,
) -> None:
    """
    Filter out the worst-scoring fraction of caption-image pairs and save a new .npz.

    Args:
        data_path: Path to the original .npz file (e.g. data/filtered/train_clean_top50.npz).
        scores: CLIP similarity scores for each caption.
        drop_fraction: Fraction in [0, 1) of lowest-scoring captions to drop.
        output_path: Path to the filtered .npz to create.
    """
    if not (0.0 <= drop_fraction < 1.0):
        raise ValueError(f"drop_fraction must be in [0, 1), got {drop_fraction}")

    data = load_data(str(data_path))
    num_captions = scores.shape[0]
    if data["captions/embeddings"].shape[0] != num_captions:
        raise RuntimeError(
            "Scores length does not match number of captions in dataset: "
            f"{num_captions} vs {data['captions/embeddings'].shape[0]}"
        )

    if drop_fraction == 0.0:
        keep_mask = np.ones(num_captions, dtype=bool)
    else:
        threshold = np.quantile(scores, drop_fraction)
        keep_mask = scores > threshold

        # Guarantee exact drop fraction (within 1 caption) by correcting ties if needed
        to_keep = keep_mask.sum()
        target_keep = int(round(num_captions * (1.0 - drop_fraction)))
        if to_keep > target_keep:
            # Drop the lowest-scoring among those currently kept
            kept_indices = np.nonzero(keep_mask)[0]
            kept_scores = scores[kept_indices]
            order = np.argsort(kept_scores)  # ascending
            num_extra = to_keep - target_keep
            drop_indices = kept_indices[order[:num_extra]]
            keep_mask[drop_indices] = False
        elif to_keep < target_keep:
            # Add back highest-scoring dropped examples
            dropped_indices = np.nonzero(~keep_mask)[0]
            dropped_scores = scores[dropped_indices]
            order = np.argsort(-dropped_scores)  # descending
            num_missing = target_keep - to_keep
            add_indices = dropped_indices[order[:num_missing]]
            keep_mask[add_indices] = True

    caption_keys = [key for key in data.keys() if key.startswith("captions/")]
    image_keys = [key for key in data.keys() if key.startswith("images/")]
    other_keys = [key for key in data.keys() if key not in caption_keys + image_keys]

    subset: dict[str, np.ndarray] = {}
    for key in caption_keys:
        subset[key] = data[key][keep_mask]
    for key in image_keys + other_keys:
        subset[key] = data[key]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(output_path), **subset)
    kept = int(keep_mask.sum())
    print(
        f"Saved filtered dataset to {output_path} with {kept:,} / {num_captions:,} captions "
        f"(drop_fraction={drop_fraction:.3f})"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Filter image-caption pairs using CLIP similarity scores."
    )
    parser.add_argument(
        "--input-npz",
        type=Path,
        default=Path("data/filtered/train_clean_top50.npz"),
        help="Path to input .npz file containing captions and images.",
    )
    parser.add_argument(
        "--images-root",
        type=Path,
        default=Path("data/train/Images"),
        help="Root directory containing image files referenced in the .npz.",
    )
    parser.add_argument(
        "--output-npz",
        type=Path,
        default=Path("data/filtered/train_clean_clip_q0.1.npz"),
        help="Path for the filtered output .npz file.",
    )
    parser.add_argument(
        "--scores-output",
        type=Path,
        default=None,
        help="Optional path to save raw CLIP scores as a .npy file.",
    )
    parser.add_argument(
        "--drop-fraction",
        type=float,
        default=0.10,
        help="Fraction of lowest-scoring caption-image pairs to drop (e.g., 0.10 for 10%%).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (e.g., 'cuda', 'cuda:0', 'cpu'). Auto-detect if not set.",
    )
    parser.add_argument(
        "--image-batch-size",
        type=int,
        default=64,
        help="Batch size for CLIP image encoding.",
    )
    parser.add_argument(
        "--text-batch-size",
        type=int,
        default=256,
        help="Batch size for CLIP text encoding.",
    )

    args = parser.parse_args()

    scores, _ = compute_clip_pair_scores(
        data_path=args.input_npz,
        images_root=args.images_root,
        device=args.device,
        image_batch_size=args.image_batch_size,
        text_batch_size=args.text_batch_size,
    )

    if args.scores_output is not None:
        args.scores_output.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(args.scores_output), scores)
        print(f"Saved raw scores to {args.scores_output}")

    export_filtered_dataset(
        data_path=args.input_npz,
        scores=scores,
        drop_fraction=args.drop_fraction,
        output_path=args.output_npz,
    )


if __name__ == "__main__":
    main()


