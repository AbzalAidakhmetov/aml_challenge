import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.common import load_data, prepare_train_data
from src.eval import visualize_retrieval
from train_mrr import (
    MLP,
    MODEL_PATH,
    VAL_RATIO,
    RANDOM_SEED,
    DEVICE,
    MODEL_PATH_MRR,
    seed_everything,
)

TOP_K_FILTER = 50
PRED_BATCH_SIZE = 1024
SIM_CHUNK_SIZE = 512
MAX_SUSPECT_VIZ = 5

def main() -> None:
    # 1) Recreate the image-based validation split
    seed_everything(RANDOM_SEED)

    train_data = load_data("data/train/train.npz")
    X, y, label = prepare_train_data(train_data)

    caption_image_ids = torch.argmax(label.float(), dim=1).long()
    num_images = label.shape[1]

    g = torch.Generator().manual_seed(RANDOM_SEED)
    permuted = torch.randperm(num_images, generator=g)
    num_val_images = max(1, int(num_images * VAL_RATIO))
    val_image_ids = permuted[-num_val_images:]
    val_image_mask = torch.zeros(num_images, dtype=torch.bool)
    val_image_mask[val_image_ids] = True

    val_mask = val_image_mask[caption_image_ids]
    val_text_embd = X[val_mask]
    val_caption_ids = caption_image_ids[val_mask]

    val_caption_text = train_data["captions/text"][val_mask.cpu().numpy()]
    val_img_mask = val_image_mask.cpu().numpy()
    val_img_names = train_data["images/names"][val_img_mask]
    val_img_embd = torch.from_numpy(train_data["images/embeddings"][val_img_mask]).float()

    # Map original image ids -> contiguous indices within the validation image pool
    image_id_to_val_idx = {
        int(orig_id): idx
        for idx, orig_id in enumerate(torch.arange(num_images)[val_img_mask])
    }
    val_label = np.array([image_id_to_val_idx[int(img_id)] for img_id in val_caption_ids])

    # 2) Run best baseline (train_mrr) checkpoint on all validation captions
    model = MLP().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH_MRR, map_location=DEVICE))
    model.eval()

    val_ds = TensorDataset(val_text_embd, val_caption_ids)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)

    filtered_dir = Path("data/filtered")
    filtered_dir.mkdir(parents=True, exist_ok=True)
    clean_dataset_path = filtered_dir / "train_clean_top50.npz"
    suspect_dataset_path = filtered_dir / "train_suspect_top50.npz"

    # 1) Embed all captions with the current model
    full_dataset = TensorDataset(X, caption_image_ids)
    full_loader = DataLoader(full_dataset, batch_size=PRED_BATCH_SIZE, shuffle=False)
    model.eval()
    all_caption_preds = []
    with torch.no_grad():
        for text_batch, _ in tqdm(full_loader, desc="Embedding all captions"):
            preds = model(text_batch.to(DEVICE)).cpu()
            all_caption_preds.append(preds)
    all_caption_preds = torch.cat(all_caption_preds, dim=0)

    # 2) Check whether each ground-truth image appears in the top-k list
    norm_caption_preds = F.normalize(all_caption_preds, dim=-1)
    all_image_embd = torch.from_numpy(train_data["images/embeddings"]).float()
    norm_image_embd = F.normalize(all_image_embd, dim=-1)
    num_captions = norm_caption_preds.size(0)

    hit_mask = torch.zeros(num_captions, dtype=torch.bool)
    gt_ranks = torch.full((num_captions,), -1, dtype=torch.int32)

    for start in tqdm(
        range(0, num_captions, SIM_CHUNK_SIZE), desc="Computing top-k hits"
    ):
        end = min(start + SIM_CHUNK_SIZE, num_captions)
        sims = norm_caption_preds[start:end] @ norm_image_embd.T
        top_vals, top_idx = sims.topk(TOP_K_FILTER, dim=1)
        gt_chunk = caption_image_ids[start:end]
        matches = top_idx == gt_chunk.unsqueeze(1)
        chunk_hits = matches.any(dim=1)
        hit_mask[start:end] = chunk_hits
        ranks = torch.argmax(matches.int(), dim=1) + 1
        gt_ranks[start:end] = torch.where(chunk_hits, ranks, torch.full_like(ranks, -1))

    clean_mask = hit_mask.numpy()
    suspect_mask = ~clean_mask
    rank_np = gt_ranks.numpy()
    print(f"Total captions: {num_captions:,}")
    print(f"Clean captions (GT in top-{TOP_K_FILTER}): {clean_mask.sum():,}")
    print(f"Suspect captions: {suspect_mask.sum():,}")

    # 3) Export filtered datasets
    caption_keys = [key for key in train_data.keys() if key.startswith("captions/")]
    image_keys = [key for key in train_data.keys() if key.startswith("images/")]
    other_keys = [
        key for key in train_data.keys() if key not in caption_keys + image_keys
    ]

    def export_subset(mask, output_path):
        subset = {}
        for key in caption_keys:
            subset[key] = train_data[key][mask]
        for key in image_keys + other_keys:
            subset[key] = train_data[key]
        np.savez_compressed(output_path, **subset)
        print(f"→ Saved {output_path} ({subset[caption_keys[0]].shape[0]:,} captions)")

    export_subset(clean_mask, clean_dataset_path)
    export_subset(suspect_mask, suspect_dataset_path)


if __name__ == "__main__":
    main()
