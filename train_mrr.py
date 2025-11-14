import random
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Sampler, TensorDataset
from tqdm import tqdm

from src.common import generate_submission, load_data, prepare_train_data


# Configuration
MODEL_PATH = "models/mlp_baseline_mrr_best_loss.pth"
MODEL_PATH_MRR = "models/mlp_baseline_mrr_best_mrr.pth"
SUBMISSION_PATH = "submission_mrr.csv"
EPOCHS = 50
BATCH_SIZE = 1024 * 3
LR = 0.001
VAL_RATIO = 0.1
RANDOM_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loss / contrastive configuration
LAMBDA_REG = 0
LAMBDA_NCE_MAX = 1
LAMBDA_NCE_WARMUP_EPOCHS = 1
INFO_NCE_TAU = 0.01


class MLP(nn.Module):
    def __init__(self, input_dim=1024, output_dim=1536, hidden_dim=2048):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.middle = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.decoder = nn.Sequential(nn.Linear(hidden_dim, output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        h = h + self.middle(h)
        return self.decoder(h)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_image_based_split(
    X: torch.Tensor,
    y: torch.Tensor,
    label: torch.Tensor,
    val_ratio: float,
    seed: int,
) -> Tuple[TensorDataset, TensorDataset]:
    caption_image_ids = torch.argmax(label.float(), dim=1).long()
    num_images = label.shape[1]

    generator = torch.Generator().manual_seed(seed)
    permuted_images = torch.randperm(num_images, generator=generator)

    num_val_images = max(1, int(num_images * val_ratio))
    num_train_images = num_images - num_val_images

    train_image_ids = permuted_images[:num_train_images]
    val_image_ids = permuted_images[num_train_images:]

    train_image_mask = torch.zeros(num_images, dtype=torch.bool)
    val_image_mask = torch.zeros(num_images, dtype=torch.bool)
    train_image_mask[train_image_ids] = True
    val_image_mask[val_image_ids] = True

    train_mask = train_image_mask[caption_image_ids]
    val_mask = val_image_mask[caption_image_ids]

    train_dataset = TensorDataset(
        X[train_mask],
        y[train_mask],
        caption_image_ids[train_mask],
    )
    val_dataset = TensorDataset(
        X[val_mask],
        y[val_mask],
        caption_image_ids[val_mask],
    )

    return train_dataset, val_dataset


def build_image_index_groups(image_ids: torch.Tensor) -> list[list[int]]:
    groups: dict[int, list[int]] = {}
    for idx, img_id in enumerate(image_ids.tolist()):
        groups.setdefault(int(img_id), []).append(idx)
    return list(groups.values())


def estimate_captions_per_image(image_ids: torch.Tensor) -> int:
    unique_ids, counts = torch.unique(image_ids, return_counts=True)
    if counts.numel() == 0:
        raise ValueError("No image ids found")
    avg = torch.round(counts.float().mean()).item()
    return int(avg)


class ImageGroupedBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        groups: list[list[int]],
        images_per_batch: int,
        drop_last: bool = False,
        seed: int = 0,
    ):
        self.groups = groups
        self.images_per_batch = max(1, images_per_batch)
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        if not self.groups:
            return
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        permuted = torch.randperm(len(self.groups), generator=generator).tolist()
        current_groups: list[list[int]] = []
        for group_idx in permuted:
            current_groups.append(self.groups[group_idx])
            if len(current_groups) == self.images_per_batch:
                yield [idx for group in current_groups for idx in group]
                current_groups = []
        if current_groups and not self.drop_last:
            yield [idx for group in current_groups for idx in group]

    def __len__(self) -> int:
        full = len(self.groups) // self.images_per_batch
        if not self.drop_last and len(self.groups) % self.images_per_batch != 0:
            full += 1
        return full


def build_dataloaders(
    train_dataset: TensorDataset,
    val_dataset: TensorDataset,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> Tuple[DataLoader, DataLoader]:
    train_image_ids = train_dataset.tensors[2]
    captions_per_image = estimate_captions_per_image(train_image_ids)
    images_per_batch = max(1, batch_size // max(1, captions_per_image))

    train_groups = build_image_index_groups(train_image_ids)
    train_batch_sampler = ImageGroupedBatchSampler(
        train_groups,
        images_per_batch=images_per_batch,
        drop_last=False,
        seed=seed,
    )

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader


def compute_multipositive_infonce_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    image_ids: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    norm_preds = F.normalize(preds, dim=-1)
    norm_targets = F.normalize(targets, dim=-1)

    logits = norm_preds @ norm_targets.T / tau
    target_mask = (image_ids.unsqueeze(1) == image_ids.unsqueeze(0)).float()
    positive_counts = target_mask.sum(dim=1).clamp(min=1.0)

    log_softmax = F.log_softmax(logits, dim=1)
    log_likelihood = (log_softmax * target_mask).sum(dim=1) / positive_counts
    loss_captions = -log_likelihood.mean()

    log_softmax_t = F.log_softmax(logits.T, dim=1)
    positive_counts_t = target_mask.T.sum(dim=1).clamp(min=1.0)
    log_likelihood_t = (log_softmax_t * target_mask.T).sum(dim=1) / positive_counts_t
    loss_images = -log_likelihood_t.mean()

    return 0.5 * (loss_captions + loss_images)


@torch.inference_mode()
def compute_global_mrr(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    chunk_size: int = 4096,
) -> float:
    model.eval()
    caption_preds = []
    caption_image_ids = []
    image_latent_dict: dict[int, torch.Tensor] = {}

    for text_embeds, image_latents, image_ids in val_loader:
        text_embeds = text_embeds.to(device)
        image_latents = image_latents.to(device)
        image_ids = image_ids.to(device)

        outputs = model(text_embeds)
        caption_preds.append(outputs.cpu())
        caption_image_ids.append(image_ids.cpu())

        for img_id, latent in zip(image_ids, image_latents):
            key = int(img_id.item())
            if key not in image_latent_dict:
                image_latent_dict[key] = latent.detach().cpu()

    if not caption_preds:
        return 0.0

    all_caption_preds = torch.cat(caption_preds, dim=0)
    all_caption_ids = torch.cat(caption_image_ids, dim=0).long()

    sorted_image_ids = sorted(image_latent_dict.keys())
    image_id_to_col = {img_id: idx for idx, img_id in enumerate(sorted_image_ids)}
    all_col_indices = torch.tensor(
        [image_id_to_col[int(idx)] for idx in all_caption_ids.tolist()],
        dtype=torch.long,
    )

    all_image_latents = torch.stack(
        [image_latent_dict[idx] for idx in sorted_image_ids], dim=0
    )
    all_image_latents = F.normalize(all_image_latents, dim=-1)
    if device.type == "cuda":
        all_image_latents = all_image_latents.to(device)

    reciprocal_ranks = []
    num_queries = all_caption_preds.size(0)

    for start in range(0, num_queries, chunk_size):
        end = min(start + chunk_size, num_queries)
        chunk_preds = all_caption_preds[start:end]
        chunk_cols = all_col_indices[start:end]

        if device.type == "cuda":
            chunk_preds = chunk_preds.to(device)
            chunk_cols = chunk_cols.to(device)

        chunk_preds = F.normalize(chunk_preds, dim=-1)
        sim_chunk = chunk_preds @ all_image_latents.T
        row_indices = torch.arange(sim_chunk.size(0), device=sim_chunk.device)
        correct_scores = sim_chunk[row_indices, chunk_cols]
        ranks = (sim_chunk > correct_scores.unsqueeze(1)).sum(dim=1) + 1
        reciprocal = (1.0 / ranks.float()).cpu()
        reciprocal_ranks.append(reciprocal)

    return torch.cat(reciprocal_ranks).mean().item()


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
) -> None:
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float("inf")
    best_val_mrr = -1.0

    for epoch in range(epochs):
        batch_sampler = getattr(train_loader, "batch_sampler", None)
        if hasattr(batch_sampler, "set_epoch"):
            batch_sampler.set_epoch(epoch)

        model.train()
        train_total = 0.0
        train_reg_total = 0.0
        train_nce_total = 0.0
        lambda_nce_weight = LAMBDA_NCE_MAX
        if LAMBDA_NCE_WARMUP_EPOCHS > 0:
            warmup_progress = min(1.0, (epoch + 1) / LAMBDA_NCE_WARMUP_EPOCHS)
            lambda_nce_weight = LAMBDA_NCE_MAX * warmup_progress

        for text_embeds, image_latents, image_ids in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{epochs}"
        ):
            text_embeds = text_embeds.to(device)
            image_latents = image_latents.to(device)
            image_ids = image_ids.to(device)

            optimizer.zero_grad()
            preds = model(text_embeds)
            reg_loss = F.smooth_l1_loss(preds, image_latents)
            nce_loss = compute_multipositive_infonce_loss(
                preds, image_latents, image_ids, INFO_NCE_TAU
            )
            loss = LAMBDA_REG * reg_loss + lambda_nce_weight * nce_loss
            loss.backward()
            optimizer.step()
            batch_size = text_embeds.size(0)
            train_total += loss.item() * batch_size
            train_reg_total += reg_loss.item() * batch_size
            train_nce_total += nce_loss.item() * batch_size

        num_train_samples = len(train_loader.dataset)
        train_loss = train_total / num_train_samples
        train_reg = train_reg_total / num_train_samples
        train_nce = train_nce_total / num_train_samples

        model.eval()
        val_reg_total = 0.0
        with torch.no_grad():
            for text_embeds, image_latents, _ in val_loader:
                text_embeds = text_embeds.to(device)
                image_latents = image_latents.to(device)
                preds = model(text_embeds)
                batch_loss = F.smooth_l1_loss(preds, image_latents)
                val_reg_total += batch_loss.item() * text_embeds.size(0)
        val_loss = val_reg_total / len(val_loader.dataset)
        val_mrr = compute_global_mrr(model, val_loader, device)

        print(
            f"Epoch {epoch+1}: Train Loss = {train_loss:.6f} "
            f"(reg={train_reg:.6f}, nce={train_nce:.6f}) | "
            f"Val Reg = {val_loss:.6f} | Val MRR = {val_mrr:.6f} | "
            f"λ_nce = {lambda_nce_weight:.3f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  ✓ Saved new best loss checkpoint ({val_loss:.6f})")

        if val_mrr > best_val_mrr:
            best_val_mrr = val_mrr
            Path(MODEL_PATH_MRR).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), MODEL_PATH_MRR)
            print(f"  ✓ Saved new best MRR checkpoint ({val_mrr:.6f})")


def main():
    seed_everything(RANDOM_SEED)
    print("1. Loading training data...")
    train_data = load_data("data/train/train.npz")
    X, y, label = prepare_train_data(train_data)
    print(f"   Captions: {len(X):,} | Images: {label.shape[1]:,}")

    print("2. Creating image-based split...")
    train_dataset, val_dataset = create_image_based_split(
        X, y, label, val_ratio=VAL_RATIO, seed=RANDOM_SEED
    )
    train_loader, val_loader = build_dataloaders(
        train_dataset,
        val_dataset,
        batch_size=BATCH_SIZE,
        device=DEVICE,
        seed=RANDOM_SEED,
    )
    print(
        f"   Train captions: {len(train_dataset):,} | "
        f"Val captions: {len(val_dataset):,}"
    )

    print("3. Initializing baseline MLP...")
    model = MLP().to(DEVICE)
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("\n4. Training with grouped batches + MRR evaluation...")
    train_model(model, train_loader, val_loader, DEVICE, EPOCHS, LR)

    print("\n5. Loading best checkpoint for evaluation...")
    best_eval_path = Path(MODEL_PATH_MRR)
    if not best_eval_path.exists():
        best_eval_path = Path(MODEL_PATH)
    model.load_state_dict(torch.load(best_eval_path, map_location=DEVICE))
    print(f"   Loaded checkpoint: {best_eval_path}")
    model.eval()
    final_mrr = compute_global_mrr(model, val_loader, DEVICE)
    print(f"   Final validation MRR: {final_mrr:.6f}")

    print("\n6. Generating submission from best baseline...")
    test_data = load_data("data/test/test.clean.npz")
    test_embeds = torch.from_numpy(test_data["captions/embeddings"]).float()
    with torch.no_grad():
        preds = model(test_embeds.to(DEVICE)).cpu()
    generate_submission(
        test_data["captions/ids"],
        preds,
        output_file=SUBMISSION_PATH,
    )
    print(f"✓ Best-loss model saved to: {MODEL_PATH}")
    print(f"✓ Best-MRR model saved to: {MODEL_PATH_MRR}")
    print(f"✓ Submission saved to: {SUBMISSION_PATH}")


if __name__ == "__main__":
    main()

