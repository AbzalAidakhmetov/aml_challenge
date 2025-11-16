# GRID SEARCH 
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

import json
import csv
from datetime import datetime

import itertools
import copy

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

# Similarity metric for InfoNCE loss (will be overridden by HPO)
# Options: "cosine", "euclidean", "manhattan"
SIMILARITY_METRIC_TRAIN = "cosine"

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

def compute_pairwise_logits(
    a: torch.Tensor,
    b: torch.Tensor,
    tau: float,
    metric: str = "cosine",
) -> torch.Tensor:
    """
    Compute pairwise similarity logits between a and b using a chosen metric.
    Higher logits = "more similar".
    Shapes:
        a: [N, D]
        b: [M, D]
    Returns:
        logits: [N, M]
    """
    if metric == "cosine":
        a_norm = F.normalize(a, dim=-1)
        b_norm = F.normalize(b, dim=-1)
        logits = a_norm @ b_norm.T

    elif metric == "euclidean":
        # Use negative squared Euclidean distance as similarity
        # dist^2(x, y) = ||x||^2 + ||y||^2 - 2 x·y
        a2 = (a ** 2).sum(dim=1, keepdim=True)      # [N, 1]
        b2 = (b ** 2).sum(dim=1, keepdim=True).T    # [1, M]
        dist_sq = a2 + b2 - 2.0 * (a @ b.T)         # [N, M]
        dist_sq = torch.clamp(dist_sq, min=0.0)
        logits = -dist_sq

    elif metric in ("manhattan", "l1"):
        # Negative L1 distance as similarity
        # torch.cdist is optimized and works for p=1
        logits = -torch.cdist(a, b, p=1)

    else:
        raise ValueError(f"Unknown similarity metric: {metric}")

    return logits / tau


def compute_multipositive_infonce_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    image_ids: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    
    # Use selected metric to compute logits
    logits = compute_pairwise_logits(
        preds,
        targets,
        tau=tau,
        metric=SIMILARITY_METRIC_TRAIN,
    )

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
):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float("inf")
    best_val_mrr = -1.0

    # NEW: keep track of where best checkpoints were saved
    best_loss_path = None
    best_mrr_path = None

    for epoch in range(epochs):
        batch_sampler = getattr(train_loader, "batch_sampler", None)
        if hasattr(batch_sampler, "set_epoch"):
            batch_sampler.set_epoch(epoch)

        model.train()
        train_total = 0.0
        train_reg_total = 0.0
        train_nce_total = 0.0

        # NCE warmup scheduling per epoch
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

        # Validation (regression loss)
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

        # Validation MRR (uses current model weights)
        val_mrr = compute_global_mrr(model, val_loader, device)

        print(
            f"Epoch {epoch+1}: Train Loss = {train_loss:.6f} "
            f"(reg={train_reg:.6f}, nce={train_nce:.6f}) | "
            f"Val Reg = {val_loss:.6f} | Val MRR = {val_mrr:.6f} | "
            f"λ_nce = {lambda_nce_weight:.3f}"
        )

        # Track best checkpoints and remember their paths
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), MODEL_PATH)
            best_loss_path = MODEL_PATH
            print(f"  ✓ Saved new best loss checkpoint ({val_loss:.6f})")

        if val_mrr > best_val_mrr:
            best_val_mrr = val_mrr
            Path(MODEL_PATH_MRR).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), MODEL_PATH_MRR)
            best_mrr_path = MODEL_PATH_MRR
            print(f"  ✓ Saved new best MRR checkpoint ({val_mrr:.6f})")

    # NEW: return metrics + paths so HPO can log them
    return best_val_loss, best_val_mrr, best_loss_path, best_mrr_path


# ------------------------------------------------------------------------
# Hyperparameter search configuration
# ------------------------------------------------------------------------

HP_NUM_TRIALS = 100 #30  # you can bump this up if you have more time/GPU

HP_SPACE = {
    "lr": [3e-4, 7e-4, 1e-3],
    # Careful with GPU memory; adjust if needed
    "batch_size": [1024, 2048, 3072, 4096],
    "hidden_dim": [1024, 2048, 3072, 4096],
    "lambda_reg": [0.0, 0.1],
    "lambda_nce_max": [0.5, 1.0],
    "lambda_nce_warmup_epochs": [0, 1, 3],
    "info_nce_tau": [0.005, 0.01, 0.02],
    "epochs": [50],
    "similarity_metric_train": ["cosine", "euclidean", "manhattan"],  # NEW

}
HP_RESULTS_CSV = "hpsearch_results.csv"
HP_BEST_CONFIG_JSON = "models/hpsearch_best_config.json"
HP_BEST_MODEL_PATH = "models/hpsearch_best_overall_mrr.pth"


def build_hparam_grid() -> list[dict]:
    """
    Build full grid of hyperparameter combinations from HP_SPACE.
    """
    keys = list(HP_SPACE.keys())
    values_lists = [HP_SPACE[k] for k in keys]
    grid = []
    for combo in itertools.product(*values_lists):
        cfg = dict(zip(keys, combo))
        grid.append(cfg)
    return grid

def run_single_trial(
    trial_idx: int,
    config: dict,
    train_dataset: TensorDataset,
    val_dataset: TensorDataset,
) -> dict:
    global EPOCHS, BATCH_SIZE, LR
    global LAMBDA_REG, LAMBDA_NCE_MAX, LAMBDA_NCE_WARMUP_EPOCHS, INFO_NCE_TAU
    global MODEL_PATH, MODEL_PATH_MRR, SIMILARITY_METRIC_TRAIN

    print("\n" + "=" * 60)
    print(f"TRIAL {trial_idx} CONFIG:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print("=" * 60)

    # Set per-trial random seeds
    #seed_everything(config["seed"])
    seed_everything(RANDOM_SEED)
    # Apply config to globals used by training
    EPOCHS = config["epochs"]
    BATCH_SIZE = config["batch_size"]
    LR = config["lr"]
    LAMBDA_REG = config["lambda_reg"]
    LAMBDA_NCE_MAX = config["lambda_nce_max"]
    LAMBDA_NCE_WARMUP_EPOCHS = config["lambda_nce_warmup_epochs"]
    INFO_NCE_TAU = config["info_nce_tau"]
    SIMILARITY_METRIC_TRAIN = config["similarity_metric_train"]

    # Use a single temporary path for all trials (overwritten each time)
    MODEL_PATH = "models/hpsearch/current_trial_best_loss.pth"
    MODEL_PATH_MRR = "models/hpsearch/current_trial_best_mrr.pth"

    # Build loaders for this batch size
    train_loader, val_loader = build_dataloaders(
        train_dataset,
        val_dataset,
        batch_size=BATCH_SIZE,
        device=DEVICE,
        seed=RANDOM_SEED,  # keep split comparable across trials
    )

    # Build model with chosen hidden_dim
    model = MLP(hidden_dim=config["hidden_dim"]).to(DEVICE)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {num_params:,}")

    # Train
    best_val_loss, best_val_mrr, best_loss_path, best_mrr_path = train_model(
        model, train_loader, val_loader, DEVICE, EPOCHS, LR
    )

    # Just to be safe: reload the best-MRR checkpoint before final MRR eval
    if best_mrr_path is not None and Path(best_mrr_path).exists():
        state = torch.load(best_mrr_path, map_location=DEVICE)
        model.load_state_dict(state)
        model.to(DEVICE)

    final_val_mrr = compute_global_mrr(model, val_loader, DEVICE)
    print(
        f"[TRIAL {trial_idx}] best_val_loss={best_val_loss:.6f}, "
        f"logged_best_val_MRR={best_val_mrr:.6f}, "
        f"recomputed_final_MRR={final_val_mrr:.6f}"
    )

    result = {
        "trial_idx": trial_idx,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "num_params": num_params,
        "best_val_loss": float(best_val_loss),
        "best_val_mrr": float(best_val_mrr),
        "final_val_mrr": float(final_val_mrr),
        "best_loss_path": best_loss_path,
        "best_mrr_path": best_mrr_path,
    }
    # merge hyperparams into result
    result.update(config)
    return result


def hyperparam_search():
    # 1. Data + split (done once)
    seed_everything(RANDOM_SEED)
    print("1. Loading training data for HPO...")
    train_data = load_data("data/train/train.npz")
    X, y, label = prepare_train_data(train_data)
    print(f"   Captions: {len(X):,} | Images: {label.shape[1]:,}")

    print("2. Creating image-based split (fixed for all trials)...")
    train_dataset, val_dataset = create_image_based_split(
        X, y, label, val_ratio=VAL_RATIO, seed=RANDOM_SEED
    )
    print(
        f"   Train captions: {len(train_dataset):,} | "
        f"Val captions: {len(val_dataset):,}"
    )

    grid = build_hparam_grid()
    print(f"Total hyperparameter combinations in grid: {len(grid)}")

    # Optional cap using HP_NUM_TRIALS
    if HP_NUM_TRIALS is None:
        max_trials = len(grid)
    else:
        max_trials = min(HP_NUM_TRIALS, len(grid))
    print(f"Running {max_trials} trial(s) out of the grid")
    
    results = []
    global_best_mrr = -1.0
    global_best_result = None

    for trial in range(max_trials):
        config = grid[trial]
        result = run_single_trial(trial, config, train_dataset, val_dataset)
        results.append(result)

        # Track global best by MRR (use recomputed final MRR)
        trial_best_mrr = result["final_val_mrr"]
        if trial_best_mrr > global_best_mrr:
            global_best_mrr = trial_best_mrr
            global_best_result = result

            print(
                f"\n>>> NEW GLOBAL BEST MRR: {global_best_mrr:.6f} "
                f"(trial {trial})"
            )

            # Save a copy of the best model weights
            if result["best_mrr_path"] is not None:
                Path(HP_BEST_MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
                state = torch.load(result["best_mrr_path"], map_location="cpu")
                torch.save(state, HP_BEST_MODEL_PATH)

            # Save best config as JSON
            best_cfg = {
                "global_best_mrr": global_best_mrr,
                "trial_idx": trial,
                "config": {k: v for k, v in result.items() if k in HP_SPACE or k == "seed"},
                "paths": {
                    "best_mrr_path": result["best_mrr_path"],
                    "best_loss_path": result["best_loss_path"],
                    "best_model_copy": HP_BEST_MODEL_PATH,
                },
            }
            with open(HP_BEST_CONFIG_JSON, "w") as f:
                json.dump(best_cfg, f, indent=2)
            print(f"    ✓ Saved best config to {HP_BEST_CONFIG_JSON}")

        # Append results to CSV incrementally (so you don’t lose progress)
        if len(results) == 1:
            # Write header
            with open(HP_RESULTS_CSV, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerow(results[0])
        else:
            with open(HP_RESULTS_CSV, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writerow(results[-1])

        print(f"   Logged trial {trial} to {HP_RESULTS_CSV}")

    # 3. After all trials, print and use the best config to make a submission
    if global_best_result is None:
        print("No successful trials in hyperparam search.")
        return

    print("\n" + "=" * 60)
    print("HYPERPARAM SEARCH COMPLETED")
    print(f"Best MRR: {global_best_mrr:.6f}")
    print("Best trial result:")
    for k, v in global_best_result.items():
        if k in ("best_loss_path", "best_mrr_path", "timestamp"):
            continue
        print(f"  {k}: {v}")
    print("=" * 60)

    # 4. Generate submission from the best model (no interruption of HPO; done at the end)
    print("\nGenerating submission with best HPO model...")
    best_hidden_dim = global_best_result["hidden_dim"]

    # Load best model
    best_model = MLP(hidden_dim=best_hidden_dim).to(DEVICE)
    state = torch.load(HP_BEST_MODEL_PATH, map_location=DEVICE)
    best_model.load_state_dict(state)
    best_model.eval()

    test_data = load_data("data/test/test.clean.npz")
    test_embeds = torch.from_numpy(test_data["captions/embeddings"]).float().to(DEVICE)

    with torch.no_grad():
        preds = best_model(test_embeds).cpu()

    submission_file = "submission_hpo_best.csv"
    generate_submission(
        test_data["captions/ids"],
        preds,
        output_file=submission_file,
    )
    print(f"✓ HPO best model saved to: {HP_BEST_MODEL_PATH}")
    print(f"✓ HPO best config saved to: {HP_BEST_CONFIG_JSON}")
    print(f"✓ HPO submission saved to: {submission_file}")



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
    best_val_loss, best_val_mrr, _, _ = train_model(
        model, train_loader, val_loader, DEVICE, EPOCHS, LR
    )
    print(
        f"\n[Baseline] Best val loss = {best_val_loss:.6f}, "
        f"best val MRR = {best_val_mrr:.6f}"
    )

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
    # main()             # <- baseline training + submission
    hyperparam_search()  # <- HPO loop