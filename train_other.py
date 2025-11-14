import csv
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.common import load_data, prepare_train_data, generate_submission


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
TRAIN_DATA_PATH = Path("data/train/train.npz")
TEST_DATA_PATH = Path("data/test/test.clean.npz")

EXPERIMENT_ROOT = Path("experiments")
MODELS_DIR = EXPERIMENT_ROOT / "models"
PLOTS_DIR = EXPERIMENT_ROOT / "plots"
SUBMISSIONS_DIR = EXPERIMENT_ROOT / "submissions"
SUMMARY_PATH = EXPERIMENT_ROOT / "grid_summary.csv"

FINAL_MODEL_PATH = Path("models/translator_resmlp.pth")
FINAL_PLOT_PATH = Path("training_curves.png")
FINAL_SUBMISSION_PATH = Path("submission.csv")

INPUT_DIM = 1024
OUTPUT_DIM = 1536
VAL_RATIO = 0.2
RANDOM_SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------
# Experiment configuration dataclass & grid
# ----------------------------------------------------------------------
@dataclass
class ExperimentConfig:
    name: str
    model_width: int = 1536
    num_blocks: int = 3
    block_dropout: float = 0.1
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 30
    lambda_reg: float = 1.0
    lambda_nce_max: float = 0.5
    lambda_nce_warmup_epochs: int = 5
    info_nce_tau: float = 0.12
    images_per_batch_cap: int | None = 128
    seed: int | None = None


def build_experiment_grid() -> list[ExperimentConfig]:
    """
    Build a reasonably large experiment grid (~100 configs) so an overnight run
    keeps the GPU busy for 8-9 hours. Each config tweaks core hyperparameters.
    """

    widths = [1024, 1536, 2048]
    blocks = [1, 3]
    batch_sizes = [256, 384, 512]
    lambda_maxes = [0.3, 0.5, 0.7]
    taus = [0.10, 0.12]

    configs: list[ExperimentConfig] = []
    idx = 0
    for width in widths:
        for num_blocks in blocks:
            # Avoid very shallow large models to keep runtime sane
            if width == 2048 and num_blocks == 1:
                continue
            for batch in batch_sizes:
                for lam in lambda_maxes:
                    for tau in taus:
                        epochs = 40 if batch == 256 else 30
                        lr = 1e-3
                        if width == 1024:
                            lr = 1.2e-3
                        elif width == 2048:
                            lr = 8e-4
                        warmup = max(3, int(0.2 * epochs))
                        name = (
                            f"exp_{idx:03d}_w{width}_b{num_blocks}_bs{batch}"
                            f"_lam{int(lam*100):02d}_tau{int(tau*100):02d}"
                        )
                        configs.append(
                            ExperimentConfig(
                                name=name,
                                model_width=width,
                                num_blocks=num_blocks,
                                batch_size=batch,
                                lambda_nce_max=lam,
                                lambda_nce_warmup_epochs=warmup,
                                info_nce_tau=tau,
                                epochs=epochs,
                                lr=lr,
                                seed=RANDOM_SEED + idx,
                            )
                        )
                        idx += 1
    return configs


# ----------------------------------------------------------------------
# Utility helpers
# ----------------------------------------------------------------------
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
) -> tuple[TensorDataset, TensorDataset]:
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
        return 1
    avg = torch.round(counts.float().mean()).item()
    return max(1, int(avg))


class ImageGroupedBatchSampler(torch.utils.data.Sampler[list[int]]):
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
    cfg: ExperimentConfig,
    device: torch.device,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    train_image_ids = train_dataset.tensors[2]
    captions_per_image = estimate_captions_per_image(train_image_ids)
    images_per_batch = max(1, cfg.batch_size // max(1, captions_per_image))
    if cfg.images_per_batch_cap is not None:
        images_per_batch = min(images_per_batch, cfg.images_per_batch_cap)

    train_groups = build_image_index_groups(train_image_ids)
    train_batch_sampler = ImageGroupedBatchSampler(
        train_groups,
        images_per_batch=images_per_batch,
        drop_last=False,
        seed=seed,
    )

    pin_mem = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler,
        pin_memory=pin_mem,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=pin_mem,
    )
    return train_loader, val_loader


# ----------------------------------------------------------------------
# Model definition
# ----------------------------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, width: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.linear1 = nn.Linear(width, width * 4)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(width * 4, width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.norm(x)
        out = self.linear1(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.linear2(out)
        out = self.dropout(out)
        return residual + out


class Translator(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        width: int,
        num_blocks: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, width),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            [ResidualBlock(width, dropout=dropout) for _ in range(num_blocks)]
        )
        self.out_norm = nn.LayerNorm(width)
        self.out_proj = nn.Linear(width, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        for block in self.blocks:
            h = block(h)
        h = self.out_norm(h)
        return self.out_proj(h)


# ----------------------------------------------------------------------
# Training & evaluation
# ----------------------------------------------------------------------
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
    cfg: ExperimentConfig,
    model_path: Path,
) -> tuple[float, list[dict[str, float]]]:
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.epochs,
        eta_min=1e-5,
    )
    best_mrr = -1.0
    history: list[dict[str, float]] = []

    for epoch in range(1, cfg.epochs + 1):
        batch_sampler = getattr(train_loader, "batch_sampler", None)
        if hasattr(batch_sampler, "set_epoch"):
            batch_sampler.set_epoch(epoch - 1)

        model.train()
        total_loss = 0.0
        total_reg = 0.0
        total_nce = 0.0
        lambda_nce_weight = cfg.lambda_nce_max
        if cfg.lambda_nce_warmup_epochs > 0:
            warmup_progress = min(1.0, epoch / cfg.lambda_nce_warmup_epochs)
            lambda_nce_weight = cfg.lambda_nce_max * warmup_progress

        for text_embeds, image_latents, image_ids in tqdm(
            train_loader, desc=f"Epoch {epoch}/{cfg.epochs}"
        ):
            text_embeds = text_embeds.to(device)
            image_latents = image_latents.to(device)
            image_ids = image_ids.to(device)

            preds = model(text_embeds)
            reg_loss = F.smooth_l1_loss(preds, image_latents)
            nce_loss = compute_multipositive_infonce_loss(
                preds, image_latents, image_ids, cfg.info_nce_tau
            )
            loss = cfg.lambda_reg * reg_loss + lambda_nce_weight * nce_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = text_embeds.size(0)
            total_loss += loss.item() * batch_size
            total_reg += reg_loss.item() * batch_size
            total_nce += nce_loss.item() * batch_size

        num_train_samples = len(train_loader.dataset)
        avg_loss = total_loss / num_train_samples
        avg_reg = total_reg / num_train_samples
        avg_nce = total_nce / num_train_samples

        model.eval()
        val_reg_loss = 0.0
        with torch.no_grad():
            for text_embeds, image_latents, _ in val_loader:
                text_embeds = text_embeds.to(device)
                image_latents = image_latents.to(device)
                preds = model(text_embeds)
                val_reg_loss += (
                    F.smooth_l1_loss(preds, image_latents).item()
                    * text_embeds.size(0)
                )
        val_reg_loss /= len(val_loader.dataset)

        val_mrr = compute_global_mrr(model, val_loader, device)
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch}: train_loss={avg_loss:.4f} "
            f"(reg={avg_reg:.4f}, nce={avg_nce:.4f}) "
            f"| val_reg={val_reg_loss:.4f} | val_mrr={val_mrr:.4f} "
            f"| λ_nce={lambda_nce_weight:.3f} | lr={current_lr:.2e}"
        )

        if val_mrr > best_mrr:
            best_mrr = val_mrr
            model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), model_path)
            print(f"  ✓ Saved new best checkpoint (MRR={best_mrr:.4f})")

        scheduler.step()

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(avg_loss),
                "train_reg": float(avg_reg),
                "train_nce": float(avg_nce),
                "val_reg": float(val_reg_loss),
                "val_mrr": float(val_mrr),
                "lambda_nce": float(lambda_nce_weight),
                "lr": float(current_lr),
            }
        )

    return best_mrr, history


def plot_training_curves(
    history: list[dict[str, float]], output_path: Path, title_suffix: str | None = None
) -> None:
    if not history:
        return

    epochs = [entry["epoch"] for entry in history]
    train_loss = [entry["train_loss"] for entry in history]
    train_reg = [entry["train_reg"] for entry in history]
    train_nce = [entry["train_nce"] for entry in history]
    val_reg = [entry["val_reg"] for entry in history]
    val_mrr = [entry["val_mrr"] for entry in history]

    output_path = str(output_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax_loss, ax_mrr = axes
    ax_loss.plot(epochs, train_loss, label="Train total", color="tab:blue")
    ax_loss.plot(epochs, train_reg, label="Train reg", color="tab:green")
    ax_loss.plot(epochs, train_nce, label="Train NCE", color="tab:orange")
    ax_loss.plot(epochs, val_reg, label="Val reg", color="tab:red", linestyle="--")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    loss_title = "Loss Components"
    if title_suffix:
        loss_title = f"{loss_title} ({title_suffix})"
    ax_loss.set_title(loss_title)
    ax_loss.grid(True, linestyle="--", alpha=0.3)
    ax_loss.legend()

    ax_mrr.plot(epochs, val_mrr, marker="o", color="tab:purple")
    ax_mrr.set_xlabel("Epoch")
    ax_mrr.set_ylabel("Global MRR")
    ax_mrr.set_ylim(0.0, max(0.05, max(val_mrr) * 1.05))
    mrr_title = "Validation MRR"
    if title_suffix:
        mrr_title = f"{mrr_title} ({title_suffix})"
    ax_mrr.set_title(mrr_title)
    ax_mrr.grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"✓ Saved training curves to: {output_path}")


def run_experiment(
    cfg: ExperimentConfig,
    train_dataset: TensorDataset,
    val_dataset: TensorDataset,
    device: torch.device,
    seed: int,
    model_path: Path,
    plot_path: Path,
) -> dict:
    print(
        f"\n=== Running {cfg.name} | width={cfg.model_width} | "
        f"blocks={cfg.num_blocks} | batch={cfg.batch_size} | "
        f"λ_max={cfg.lambda_nce_max} | τ={cfg.info_nce_tau} ==="
    )
    seed_everything(seed)
    train_loader, val_loader = build_dataloaders(train_dataset, val_dataset, cfg, device, seed)

    model = Translator(
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        width=cfg.model_width,
        num_blocks=cfg.num_blocks,
        dropout=cfg.block_dropout,
    ).to(device)
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

    best_mrr, history = train_model(model, train_loader, val_loader, device, cfg, model_path)
    print(f"   Best validation MRR ({cfg.name}): {best_mrr:.4f}")

    plot_training_curves(history, output_path=plot_path, title_suffix=cfg.name)

    return {
        "config": cfg,
        "seed": seed,
        "best_mrr": best_mrr,
        "model_path": model_path,
        "plot_path": plot_path,
        "history": history,
    }


def main():
    EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)
    for directory in (MODELS_DIR, PLOTS_DIR, SUBMISSIONS_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    seed_everything(RANDOM_SEED)

    print("1. Loading training data...")
    train_data = load_data(TRAIN_DATA_PATH)
    X, y, label = prepare_train_data(train_data)
    print(f"   Captions: {len(X):,}, Images: {label.shape[1]:,}")

    print("2. Creating image-based split...")
    train_dataset, val_dataset = create_image_based_split(
        X, y, label, val_ratio=VAL_RATIO, seed=RANDOM_SEED
    )
    print(
        f"   Train captions: {len(train_dataset):,} | "
        f"Val captions: {len(val_dataset):,}"
    )

    experiments = build_experiment_grid()
    if not experiments:
        print("No experiments defined. Exiting.")
        return

    print(f"\nScheduled {len(experiments)} experiments.")
    results = []
    for idx, cfg in enumerate(experiments):
        exp_seed = cfg.seed if cfg.seed is not None else RANDOM_SEED + idx
        model_path = MODELS_DIR / f"{cfg.name}.pth"
        plot_path = PLOTS_DIR / f"{cfg.name}.png"
        result = run_experiment(
            cfg=cfg,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            device=DEVICE,
            seed=exp_seed,
            model_path=model_path,
            plot_path=plot_path,
        )
        results.append(result)

    if not results:
        print("Experiments failed to run.")
        return

    results.sort(key=lambda r: r["best_mrr"], reverse=True)
    print("\n=== Experiment Summary ===")
    print(
        f"{'Rank':<5}{'Name':<28}{'MRR':<8}{'Width':<8}{'Blocks':<8}"
        f"{'Batch':<8}{'λ_max':<8}{'τ':<6}{'Epochs':<8}"
    )
    summary_rows = []
    for rank, result in enumerate(results, start=1):
        cfg = result["config"]
        summary_rows.append(
            [
                cfg.name,
                result["best_mrr"],
                cfg.model_width,
                cfg.num_blocks,
                cfg.batch_size,
                cfg.lambda_nce_max,
                cfg.info_nce_tau,
                cfg.epochs,
            ]
        )
        print(
            f"{rank:<5}{cfg.name:<28}{result['best_mrr']:<8.4f}"
            f"{cfg.model_width:<8}{cfg.num_blocks:<8}{cfg.batch_size:<8}"
            f"{cfg.lambda_nce_max:<8.2f}{cfg.info_nce_tau:<6.2f}{cfg.epochs:<8}"
        )

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_PATH.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "name",
                "best_val_mrr",
                "model_width",
                "num_blocks",
                "batch_size",
                "lambda_nce_max",
                "info_nce_tau",
                "epochs",
            ]
        )
        writer.writerows(summary_rows)
    print(f"\n✓ Wrote summary to {SUMMARY_PATH}")

    best_result = results[0]
    best_cfg = best_result["config"]
    print(f"\n→ Best experiment: {best_cfg.name} (MRR={best_result['best_mrr']:.4f})")

    best_model = Translator(
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        width=best_cfg.model_width,
        num_blocks=best_cfg.num_blocks,
        dropout=best_cfg.block_dropout,
    ).to(DEVICE)
    best_model.load_state_dict(torch.load(best_result["model_path"], map_location=DEVICE))
    best_model.eval()

    print("\nGenerating submission with best checkpoint...")
    test_data = load_data(TEST_DATA_PATH)
    test_embeds = torch.from_numpy(test_data["captions/embeddings"]).float()
    with torch.no_grad():
        preds = best_model(test_embeds.to(DEVICE)).cpu()

    best_submission_path = SUBMISSIONS_DIR / f"{best_cfg.name}.csv"
    generate_submission(
        test_data["captions/ids"],
        preds,
        output_file=str(best_submission_path),
    )

    FINAL_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_result["model_path"], FINAL_MODEL_PATH)
    shutil.copy2(best_result["plot_path"], FINAL_PLOT_PATH)
    shutil.copy2(best_submission_path, FINAL_SUBMISSION_PATH)

    print(f"\n✓ Best model copied to: {FINAL_MODEL_PATH}")
    print(f"✓ Best plot copied to: {FINAL_PLOT_PATH}")
    print(f"✓ Submission copied to: {FINAL_SUBMISSION_PATH}")


if __name__ == "__main__":
    main()

