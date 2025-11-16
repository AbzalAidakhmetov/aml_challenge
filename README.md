# AML Challenge Solution

Minimal starter for the AML image–text retrieval task: preprocess captions/images, filter noisy pairs, train the MLP translator, and export submissions that score on MRR/recall@K.

## Setup

```bash
bash setup.sh 
```

## Workflow
1. **Baseline run**
   - Execute `python train_mrr.py` on the raw dataset to establish the baseline model, that will be further used in `visualize_split_top_50.ipynb` for splitting.
2. **Dataset preparation**
   - *Method 1 – Top‑k filtering*: run `visualize_split_top_50.ipynb` (after the baseline model is availabel) to review captions per image, keep the strongest pairs, and export the filtered NPZ.
   - *Method 2 – CLIP filtering*:  
     `python -m src.clip_filter_pairs --input-npz data/train/train.npz --images-root data/train/Images --output-npz data/filtered/train_clip_q0.10_from_raw.npz --scores-output data/filtered/train_clip_scores_raw.npy --drop-fraction 0.10`
3. **Train & submit**
   - Re-run `python train_mrr.py` on the filtered dataset.
   - Checkpoints land in `models/` and the submission CSV defaults to `submission_mrr_baseline.csv`.

## Key Files

| File | Purpose |
| --- | --- |
| `src/clip_filter_pairs.py` | OpenCLIP scoring + dataset pruning. |
| `train_mrr.py` | Residual MLP + InfoNCE baseline, grouped batches, submission export. |
| `train_mrr_hpp_tunning_RandSearch.py` | Randomized HPO driver (logs to `hpsearch_results.csv`, checkpoints under `models/hpsearch/`). |

