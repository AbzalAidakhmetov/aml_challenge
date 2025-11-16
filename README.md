# AML Challenge Solution

Minimal starter for the AML image–text retrieval task: preprocess captions/images, filter noisy pairs, train the MLP translator, and export submissions that score on MRR/recall@K.

## Setup

```bash
bash setup.sh 
```

## Workflow

1. **CLIP filtering**  
   `python -m src.clip_filter_pairs   --input-npz data/train/train.npz   --images-root data/train/Images   --output-npz data/filtered/train_clip_q0.10_from_raw.npz   --scores-output data/filtered/train_clip_scores_raw.npy   --drop-fraction 0.10`

2. **Train & submit**  
   `python train_mrr.py`  
   Saves checkpoints in `models/` and a submission CSV (`submission_mrr_bs_5k_seed_100002.csv`).

## Key Files

| File | Purpose |
| --- | --- |
| `src/clip_filter_pairs.py` | OpenCLIP scoring + dataset pruning. |
| `train_mrr.py` | Residual MLP + InfoNCE baseline, grouped batches, submission export. |
| `train_mrr_hpp_tunning_RandSearch.py` | Randomized HPO driver (logs to `hpsearch_results.csv`, checkpoints under `models/hpsearch/`). |
| `src/eval/` | Metrics, retrieval evaluator, visualization helpers. |


