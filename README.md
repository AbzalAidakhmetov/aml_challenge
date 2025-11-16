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
3. **Train Models for Ensemble**
   - Re-run `python train_mrr.py` on the filtered dataset.
   - Checkpoints land in `models/` and the submission CSV defaults to `submission_mrr_baseline.csv`.
4. **Multi-seed training for ensembles**
   - `train_mrr_Ensemble_old_dataset.py`: run the same model on the **original raw dataset** with 11 different seeds to capture variance.
   - `train_mrr_Ensemble_new_dataset_CLIP.py`: runs training on the **CLIP-filtered dataset** with differenet 6 seeds.
   - `train_mrr_Ensemble_new_dataset_CLIP_AdamW_5_batches.py`: train again on the CLIP-filtered dataset but with **batch_size = 5000** and **AdamW**, using 6 more seeds.
5. **Ensemble submissions**
   - `ensemble_combine_CSVs_16seeds.py`: averages the predictions from a selected set of **16 seed runs** and writes `submission_ensemble_mean_16seeds.csv` (best Kaggle score).
   - `ensemble_combine_CSVs_all_23seeds.py`: averages predictions from **all 23 trained seeds** and writes `submission_ensemble_mean_23seeds.csv` (second-best Kaggle score).
   - A summary table of all 23 models (seed, config, validation MRR) is provided below for easy comparison.

## Key Files

| File | Purpose |
| --- | --- |
| `src/clip_filter_pairs.py` | OpenCLIP scoring + dataset pruning. |
| `train_mrr.py` | Residual MLP + InfoNCE baseline, grouped batches, submission export. |
| `train_mrr_hpp_tunning_RandSearch.py` | Randomized HPO driver (logs to `hpsearch_results.csv`, checkpoints under `models/hpsearch/`). |
| `train_mrr_Ensemble_old_dataset.py` | Multi-seed training (11 seeds) on the original raw dataset. |
| `train_mrr_Ensemble_new_dataset_CLIP.py` | Multi-seed training (6 seeds) on the CLIP-filtered dataset. |
| `train_mrr_Ensemble_new_dataset_CLIP_AdamW_5_batches.py` | Multi-seed training (6 seeds) on CLIP-filtered data with batch_size=5000 and AdamW. |
| `ensemble_combine_CSVs_16seeds.py` | Build `submission_ensemble_16seeds.csv` by averaging predictions from 16 selected seeds (best Kaggle score). |
| `ensemble_combine_CSVs_all_23seeds.py` | Build `submission_ensemble_23seeds.csv` by averaging predictions from all 23 seeds (second-best Kaggle score). |


## Model inventory and validation scores

The table below lists all trained models, their seeds, validation MRR, and whether they were included in the 16-seed ensemble (`submission_ensemble_16seeds.csv`).

| # | Script                                                      | Seed  | Dataset / Config                     | Val MRR | In 16-seed ensemble? |
|---|-------------------------------------------------------------|:-----:|--------------------------------------|:-------:|:--------------------:|
| 1 | `train_mrr_Ensemble_old_dataset.py`                         | 42    | Old dataset                          | None    | ✓                    |
| 2 | `train_mrr_Ensemble_old_dataset.py`                         | 1337  | Old dataset                          | None    | ✓                    |
| 3 | `train_mrr_Ensemble_old_dataset.py`                         | 2025  | Old dataset                          | 0.568535| ✓                    |
| 4 | `train_mrr_Ensemble_old_dataset.py`                         | 31415 | Old dataset                          | 0.578568| ✓                    |
| 5 | `train_mrr_Ensemble_old_dataset.py`                         | 123456| Old dataset                          | 0.554589| ✓                    |
| 6 | `train_mrr_Ensemble_old_dataset.py`                         | 7     | Old dataset                          | 0.567784| ✓                    |
| 7 | `train_mrr_Ensemble_old_dataset.py`                         | 111   | Old dataset                          | 0.567888| ✓                    | 
| 8 | `train_mrr_Ensemble_old_dataset.py`                         | 7777  | Old dataset                          | 0.567649| ✓                    |
| 9 | `train_mrr_Ensemble_old_dataset.py`                         | 271828| Old dataset                          | 0.571855| ✓                    |
| 10| `train_mrr_Ensemble_old_dataset.py`                         | 98765 | Old dataset                          | 0.573338| ✓                    |
| 11| `train_mrr_Ensemble_old_dataset.py`                         | 4     | Old dataset                          | 0.567313| ✓                    |
| 12| `train_mrr_Ensemble_new_dataset_CLIP.py`                    | 8     | CLIP-filtered                        | 0.587656| ✓                    |
| 13| `train_mrr_Ensemble_new_dataset_CLIP.py`                    | 26    | CLIP-filtered                        | 0.593147| ✓                    |
| 14| `train_mrr_Ensemble_new_dataset_CLIP.py`                    | 256   | CLIP-filtered                        | 0.588814| ✓                    |
| 15| `train_mrr_Ensemble_new_dataset_CLIP.py`                    | 4096  | CLIP-filtered                        | 0.598578| ✗                    |
| 16| `train_mrr_Ensemble_new_dataset_CLIP.py`                    | 8888  | CLIP-filtered                        | 0.598682| ✗                    |
| 17| `train_mrr_Ensemble_new_dataset_CLIP.py`                    | 1234  | CLIP-filtered                        | 0.600019| ✗                    |
| 18| `train_mrr_Ensemble_new_dataset_CLIP_AdamW_5_batches.py`    | 42    | CLIP-filtered, bs=5000, AdamW        | 0.594952| ✓                    |
| 19| `train_mrr_Ensemble_new_dataset_CLIP_AdamW_5_batches.py`    | 322   | CLIP-filtered, bs=5000, AdamW        | 0.595863| ✓                    |
| 20| `train_mrr_Ensemble_new_dataset_CLIP_AdamW_5_batches.py`    | 323   | CLIP-filtered, bs=5000, AdamW        | 0.586454| ✗                    |
| 21| `train_mrr_Ensemble_new_dataset_CLIP_AdamW_5_batches.py`    | 10000 | CLIP-filtered, bs=5000, AdamW        | 0.593430| ✗                    |
| 22| `train_mrr_Ensemble_new_dataset_CLIP_AdamW_5_batches.py`    | 100001| CLIP-filtered, bs=5000, AdamW        | 0.593381| ✗                    |
| 23| `train_mrr_Ensemble_new_dataset_CLIP_AdamW_5_batches.py`    | 100002| CLIP-filtered, bs=5000, AdamW        | 0.593694| ✗                    |