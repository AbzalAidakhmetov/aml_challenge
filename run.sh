# 1. get the baseline model
printf "Getting the baseline model...\n"
python train_mrr.py

# 2. get the top 50 captions, drop the rest, save the filtered dataset
printf "Getting the top 50 captions, dropping the rest, saving the filtered dataset...\n"
python -m src.topk_filter_pairs

# 3. get the clip scores, drop 10% of the captions with lowest scores, save the filtered dataset
printf "Getting the clip scores, dropping 10%% of the captions with lowest scores, saving the filtered dataset...\n"
python -m src.clip_filter_pairs --input-npz data/train/train.npz \
    --images-root data/train/Images \
    --output-npz data/filtered/train_clip_q0.10_from_raw.npz \
    --scores-output data/filtered/train_clip_scores_raw.npy \
    --drop-fraction 0.10

# 4. get models with different 11 seeds on old dataset
printf "Getting models with different 11 seeds on old dataset...\n"
python train_mrr_Ensemble_old_dataset.py

# 5. get models with different 6 seeds on new dataset
printf "Getting models with different 6 seeds on new dataset...\n"
python train_mrr_Ensemble_new_dataset_CLIP.py

# 6. get models with 6 more seeds trained on new dataset, AdamW, 5k batch size
printf "Getting models with 6 more seeds trained on new dataset, AdamW, 5k batch size...\n"
python train_mrr_Ensemble_new_dataset_CLIP_AdamW_5_batches.py

# 7, [ensemble] combine resulting csv-s to get the ensemble (average of submissions.csv-s) of 16 seeds (best Kaggle public score)
printf "Combining resulting csv-s to get the ensemble (average of submissions.csv-s) of 16 seeds (best Kaggle public score)...\n"
python ensemble_combine_CSVs_16seeds.py

# 8. [ensemble] combine all 23 models (second-best Kaggle public score)
printf "Combining all 23 models (second-best Kaggle public score)...\n"
python ensemble_combine_CSVs_all_23seeds.py