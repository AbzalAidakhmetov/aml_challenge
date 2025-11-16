import ast
import numpy as np
import pandas as pd

# -------- CONFIG --------
seeds = [42, 1337, 2025, 31415, 123456, 7, 111, 7777, 271828, 98765, 4]

# new seeds trained on filtered dataset using clip
new_seeds = [8, 26, 256, 404, 322, 4096, 323, 10000, 100001, 1234, 8888, 100002] #,  54321,  13579, 246813, 101010]

seeds += new_seeds

input_pattern = "submission_seed{}_mrr.csv"
output_path = "submission_ensemble_mean_23seeds.csv"
id_column = "id"
emb_column = "embedding"
# ------------------------


def parse_embedding(s: str) -> np.ndarray:
    """
    Turn a string like "[1.0, 2.0, ...]" into a 1D np.array of floats.
    """
    if isinstance(s, str):
        return np.array(ast.literal_eval(s), dtype=np.float32)
    elif isinstance(s, (list, tuple, np.ndarray)):
        return np.array(s, dtype=np.float32)
    else:
        raise TypeError(f"Unexpected embedding type: {type(s)}")


def format_embedding(vec: np.ndarray) -> str:
    # Use repr(float(x)) to get a long, full-precision string
    return "[" + ", ".join(repr(float(x)) for x in vec) + "]"


# 1. Load all submissions
dfs = []
for seed in seeds:
    path = input_pattern.format(seed)
    df = pd.read_csv(path)
    df = df.sort_values(id_column).reset_index(drop=True)
    dfs.append(df)
    print(f"Loaded {path} with shape {df.shape}")

# 2. Check columns & IDs match
base_cols = dfs[0].columns
for i, df in enumerate(dfs[1:], start=1):
    if not df.columns.equals(base_cols):
        raise ValueError(f"Column mismatch between submissions[0] and submissions[{i}]")

print("All column names and order match across submissions.")

base_ids = dfs[0][id_column].values
for i, df in enumerate(dfs[1:], start=1):
    if not np.array_equal(df[id_column].values, base_ids):
        raise ValueError(f"ID mismatch between submissions[0] and submissions[{i}]")

print("All ID columns match across submissions.")

# 3. Parse all embedding strings into numeric arrays and stack
emb_mats = []  # list of arrays, each (num_rows, dim)
for i, df in enumerate(dfs):
    emb_array = np.stack(df[emb_column].apply(parse_embedding).to_list(), axis=0)
    emb_mats.append(emb_array)
    print(f"Parsed embeddings for seed index {i}: shape {emb_array.shape}")

# shape: (num_seeds, num_rows, dim)
stacked = np.stack(emb_mats, axis=0)
print("Stacked embeddings shape:", stacked.shape)

# 4. Average over seeds
avg_emb = stacked.mean(axis=0)  # (num_rows, dim)

# 5. Build ensemble df preserving original structure
ensemble_df = dfs[0].copy()
ensemble_df[emb_column] = [format_embedding(row) for row in avg_emb]

# 6. Save
ensemble_df.to_csv(output_path, index=False)
print(f"Ensembled submission saved to: {output_path}")
