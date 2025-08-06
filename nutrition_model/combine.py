import os
import pandas as pd
import re

# Directory containing the CSVs
csv_dir = "extracted_tables"

# File patterns and type mapping
type_map = {
    "growing_steer_heiver": 0,
    "growing_yearlings": 1,
    "growing_mature_bulls": 2
}

# List of files to process
files = [
    "growing_steer_heiver_1200_pounds_at_finishing.csv",
    "growing_steer_heiver_1400_pounds_at_finishing.csv",
    "growing_yearlings_1100_pounds_at_finishing.csv",
    "growing_yearlings_1200_pounds_at_finishing.csv",
    "growing_yearlings_1300_pounds_at_finishing.csv",
    "growing_yearlings_1400_pounds_at_finishing.csv",
    "growing_mature_bulls_1700_pounds_at_finishing.csv",
    "growing_mature_bulls_2000_pounds_at_finishing.csv",
    "growing_mature_bulls_2300_pounds_at_finishing.csv"
]

all_dfs = []

for fname in files:
    fpath = os.path.join(csv_dir, fname)
    # Extract type and target weight from filename
    m = re.match(r"(growing_[a-z_]+)_(\d+)_pounds_at_finishing\.csv", fname)
    if not m:
        print(f"Skipping {fname}: pattern not matched")
        continue
    type_str, weight_str = m.groups()
    type_val = type_map[type_str]
    target_weight = int(weight_str)
    # Read CSV
    df = pd.read_csv(fpath)
    df["target_weight"] = target_weight
    df["type"] = type_val
    all_dfs.append(df)

# Combine all
combined = pd.concat(all_dfs, ignore_index=True)
combined.to_csv("combined_growing.csv", index=False)
print("Combined CSV saved as combined_growing.csv")