"""
This is the utility module for data configuration and train/val splits.
"""

import argparse
import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

def walk_folder(folder_path: Path, only_files: bool = True):
    walked_files = []
    for root, dirs, files in os.walk(folder_path, topdown=True, onerror=None, followlinks=False):
        if only_files:
            for f in files:
                walked_files.append(os.path.join(root, f))
        else:
            for d in dirs:
                walked_files.append(os.path.join(root, d))
    return walked_files

def check_from_icnale(file_path: str):
    basename = os.path.basename(file_path)
    splits = basename.split("_")
    return len(splits) == 6

def group_by_id(file_path: str):
    basename = os.path.basename(file_path)
    splits = basename.split("_")
    id = splits[0] + splits[1] + splits[3]
    return id

def label_from_icnale(file_path: str):
    basename = os.path.basename(file_path)
    splits = basename.split("_")
    label = splits[-2] + splits[-1][0]
    return label

def create_dataframe_from_files(
        files: list[str],
        label_method: callable,
        check_method: callable = None,
        group_method: callable = None
        ) -> pd.DataFrame:
    data = {
        "files": [],
        "label": [],
    }
    
    checked_files = []

    if check_method is not None: checked_files = [f for f in files if check_method(f)]
    else: checked_files = files

    for f in checked_files:
        data["files"].append(f)
        data["label"].append(label_method(f))

    df = pd.DataFrame(data)

    if group_method is not None: df["groups"] = df["files"].apply(group_method)

    return df

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Setup data configuration for ICNALE-SM dataset.")
    parser.add_argument("--data-path", type=str, required=True, help="Path to the ICNALE-SM dataset.")
    parser.add_argument("--output-path", type=str, required=True, help="Path to save the processed dataset.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio.")
    parser.add_argument("--data-ext", type=str, default=".mp3,.wav,.txt", help="Comma-separated list of file extensions to include.")
    args = parser.parse_args()

    # CONSTANTS
    DATA_PATH = args.data_path
    OUTPUT_PATH = args.output_path
    TRAIN_RATIO = args.train_ratio
    DATA_EXT = args.data_ext

    # Main
    print(f"------------------- Arguments -------------------")
    print(f"Data Path: {DATA_PATH}")
    print(f"Output Path: {OUTPUT_PATH}")
    print(f"Train Ratio: {TRAIN_RATIO}")
    print(f"Data Extensions: {DATA_EXT}")
    print(f"-------------------------------------------------")

    walked_files = walk_folder(Path(DATA_PATH), only_files=True)
    walked_files = [f for f in walked_files if f.endswith(tuple(DATA_EXT.split(",")))]
    data_df = create_dataframe_from_files(walked_files, check_method=check_from_icnale, group_method=group_by_id, label_method=label_from_icnale)
    folds = StratifiedGroupKFold(
        data_df,
        group_col="groups",
        label_col="label",
        n_splits=5,
        random_state=42
    )

    iters = folds.split(
        data_df["files"],
        data_df["label"],
        groups=data_df["groups"]
    )

    for i, (train_idx, val_idx) in enumerate(iters):
        train_df = data_df.iloc[train_idx]
        val_df = data_df.iloc[val_idx]
        train_df.to_csv(Path(OUTPUT_PATH) / f"train_fold_{i}.csv", index=False)
        val_df.to_csv(Path(OUTPUT_PATH) / f"val_fold_{i}.csv", index=False)
        print(f"Fold {i}: train={len(train_df)}, val={len(val_df)}")

    print(f"Data split completed. Files saved to {OUTPUT_PATH}")

if __name__ == "__main__":
        main()