"""
This is the utility module for data configuration and train/val/test splits.
"""

import argparse
import os
from pathlib import Path

import pandas as pd

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

def df_split(
        df: pd.DataFrame,
        ratio: float,
        random: int = 42,
        shuffle: bool = False,
        ) -> tuple[pd.DataFrame, pd.DataFrame]:
    set1 = pd.DataFrame()
    set2 = pd.DataFrame()

    # Shuffle split
    if shuffle:
        df = df.sample(frac=1, random_state=random).reset_index(drop=True)

    # Stratified split
    label_uniques = df["label"].unique()
    for label in label_uniques:
        size = len(df[df["label"] == label])
        set1_size = int(size * ratio)
        set1 = pd.concat([set1, df[df["label"] == label][:set1_size]])
        set2 = pd.concat([set2, df[df["label"] == label][set1_size:]])

    return set1, set2

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

    if group_method:
        group = {}
        for f in checked_files:
            try:
                group_id = group_method(f)
                if group_id not in group: group[group_id] = []
                group[group_id].append(f)
            except Exception as e:
                print(f"Error processing file {f}: {e}")
        for group_id, files in group.items():
            # Fallback
            labels = {label_method(f) for f in files}
            if len(labels) > 1:
                print(f"Warning: Multiple labels found for group {group_id} ({files}): {labels}")
                continue
            data["files"].append("\n".join(files))
            data["label"].append(label_method(files[0]))
    else:
        for f in checked_files:
            data["files"].append(f)
            data["label"].append(label_method(f))
    return pd.DataFrame(data)

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
    train_df, val_df = df_split(data_df, TRAIN_RATIO, shuffle=True)
    val_df, test_df = df_split(val_df, 0.5, shuffle=True)
    
    os.mkdir(Path(OUTPUT_PATH), exist_ok=True)
    train_df.to_csv(Path(OUTPUT_PATH) / "train.csv", index=False)
    val_df.to_csv(Path(OUTPUT_PATH) / "val.csv", index=False)
    test_df.to_csv(Path(OUTPUT_PATH) / "test.csv", index=False)
    print(f"Data split completed. Files saved to {OUTPUT_PATH}")

if __name__ == "__main__":
        main()