"""
This is the utility module for data configuration and train/val splits.
"""

import argparse
import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

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
    label = label_from_icnale(file_path)
    splits = basename.split("_")
    return len(splits) == 6 and (splits[-1][1] != "(") and label[:2] != "XX"

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

def id_from_icnale(file_path: str):
    basename = os.path.basename(file_path)
    basename = basename.replace(".mp3", "").replace(".wav", "").replace(".txt", "")
    return basename

def create_dataframe_from_files(
        files: list[str],
        label_method: callable,
        group_method: callable = None
        ) -> pd.DataFrame:
    data = {
        "ids": [],
        "audio_path": [],
        "text_path": [],
        "label": [],
    }
    
    data_dict = {}

    for f in files:
        id = id_from_icnale(f)
        if id not in data_dict:
            data_dict[id] = {
                "audio_path": None,
                "text_path": None,
                "label": label_from_icnale(f)
            }
        name, ext = os.path.splitext(f)
        if ext in [".mp3", ".wav"]:
            data_dict[id]["audio_path"] = f
        elif ext == ".txt":
            data_dict[id]["text_path"] = f

    for id, value in data_dict.items():
        if value["audio_path"] is None or value["text_path"] is None:
            continue
        data["ids"].append(id)
        data["audio_path"].append(value["audio_path"])
        data["text_path"].append(value["text_path"])
        data["label"].append(value["label"])

    df = pd.DataFrame(data)

    if group_method is not None: df["groups"] = df["ids"].apply(group_method)

    return df

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Setup data configuration for ICNALE-SM dataset.")
    parser.add_argument("--data-path", type=str, required=True, help="Path to the ICNALE-SM dataset.")
    parser.add_argument("--output-path", type=str, required=True, help="Path to save the processed dataset.")
    parser.add_argument("--train-ratio", type=float, default=0.9, help="Train split ratio.")
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
    walked_files = [f for f in walked_files if check_from_icnale(f)]

    data_df = create_dataframe_from_files(
        walked_files,
        label_method=label_from_icnale,
        group_method=group_by_id,
    )

    train_df, val_df = train_test_split(
        data_df,
        train_size=TRAIN_RATIO,
        shuffle=True,
        random_state=42
    )

    train_df.to_csv(os.path.join(OUTPUT_PATH, "train_common.csv"), index=False)
    val_df.to_csv(os.path.join(OUTPUT_PATH, "val_common.csv"), index=False)
    print(f"Data split completed. Files saved to {OUTPUT_PATH}")

if __name__ == "__main__":
        main()