import argparse
import os
import pandas as pd

from sklearn.model_selection import train_test_split

# Recursively find files in a folder tree
def dig_folder(file, valid_exts=[".mp3", ".wav", ".flac"]):
    returning = []
    if os.path.isdir(file):
        for f in os.listdir(file):
            returning.extend(dig_folder(os.path.join(file, f), valid_exts))
    else:
        ext = os.path.splitext(file)[1].lower()
        if ext in valid_exts:
            returning.append(file)
        else:
            print(f"Skipping non-audio file: {file}")
    return returning

# Create a data configuration DataFrame
def create_data_config(prefix, cefr_label_df):
    print(f"Creating data config from {prefix}")
    paths, labels = [], []
    for f in dig_folder(prefix):
        basename = os.path.basename(f)
        label = basename.split("_")[-2] + "_" + basename.split("_")[-1][0]
        if label in cefr_label_df["CEFR Level"].values:
            paths.append(f)
            labels.append(label)
    df = pd.DataFrame({
        'path': paths,
        'label': labels
    })
    print(f"Found {len(df)} audio files for training/evaluation.")
    return df

def main():
    # Parse CLI arguments
    parser = argparse.ArgumentParser(description="Create a data configuration for ICNALE-SM dataset.")
    parser.add_argument("--dir", type=str, required=True, help="Path to the ICNALE-SM dataset directory.")
    parser.add_argument("--cefr-labels", type=str, required=True, help="Path to the CEFR labels CSV file.")
    parser.add_argument("--save", type=str, required=True, help="Path to save the data configuration CSV file.")
    args = parser.parse_args()

    prefix_dir = args["dir"]
    cefr_labels_path = args["cefr_labels"]
    saving_path = args["save"]

    # Create data configuration
    data_config = create_data_config(prefix_dir, pd.read_csv(cefr_labels_path))

    train_config, val_config = train_test_split(
        data_config, test_size=0.2, random_state=42, stratify=data_config['label']
    )
    
    train_config.to_csv(os.path.join(saving_path, "train_config.csv"), index=False)
    val_config.to_csv(os.path.join(saving_path, "val_config.csv"), index=False)


if __name__ == "__main__":
    main()