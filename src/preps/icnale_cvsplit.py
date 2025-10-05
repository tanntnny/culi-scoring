"""
ICNALE CV split utilities.

Exposes `serpete` to create stratified group k-fold splits from the
ICNALE dataset with minimal side effects.
"""

import os
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

def walk_folder(folder_path: Path, only_files: bool = True) -> list[str]:
    """Recursively collect files (or dirs) under `folder_path`.

    - When `only_files` is True, returns file paths; otherwise directory paths.
    """
    walked: list[str] = []
    for root, dirs, files in os.walk(folder_path, topdown=True, onerror=None, followlinks=False):
        if only_files:
            for f in files:
                walked.append(os.path.join(root, f))
        else:
            for d in dirs:
                walked.append(os.path.join(root, d))
    return walked

def check_from_icnale(file_path: str) -> bool:
    """Heuristic filter to keep valid ICNALE files only."""
    basename = os.path.basename(file_path)
    label = label_from_icnale(file_path)
    splits = basename.split("_")
    return len(splits) == 6 and (splits[-1][1] != "(") and label[:2] != "XX"

def group_by_id(file_path: str) -> str:
    basename = os.path.basename(file_path)
    splits = basename.split("_")
    id = splits[0] + splits[1] + splits[3]
    return id

def label_from_icnale(file_path: str) -> str:
    basename = os.path.basename(file_path)
    splits = basename.split("_")
    label = splits[-2] + splits[-1][0]
    return label

def id_from_icnale(file_path: str) -> str:
    basename = os.path.basename(file_path)
    basename = basename.replace(".mp3", "").replace(".wav", "").replace(".txt", "")
    return basename

def create_dataframe_from_files(
    files: Iterable[str],
    label_method: Callable[[str], str],
    group_method: Callable[[str], str] | None = None,
) -> pd.DataFrame:
    """Create a multimodal dataframe from paired ICNALE files.

    Ensures each sample has both audio and text paths.
    """
    data_dict: dict[str, dict[str, str | None]] = {}

    for f in files:
        sample_id = id_from_icnale(f)
        if sample_id not in data_dict:
            data_dict[sample_id] = {"audio_path": None, "text_path": None, "label": label_method(f)}
        _, ext = os.path.splitext(f)
        if ext in (".mp3", ".wav"):
            data_dict[sample_id]["audio_path"] = f
        elif ext == ".txt":
            data_dict[sample_id]["text_path"] = f

    rows = {
        "ids": [],
        "audio_path": [],
        "text_path": [],
        "label": [],
    }
    for sid, value in data_dict.items():
        if value["audio_path"] is None or value["text_path"] is None:
            continue
        rows["ids"].append(sid)
        rows["audio_path"].append(value["audio_path"])  # type: ignore[arg-type]
        rows["text_path"].append(value["text_path"])    # type: ignore[arg-type]
        rows["label"].append(value["label"])            # type: ignore[arg-type]

    df = pd.DataFrame(rows)
    if group_method is not None:
        df["groups"] = df["ids"].apply(group_method)
    return df
def serpete(
    data_path: Path | str,
    output_path: Path | str,
    data_ext: str = ".mp3,.wav,.txt",
    n_splits: int = 5,
    random_state: int = 42,
    require_full_group: bool = True,
) -> None:
    """Create stratified group k-fold CSV splits for ICNALE.

    Saves `train_fold_{i}.csv` and `val_fold_{i}.csv` into `output_path`.
    """
    data_path = Path(data_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    walked_files = walk_folder(data_path, only_files=True)
    exts = tuple(x.strip() for x in data_ext.split(",") if x)
    if exts:
        walked_files = [f for f in walked_files if f.endswith(exts)]
    walked_files = [f for f in walked_files if check_from_icnale(f)]

    if require_full_group:
        grp: dict[str, list[str]] = {}
        for f in walked_files:
            gid = group_by_id(f)
            if gid not in grp:
                grp[gid] = []
            grp[gid].append(f)
        walked_files = [f for f in walked_files if len(grp[group_by_id(f)]) == 12]

    data_df = create_dataframe_from_files(
        walked_files,
        group_method=group_by_id,
        label_method=label_from_icnale,
    )

    folds = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for i, (train_idx, val_idx) in enumerate(
        folds.split(data_df["ids"], data_df["label"], groups=data_df["groups"])):
        train_df = data_df.iloc[train_idx]
        val_df = data_df.iloc[val_idx]
        train_df.to_csv(output_path / f"train_fold_{i}.csv", index=False)
        val_df.to_csv(output_path / f"val_fold_{i}.csv", index=False)


__all__ = [
    "walk_folder",
    "check_from_icnale",
    "group_by_id",
    "label_from_icnale",
    "id_from_icnale",
    "create_dataframe_from_files",
    "serpete",
]