"""
ICNALE CV split utilities.

Exposes `serpete` to create stratified group k-fold splits from the
ICNALE dataset with minimal side effects.
"""

import os
from pathlib import Path
from typing import Callable, Iterable, Dict, List

import pandas as pd

def walk_folder(folder_path: Path, only_files: bool = True) -> list[str]:
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
    """
    Heuristic filter to keep valid ICNALE files only.
    """
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
    """
    Create a multimodal dataframe from paired ICNALE files. Ensures each sample has both audio and text paths.
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

def get_valid_files(
        src: Path | str,
        data_exts: Iterable[str] = (".mp3", ".wav", ".txt")
) -> list[str]:
    # Return the valid ICNALE files path from the src folder
    src = Path(src)
    files = walk_folder(src, only_files=True)
    files = [f for f in files if f.endswith(tuple(data_exts))]
    files = [f for f in files if check_from_icnale(f)]

    # Ensure each group has all 12 files (6 audio + 6 text)
    group: Dict[str, List[str]] = {}
    for f in files:
        gid = group_by_id(f)
        if gid not in group:
            group[gid] = []
        group[gid].append(f)
    files = [f for f in files if len(group[group_by_id(f)]) == 12]
    
    return files