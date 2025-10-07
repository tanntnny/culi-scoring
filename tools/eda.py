from __future__ import annotations

import os

import argparse
import pandas as pd
from typing import Protocol


class BaseAnalyzer(Protocol):
    def summarize(self) -> None:
        """Summarize the data source."""
        ...

class CSVAnalyzer(BaseAnalyzer):
    def __init__(self, src: str):
        self.src = src
        self.data = self._load_data(src)
    
    def _load_data(self, src: str) -> pd.DataFrame:
        if src.endswith('.csv'):
            return pd.read_csv(src)
        else:
            raise ValueError("Currently only CSV files are supported.")
    
    def summarize(self) -> None:
        print("Data Summary:")
        print(self.data.info())
        print("\nFirst 5 Rows:")
        print(self.data.head())
        print("\nMissing Values:")
        print(self.data.isnull().sum())
        print("\nStatistical Summary:")
        print(self.data.describe())

# ---------------- Main ----------------

def main():
    parser = argparse.ArgumentParser(description="Exploratory Data Analysis Tool")
    parser.add_argument("--src", type=str, required=True, help="Source file or directory to analyze")
    args = parser.parse_args()
    
    src = args.src
    analyzer: BaseAnalyzer = None

    _, ext = os.path.splitext(src)
    if ext in ['.csv']:
        analyzer = CSVAnalyzer(src)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")
    
    analyzer.summarize()

if __name__ == "__main__":
    main()