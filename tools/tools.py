from __future__ import annotations

import argparse

_useful_commands = (
    ("python3 -m tools.tools", "Show this help message"),
    ("python3 -m tools.monitor --ext .err .out  --select latest --n 1 --cat", "Show latest .err and .out files"),
    ("python3 -m tools.monitor --select latest --n 5", "Show 5 most recent log files under outputs/ and logs/"),
    ("python3 -m tools.monitor --select oldest --n 5", "Show 5 oldest log files under outputs/ and logs/"),
    ("python3 -m tools.eda --src path/to/your.csv", "Show basic statistics of a CSV file"),
)

# ---------------- Tools ----------------

def main():
    parser = argparse.ArgumentParser(description="A set of useful tools for data processing and model evaluation.")
    args = parser.parse_args()
    
    print("Available commands:")
    for cmd, desc in _useful_commands:
        print(f"{cmd}\n     - {desc}")

if __name__ == "__main__":
    main()