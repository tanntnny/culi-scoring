from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

def parse_time(s: str) -> float:
    """Accept 'YYYY-MM-DD', ISO8601 like '2025-10-06T12:30', or epoch seconds."""
    try:
        # epoch seconds
        return float(s)
    except ValueError:
        pass
    try:
        # date only -> start of day
        if len(s) == 10:
            return dt.datetime.fromisoformat(s).timestamp()
        # datetime (naive -> local)
        return dt.datetime.fromisoformat(s).timestamp()
    except Exception:
        raise argparse.ArgumentTypeError(f"Invalid time: {s}")

def normalize_exts(exts: Sequence[str]) -> Tuple[str, ...]:
    # Accept with/without dot; case-insensitive; unique; keep input order
    seen = set()
    norm = []
    for e in exts:
        e = e.strip()
        if not e:
            continue
        e = e.lower()
        if not e.startswith("."):
            e = "." + e
        if e not in seen:
            seen.add(e)
            norm.append(e)
    return tuple(norm)

def iter_files(paths: Iterable[Path]) -> Iterable[Path]:
    for root in paths:
        try:
            if root.is_file():
                yield root
                continue
            if not root.exists():
                print(f"Directory does not exist: {root}", file=sys.stderr)
                continue
        except PermissionError:
            print(f"Permission denied: {root}", file=sys.stderr)
            continue

        # Path.rglob handles recursion neatly; follow_symlinks=False to avoid loops
        try:
            yield from (p for p in root.rglob("*") if p.is_file())
        except PermissionError:
            print(f"Permission denied while walking: {root}", file=sys.stderr)

def pick(files: List[Path], n: int, select: str) -> List[Path]:
    key = lambda p: p.stat().st_mtime
    reverse = (select == "latest")
    return sorted(files, key=key, reverse=reverse)[:n]

def human_time(ts: float) -> str:
    return dt.datetime.fromtimestamp(ts).isoformat(sep=" ", timespec="seconds")

_printable = [".out", ".log", ".txt", ".err"]

def main() -> int:
    parser = argparse.ArgumentParser(description="Monitor recent/oldest files under paths by extension.")
    parser.add_argument(
        "--select", 
        choices=["latest", "oldest"],
        default="latest",
        help="Choose newest or oldest files per extension (default: latest)."
    )
    parser.add_argument(
        "--src",
        nargs="*",
        default=["outputs", "logs"],
        help="Source directories/files to scan."
    )
    parser.add_argument(
        "--ext",
        nargs="*",
        default=[".txt", ".log", ".err", ".out"],
        help="File extensions to monitor ('.log' or 'log')."
    )
    parser.add_argument(
        "--n",
        type=int,
        default=3,
        help="How many files per extension to show."
    )
    parser.add_argument(
        "--since",
        type=parse_time,
        help="Only include files modified after this time."
    )
    parser.add_argument(
        "--until",
        type=parse_time,
        help="Only include files modified before this time."
    )
    parser.add_argument(
        "--max-char",
        type=int,
        default=None,
        help="Maximum characters per line when printing file contents; truncate with '...' if exceeded."
    )
    parser.add_argument(
        "--cat", "-c",
        action="store_true",
        help="Print the contents of the listed files."
    )
    args = parser.parse_args()

    src_paths = [Path(s) for s in args.src]
    exts = normalize_exts(args.ext)

    # Collect
    all_files = []
    for p in iter_files(src_paths):
        try:
            all_files.append(p)
        except PermissionError:
            print(f"Permission denied: {p}", file=sys.stderr)

    # Filter by ext and time
    since = args.since if args.since is not None else float("-inf")
    until = args.until if args.until is not None else float("inf")

    by_src_ext = {}
    for f in all_files:
        try:
            mtime = f.stat().st_mtime
            if since <= mtime <= until:
                ext = f.suffix.lower()
                if ext in exts:
                    for src in src_paths:
                        if f.is_relative_to(src):
                            src_str = str(src)
                            key = (src_str, ext)
                            if key not in by_src_ext:
                                by_src_ext[key] = []
                            by_src_ext[key].append(f)
                            break
        except (FileNotFoundError, PermissionError):
            # File might disappear between listing and stat, or be unreadable
            continue

    # Report
    for (src_str, ext), files in by_src_ext.items():
        if not files:
            continue
        chosen = pick(files, args.n, args.select)
        for i, path in enumerate(chosen, 1):
            try:
                mtime = path.stat().st_mtime
                print(f"{args.select.title()} {ext} in {src_str} #{i}: {path}  (mtime: {human_time(mtime)})")
                if args.cat and ext in _printable:
                    try:
                        with open(path, 'r', encoding='utf-8') as f:
                            print(f"---------------- {path} ----------------")
                            content = f.read()
                            if args.max_char is not None:
                                lines = content.splitlines()
                                processed_lines = []
                                for line in lines:
                                    if len(line) > args.max_char:
                                        line = line[:args.max_char] + "..."
                                    processed_lines.append(line)
                                content = "\n".join(processed_lines)
                            print(content)
                    except (FileNotFoundError, PermissionError, UnicodeDecodeError) as e:
                        print(f"<Error reading file: {e}>")
            except (FileNotFoundError, PermissionError):
                print(f"{args.select.title()} {ext} in {src_str} #{i}: {path}  (mtime: <unavailable>)")
                if args.cat:
                    print("<File unavailable>")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
