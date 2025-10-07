#!/usr/bin/env python3
"""
Move a whole folder safely.

Usage:
  python move_folder.py /path/to/src /path/to/dst [--overwrite] [--dry-run] [-v]

Behavior:
- If DST does not exist: src is renamed to exactly DST.
- If DST exists and is a directory: src is moved inside DST as DST/src.name.
- If the final target exists:
    - error (default)
    - or remove it first if --overwrite is set.
"""

from __future__ import annotations
import argparse
import shutil
import sys
from pathlib import Path


def _eprint(*a, **k):
    print(*a, file=sys.stderr, **k)


def resolve_paths(src: str, dst: str) -> tuple[Path, Path]:
    s = Path(src).expanduser().resolve()
    d = Path(dst).expanduser().resolve()
    return s, d


def compute_final_target(src: Path, dst: Path) -> Path:
    """
    If dst doesn't exist -> final target is dst (rename).
    If dst exists and is a directory -> final target is dst / src.name.
    If dst exists and is a file -> invalid unless overwriting to that exact path.
    """
    if not dst.exists():
        return dst
    if dst.is_dir():
        return dst / src.name
    # dst exists and is a file; only valid if we intend to replace it with the folder name
    return dst


def ensure_safe(src: Path, final_target: Path):
    if not src.exists():
        raise FileNotFoundError(f"Source does not exist: {src}")
    if not src.is_dir():
        raise NotADirectoryError(f"Source is not a directory: {src}")
    # Prevent moving into itself or a subdirectory of itself
    try:
        # .relative_to raises ValueError if not a subpath
        if final_target.resolve().relative_to(src.resolve()):
            raise ValueError(
                f"Refusing to move a folder into itself or its subdirectory:\n  src={src}\n  dst={final_target}"
            )
    except ValueError:
        # not a subpath => fine
        pass
    # Prevent no-op (src == target)
    if src.resolve() == final_target.resolve():
        raise ValueError("Source and final destination are the same path; nothing to do.")


def maybe_overwrite(path: Path, overwrite: bool, verbose: bool, dry_run: bool):
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Destination already exists: {path} (use --overwrite to replace)")
        if verbose or dry_run:
            print(f"[overwrite] removing existing: {path}")
        if not dry_run:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink(missing_ok=True)


def move_folder(src: Path, dst: Path, *, overwrite: bool, verbose: bool, dry_run: bool):
    final_target = compute_final_target(src, dst)
    ensure_safe(src, final_target)
    # If final target exists, handle overwrite policy
    if final_target.exists():
        maybe_overwrite(final_target, overwrite, verbose, dry_run)
    # Ensure parent exists
    parent = final_target.parent
    if not parent.exists():
        if verbose or dry_run:
            print(f"[mkdir] {parent}")
        if not dry_run:
            parent.mkdir(parents=True, exist_ok=True)
    # Do the move (shutil.move handles cross-filesystem moves)
    if verbose or dry_run:
        print(f"[move] {src} -> {final_target}")
    if not dry_run:
        shutil.move(str(src), str(final_target))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Move a whole folder safely.")
    parser.add_argument("src", help="Source folder")
    parser.add_argument("dst", help="Destination path or directory")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing destination")
    parser.add_argument("--dry-run", action="store_true", help="Show what would happen, do not modify")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args(argv)

    try:
        src, dst = resolve_paths(args.src, args.dst)
        move_folder(src, dst, overwrite=args.overwrite, verbose=args.verbose, dry_run=args.dry_run)
    except Exception as e:
        _eprint(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
