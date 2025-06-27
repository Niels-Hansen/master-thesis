#!/usr/bin/env python3
import os
import argparse

def count_images(folder, recursive=False, exts=None):
    """
    Count image files in `folder`.
    If recursive=True, also walks into subdirectories.
    `exts` is a set of lowercase extensions to include.
    """
    if exts is None:
        exts = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.gif'}
    total = 0

    if recursive:
        for root, dirs, files in os.walk(folder):
            for f in files:
                if os.path.splitext(f)[1].lower() in exts:
                    total += 1
    else:
        for f in os.listdir(folder):
            full = os.path.join(folder, f)
            if os.path.isfile(full) and os.path.splitext(f)[1].lower() in exts:
                total += 1

    return total

def main():
    parser = argparse.ArgumentParser(
        description="Count image files in a folder (optionally recursively)."
    )
    parser.add_argument(
        "folder",
        help="Path to the folder you want to scan."
    )
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Also count images in subdirectories."
    )
    args = parser.parse_args()

    if not os.path.isdir(args.folder):
        parser.error(f"`{args.folder}` is not a valid directory.")

    count = count_images(args.folder, recursive=args.recursive)
    scope = "recursively" if args.recursive else "in top-level"
    print(f"Found {count} image(s) {scope} within '{args.folder}'.")

if __name__ == "__main__":
    main()