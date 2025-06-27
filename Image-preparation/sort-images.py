#!/usr/bin/env python3
import os
import re
import shutil
from natsort import natsorted

# CONFIG
input_base  = r"G:\My Drive\MasterThesis\CircularMaskedData"
output_base = r"G:\My Drive\MasterThesis\CircularMaskedDataSorted"

# Intervals in minutes you want to sample “every N minutes”
intervals = [60, 120, 300, 720]

# Regex to extract the number before “min” in each filename
ts_re = re.compile(r'_(\d+)min\.')

def find_closest(keys, target):
    """Return the key in `keys` closest to `target`."""
    return min(keys, key=lambda k: abs(k - target))

def process_ibt(ibt_folder, ibt_name):
    # gather and sort all timestamped images
    files = natsorted(f for f in os.listdir(ibt_folder) if ts_re.search(f))
    if not files:
        print(f"No timestamped images in {ibt_folder}, skipping.")
        return

    # map timestamp → list of filenames
    ts_map = {}
    for fn in files:
        t = int(ts_re.search(fn).group(1))
        ts_map.setdefault(t, []).append(fn)

    max_ts = max(ts_map.keys())

    for interval in intervals:
        dst_root = os.path.join(output_base, f"{interval}min", ibt_name)
        os.makedirs(dst_root, exist_ok=True)

        t = 0
        while t <= max_ts:
            best = find_closest(ts_map.keys(), t)
            for fn in ts_map[best]:
                src = os.path.join(ibt_folder, fn)
                dst = os.path.join(dst_root, fn)
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
            t += interval

        print(f"Copied images every {interval}min for {ibt_name}")

def main():
    for ibt_name in os.listdir(input_base):
        ibt_folder = os.path.join(input_base, ibt_name)
        if not os.path.isdir(ibt_folder):
            continue
        process_ibt(ibt_folder, ibt_name)

if __name__ == "__main__":
    main()
