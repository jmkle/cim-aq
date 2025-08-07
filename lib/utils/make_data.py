#!/usr/bin/env python3
##############################################################################
# Copyright (C) 2025 Joel Klein                                              #
# All Rights Reserved                                                        #
#                                                                            #
# This work is licensed under the terms described in the LICENSE file        #
# found in the root directory of this source tree.                           #
# This work is based on the HAQ framework which can be found                 #
# here https://github.com/mit-han-lab/haq/                                   #
##############################################################################

from multiprocessing import Pool
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def create_symlink(src_dst_split: tuple[str, str, str]) -> None:
    """Create symbolic link from source to destination directory."""
    src, dst, split = src_dst_split
    src_path = Path(src)
    dst_path = Path(dst)
    relative_src = Path('..') / '..' / 'imagenet' / split / src_path.name
    dst_symlink = dst_path / src_path.name
    dst_symlink.symlink_to(relative_src)


def read_class_list(txt_path: Path) -> list[str]:
    """Read class names from text file."""
    with open(txt_path, 'r') as f:
        return [line[:9] for line in f]


def process_split(src_dir: Path,
                  dst_dir: Path,
                  split: str,
                  class_list: list[str],
                  n_threads: int = 32) -> None:
    """Process a single split (train/val) by creating symlinks."""
    src_split_dir = src_dir / split
    dst_split_dir = dst_dir / split
    dst_split_dir.mkdir(parents=True, exist_ok=True)

    pair_list = [(str(src_split_dir / cls), str(dst_split_dir), split)
                 for cls in class_list]

    with Pool(n_threads) as pool:
        if tqdm is not None:
            for _ in tqdm(pool.imap_unordered(create_symlink, pair_list),
                          total=len(pair_list),
                          desc=f"Processing {split}"):
                pass
        else:
            pool.map(create_symlink, pair_list)


def main() -> None:
    """Main function to create ImageNet100 dataset symlinks."""
    root = Path(__file__).parent.parent.parent
    data_name = 'imagenet100'
    src_dir = root / 'data' / 'imagenet'
    dst_dir = root / 'data' / data_name
    txt_path = root / 'lib' / 'utils' / f'{data_name}.txt'

    class_list = read_class_list(txt_path)

    for split in ['train', 'val']:
        process_split(src_dir, dst_dir, split, class_list)


if __name__ == '__main__':
    main()
