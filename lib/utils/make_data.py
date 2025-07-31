import os
import subprocess
from multiprocessing import Pool
from pathlib import Path

from tqdm import tqdm

root = Path.cwd()
data_name = 'imagenet100'
src_dir = root / 'data' / 'imagenet'
dst_dir = root / 'data' / data_name
txt_path = root / 'lib' / 'utils' / f'{data_name}.txt'

# os.makedirs('/dev/shm/dataset', exist_ok=True)
# os.makedirs('/dev/shm/dataset/imagenet', exist_ok=True)

n_thread = 32


def copy_func(pair):
    src, dst = pair
    # os.system('rsync -r {} {}'.format(src, dst))
    os.system('ln -s {} {}'.format(src, dst))


for split in ['train', 'val']:
    src_split_dir = src_dir / split
    dst_split_dir = dst_dir / split
    dst_split_dir.mkdir(parents=True, exist_ok=True)
    cls_list = []
    f = open(str(txt_path), 'r')
    for x in f:
        cls_list.append(x[:9])
    # pair_list = [(str(src_split_dir / c), str(dst_split_dir / c)) for c in cls_list]
    pair_list = [(str(src_split_dir / c), str(dst_split_dir))
                 for c in cls_list]

    p = Pool(n_thread)

    for _ in tqdm(p.imap_unordered(copy_func, pair_list),
                  total=len(pair_list)):
        pass
    # p.map(worker, vid_list)
    p.close()
    p.join()
