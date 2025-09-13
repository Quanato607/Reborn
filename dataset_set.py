#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据 masks_test 列表为 BraTS2018 患者随机分配 mask_id
  · 训练集：根据 train.txt 固定样本，随机选择一定比例 (full_ratio) 为全模态 (mask_id==14)，其余随机分配其它 id (0-13)
  · 测试集：根据 val.txt 固定样本，全部全模态 (mask_id==14)

生成：
  Brats2018_train_seed<seed>.json
  Brats2018_test_seed<seed>.json
-------------------------------------------------------------------------

用法示例
    python build_brats2018_splits_maskid.py \
        --root /dataset/BraTS2018/ \
        --train_txt train.txt \
        --val_txt val.txt \
        --full_ratio 0.2 \
        --outdir meta_json/
"""

import argparse, glob, json, os, pathlib, random

MASKS_TEST = [
    [False, False, False, True],   # 0
    [False, True,  False, False],  # 1
    [False, False, True,  False],  # 2
    [True,  False, False, False],  # 3
    [False, True,  False, True],   # 4
    [False, True,  True,  False],  # 5
    [True,  False, True,  False],  # 6
    [False, False, True,  True],   # 7
    [True,  False, False, True],   # 8
    [True,  True,  False, False],  # 9
    [True,  True,  True,  False],  # 10
    [True,  False, True,  True],   # 11
    [True,  True,  False, True],   # 12
    [False, True,  True,  True],   # 13
    [True,  True,  True,  True],   # 14
]

FULL_ID = 14
SEEDS = [3407, 17891, 46213, 78503, 129847]

def read_list(txt_path):
    with open(txt_path, 'r') as f:
        pts = [line.strip() for line in f.readlines()]
    return pts

def build_train_split(train_list, full_ratio, seed):
    """根据给定 seed 随机选择 full_ratio 的样本为全模态"""
    random.seed(seed)
    n_total = len(train_list)
    n_full  = int(n_total * full_ratio)
    full_ids   = set(random.sample(train_list, n_full))
    train_meta = []
    for pid in train_list:
        if pid in full_ids:
            train_meta.append({"id": pid, "mask_id": FULL_ID})
        else:
            rid = random.randint(0, FULL_ID - 1)
            train_meta.append({"id": pid, "mask_id": rid})
    return train_meta

def build_test_split(val_list):
    """测试全部设为全模态"""
    return [{"id": pid, "mask_id": FULL_ID} for pid in val_list]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='/storage_data/cyf/shared_data/BraTS2018', help='BraTS2018 数据根目录 (仅用于读取测试列表之外的目录名匹配)')
    parser.add_argument('--train_txt', default='train.txt', help='训练集 id 列表')
    parser.add_argument('--val_txt', default='val.txt', help='验证集 id 列表 (作为 test)')
    parser.add_argument('--full_ratio', type=float, default=0.0, help='训练集中全模态样本比例，0.2 表示 20%%')
    parser.add_argument('--outdir', default='./jsons', help='输出目录')
    args = parser.parse_args()

    pathlib.Path(args.outdir).mkdir(parents=True, exist_ok=True)
    train_list = read_list(args.train_txt)
    val_list   = read_list(args.val_txt)

    for seed in SEEDS:
        train_meta = build_train_split(train_list, args.full_ratio, seed)
        test_meta  = build_test_split(val_list)

        train_path = os.path.join(args.outdir, f'Brats2018_train_seed{seed}.json')
        test_path  = os.path.join(args.outdir, f'Brats2018_test_seed{seed}.json')

        with open(train_path, 'w') as f:
            json.dump(train_meta, f, indent=2)
        with open(test_path, 'w') as f:
            json.dump(test_meta, f, indent=2)

        print(f"✅ seed={seed}  saved to {args.outdir}")

if __name__ == '__main__':
    main()
