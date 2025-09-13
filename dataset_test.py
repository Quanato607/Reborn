#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_brats2018_divide.py
------------------------
快速遍历 Brats2018 数据集，兼容：
    • train=True  →  DataLoader 返回 ((partial), (full))
    • train=False →  DataLoader 返回 (sample)

统计信息：
    · modalities≈x/4  —— 由 mask 布尔向量直接计算
    · seg_pos≈y%      —— 4‑通道分割掩码中正样本体素占全集比例

用法示例：
    python test_brats2018_divide.py \
        --data_root /dataset/BraTS2018/ \
        --meta_dir  /dataset/BraTS2018/meta_json/ \
        --batch 4 --workers 2 \
        --crop 160 192 128
"""
import argparse, os, torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# -----------------------------------------------------------
from data import Brats2018_divide as DatasetClass
# -----------------------------------------------------------


def make_loader(json_path, data_root, crop, train_flag, batch, workers):
    ds = DatasetClass(
        json_path=json_path,
        data_root=data_root,
        train=train_flag,
        normalization=True,
        dataset='brats'
    )
    return DataLoader(ds, batch_size=batch, shuffle=train_flag,
                      num_workers=workers, pin_memory=True,
                      drop_last=False)


def stat_from_batch(imgs, masks, segs):
    """
    imgs : [B,4,H,W,D]
    masks: [B,4]  (bool / 0‑1 float)
    segs : [B,4,H,W,D]
    """
    modal_mean = masks.float().mean().item() * 4                  # 平均保留模态数
    seg_pos    = segs.sum().item() / segs.numel() * 100           # 正体素百分比
    return f"modalities≈{modal_mean:.2f}/4  seg_pos≈{seg_pos:.3f}%"


def gather_pair(batch):
    """
    把 ((x_p, seg_p, cls_p, m_p), (x_f, seg_f, cls_f, m_f))
    拼成全批次大张量返回 (imgs, masks, segs)
    """
    (x_p, seg_p, _, m_p), (x_f, seg_f, _, m_f) = batch
    imgs  = torch.cat([x_p, x_f],  dim=0)
    segs  = torch.cat([seg_p, seg_f], dim=0)
    masks = torch.cat([m_p, m_f], dim=0)
    return imgs, masks, segs


# ---------------- helper：把 pair 批次拼成整体 ----------------
def merge_pair_batch(pair_batch):
    """
    输入:
        pair_batch = ((x_p, seg_p, cls_p, m_p),
                      (x_f, seg_f, cls_f, m_f))
        其中 x_p, x_f shape ==> [B, 4, H, W, D]
    返回:
        imgs  : [2B, 4, H, W, D]
        segs  : [2B, 4, H, W, D]
        masks : [2B, 4]
    """
    (x_p, seg_p, _, m_p), (x_f, seg_f, _, m_f) = pair_batch
    imgs  = torch.cat([x_p,  x_f],  dim=0)
    segs  = torch.cat([seg_p, seg_f], dim=0)
    masks = torch.cat([m_p,  m_f],  dim=0)
    return imgs, segs, masks


def iterate(loader, name="set"):
    pbar = tqdm(loader, desc=f"[{name}]", unit="batch")
    for batch in pbar:
        # 训练模式：batch 是长度 2 的 pair
        if len(batch) == 2 and isinstance(batch[0], (tuple, list)):
            imgs, segs, masks = merge_pair_batch(batch)
        else:  # 推理模式：batch 直接是 (img, seg, cls, mask)
            imgs, segs, _, masks = batch

        # -------- 统计信息 --------
        modal_mean = masks.float().mean().item() * 4            # 平均保留模态
        seg_pos    = segs.sum().item() / segs.numel() * 100     # 正体素百分比
        pbar.set_postfix_str(f"modalities≈{modal_mean:.2f}/4  seg_pos≈{seg_pos:.3f}%")

    print(f"✔  {name}: {len(loader.dataset)} 个样本遍历完成\n")




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root',      default='/data/birth/cyf/shared_data/BRATS2018/BraTS2018_MT')
    ap.add_argument('--meta_dir',  default='/data/birth/cyf/output/zsh_output/CVPR25/jsons')
    ap.add_argument('--batch',   type=int, default=4)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--crop', nargs=3, type=int, metavar=('D','H','W'),
                    default=None,
                    help='三维裁剪大小；若不想裁剪请省略该参数')
    args = ap.parse_args()

    crop_tuple = tuple(args.crop) if args.crop is not None else None

    train_json = os.path.join(args.meta_dir, 'Brats2018_train_seed3407.json')
    test_json  = os.path.join(args.meta_dir, 'Brats2018_test_seed3407.json')

    train_loader = make_loader(train_json, args.data_root, crop_tuple,
                               train_flag=True,
                               batch=args.batch, workers=args.workers)
    test_loader  = make_loader(test_json,  args.data_root, crop_tuple,
                               train_flag=False,
                               batch=args.batch, workers=args.workers)

    torch.set_grad_enabled(False)
    iterate(train_loader, "train")
    iterate(test_loader,  "test")


if __name__ == '__main__':
    main()
 