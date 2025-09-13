import yaml
from data import make_data_loaders_divide
from models import build_HiKINet
from models.discriminator import get_style_discriminator
from solver import make_optimizer
from losses import get_losses, bce_loss, Dice, get_current_consistency_weight, get_losses_kd
import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from utils import *
import nibabel as nib
import numpy as np
from tqdm import tqdm
import logging
from torch.cuda.amp import autocast, GradScaler
import random

a=["flair",'t1','t1ce','t2']
masks_test = [[False, False, False, True], [False, True, False, False], [False, False, True, False], [True, False, False, False],
         [False, True, False, True], [False, True, True, False], [True, False, True, False], [False, False, True, True], [True, False, False, True], [True, True, False, False],
         [True, True, True, False], [True, False, True, True], [True, True, False, True], [False, True, True, True],
         [True, True, True, True]]

mask_name = ['t2', 't1', 't1c', 'flair',
            't1t2', 't1cet1', 'flairt1ce', 't1cet2', 'flairt2', 'flairt1',
            'flairt1cet1', 'flairt1cet2', 'flairt1t2', 't1cet1cet2',
            'flairt1cet1t2']

## Main section
task_name = 'brats_HiKINet'
dataset = task_name.split('_')[0]
config = yaml.load(open(os.path.join('./configs', (dataset + '.yml'))), Loader=yaml.FullLoader)
init_env('7')
loaders = make_data_loaders_divide(config)
model = build_HiKINet(inp_dim1 = 4, inp_dim2 = 4)
model = model.cuda()
d_style       = get_style_discriminator(num_classes = 128).cuda()
optimizer_d_style = optim.Adam(d_style.parameters(), lr=float(config['lr']), betas=(0.9, 0.99))
log_dir = os.path.join(config['path_to_log'], task_name)
optimizer, scheduler = make_optimizer(config, model)
losses = get_losses(config)
losses_kd = get_losses_kd(config)
weight_content = float(config['weight_content'])
weight_missing = float(config['weight_mispath'])
weight_full    = 1 - float(config['weight_mispath'])
epoch = 0
saved_model_path = log_dir+'/model_best.pth'
latest_model_path = log_dir+'/model_last.pth'
pretrain_model_path = log_dir+'/data/birth/cyf/output/zsh_output/CVPR25/log/brats_HiKINet_f/model_best.pth'
test_csv = log_dir+'/results.csv'
if os.path.exists(latest_model_path):
    model, d_style, optimizer, epoch, best_dice = load_old_model(model, d_style, optimizer, latest_model_path)
else:
    best_dice = 0.0

if not os.path.exists(log_dir):
    os.makedirs(log_dir)  

logger = get_logging(os.path.join(log_dir, 'train&valid.log'))

continue_training = False
pretrain_full_path = None

criteria = Dice() 
    
def train_val(model, d_style, loaders, optimizer, scheduler, losses, losses_kd, epoch_init=1, best_dice=0.0, pretrain_full_path=None):
    # 可选：启用 TF32（不影响显存，提升吞吐）
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    n_epochs = int(config['epochs'])
    logger.info(f'Start from epoch {epoch_init} to {n_epochs}')

    for epoch in range(epoch_init, n_epochs):
        weight_consistency = get_current_consistency_weight(epoch)
        scheduler.step()
        train_loss = 0.0
        kd_loss = 0.0

        dice_wt=0.0
        dice_et=0.0
        dice_tc=0.0
        hd95_wt=0.0
        hd95_et=0.0
        hd95_tc=0.0
        hd95=0.0
        dice=0.0

        for phase in ['train', 'eval']:
            if phase == 'eval' and epoch%10!=0 :
                continue

            loader = loaders[phase]
            total  = len(loader)

            if phase == 'train':
                model.train()
                d_style.train()
            else:
                model.eval()
                d_style.eval() 

            for batch_id, batch in tqdm(enumerate(loader), total=total, desc=f"{phase}"):
                if phase == 'train':
                    (x_m, y_m, cls_m, mask), (x_f, y_f, cls_f, _) = batch
                    x_m, y_m, cls_m = x_m.cuda(non_blocking=True), y_m.cuda(non_blocking=True), cls_m.cuda(non_blocking=True)
                    x_f, y_f, cls_f = x_f.cuda(non_blocking=True), y_f.cuda(non_blocking=True), cls_f.cuda(non_blocking=True)
                    mask_tensor = mask.cuda(non_blocking=True).to(x_m.dtype)
                else:
                    (x_f, y_f, cls_f, mask) = batch
                    x_f, y_f, cls_f = x_f.cuda(non_blocking=True), y_f.cuda(non_blocking=True), cls_f.cuda(non_blocking=True)
                    rb = random.randint(0, 14)
                    mask = masks_test[rb]
                    mask_tensor = (
                        torch.tensor(mask, dtype=torch.float32)
                        .view(1, 4, 1, 1, 1)
                        .cuda(non_blocking=True)
                    )
                    xn = x_f * mask_tensor
                

                with torch.set_grad_enabled(phase == 'train'):
                    if phase == 'train':
                        # ====== A. 前向 + 主损失 计算，包 autocast ======
                        (proj_f, seg_f,
                        proj_m, seg_m,
                        fus_c4d, c4d_m, kd_full) = model.forward(x_f, x_m, mask_tensor)
    
                        # ----- 计算联合生成端损失（一次 backward）-----
                        loss_dict = losses['co_loss'](config, seg_m, y_m)

                        d_style.train()
                        source_label = 0
                        target_label = 1

                        optimizer.zero_grad()
                        optimizer_d_style.zero_grad()

                        for param in d_style.parameters():
                            param.requires_grad = False

                        kd_full = kd_full * 0.1
                        loss_dict['loss_Co'] += kd_full

                        # ====== B. 用 scaler 做第一次反传（保持 retain_graph=True）======
                        loss_dict['loss_Co'].backward(retain_graph=True)

                        train_loss += loss_dict['loss_Co'].item()
                        kd_loss += float((kd_full.detach()).item())

                        # ====== C. 生成器对抗项前向（同样包 autocast）======
                        df_src_main = fus_c4d
                        df_trg_main = c4d_m
                        d_df_out_main = d_style(df_trg_main)
                        loss_adv_df_trg_main = bce_loss(d_df_out_main, source_label)
                        loss = 0.0002 * loss_adv_df_trg_main

                        loss.backward()


                        ####################### Train discriminator networks ######################################
                        # enable training mode on discriminator networks
                        for param in d_style.parameters():
                            param.requires_grad = True
                        df_src_main = df_src_main.detach()
                        d_df_out_main = d_style(df_src_main)
                        loss_d_feature_main = bce_loss(d_df_out_main, source_label)

                        loss_d_feature_main.backward()
                    
                        ####################### train with target ##################################################
                        df_trg_main = df_trg_main.detach()
                        d_df_out_main = d_style(df_trg_main)
                        loss_d_feature_main = bce_loss(d_df_out_main, target_label)

                        loss_d_feature_main.backward()
                    else:
                        _, seg_eval, _ = model.forward_missing(xn)
                        y_eval = y_f.detach()

                if phase == 'train':
                    optimizer.step()         # 原来的 optimizer.step()
                    optimizer_d_style.step()   # 原来的 optimizer_d_style.step()
                    if (batch_id + 1) % 20 == 0:
                        print(f'Epoch {epoch+1}>> itteration {batch_id+1}>> training loss>> {train_loss/(batch_id+1)} >> kd loss>> {kd_loss}')
 
                else:
                    wt,et,tc=measure_dice_score(seg_eval, y_eval, divide=True)
                    dice+=(wt+et+tc)/3.0
                    dice_wt+=wt
                    dice_et+=et
                    dice_tc+=tc

                    hdwt,hdet,hdtc= measure_hd95(seg_eval, y_eval, divide=True)
                    hd95+=(hdwt+hdet+hdtc)/3.0
                    hd95_wt+=hdwt
                    hd95_et+=hdet
                    hd95_tc+=hdtc

            if phase == 'train':
                logger.info(f'Epoch {epoch+1} overall training loss>> {train_loss/(batch_id+1)}')
                state = {}
                state['model'] = model.state_dict()
                state['d_style'] = d_style.state_dict()
                state['optimizer'] = optimizer.state_dict()
                state['epochs'] = epoch
                state['dice'] = dice
                state['best_dice'] = best_dice
                file_name = log_dir + '/model_last.pth'
                torch.save(state, file_name)
            else:
                dice = (dice/(batch_id+1))
                hd95 = (hd95/(batch_id+1))
                logger.info(f'Epoch {epoch+1} validation dice score for missing modality whole>> {dice_wt/(batch_id+1)} ,core{dice_tc/(batch_id+1)},enhance{dice_et/(batch_id+1)}')
                logger.info(f'Epoch {epoch+1} validation hd95 score for missing modality whole>> {hd95_wt/(batch_id+1)} ,core{hd95_tc/(batch_id+1)},enhance{hd95_et/(batch_id+1)}')

                if dice > best_dice:
                    logger.info('Get the Best Dice Score !!!')
                    state = {}
                    state['model'] = model.state_dict()
                    state['d_style'] = d_style.state_dict()
                    state['optimizer'] = optimizer.state_dict()
                    state['epochs'] = epoch
                    state['dice'] = dice
                    state['best_dice'] = best_dice
                    file_name = log_dir+'/model_best.pth'
                    torch.save(state, file_name)
                    best_dice = dice

import os, csv, numpy as np      # 只有缺失时再补
# ...

def test_val(model_missing,
             loaders,
             rb: int,
             *,
             calc_dice: bool = True,
             calc_hd95: bool = True,
             calc_sen:  bool = True,
             calc_iou:  bool = True,
             csv_path: str = test_csv):
    """
    保持原先推理流程，但输出格式 / CSV 写入与旧版 test_val 对齐
    """
    # ── 初始化累加器 & 列表 ──────────────────────────
    def init_hist(flag): 
        return (0.0, []) if flag else (None, None)

    dice_sum, dice_hist   = init_hist(calc_dice)      # overall mean(三类平均)
    hd95_sum, hd95_hist   = init_hist(calc_hd95)

    dice_wt_sum, dice_wt_hist = init_hist(calc_dice)
    dice_tc_sum, dice_tc_hist = init_hist(calc_dice)
    dice_et_sum, dice_et_hist = init_hist(calc_dice)

    hd95_wt_sum, hd95_wt_hist = init_hist(calc_hd95)
    hd95_tc_sum, hd95_tc_hist = init_hist(calc_hd95)
    hd95_et_sum, hd95_et_hist = init_hist(calc_hd95)

    # 可选指标
    sen_wt_hist = sen_tc_hist = sen_et_hist = None
    iou_wt_hist = iou_tc_hist = iou_et_hist = None
    if calc_sen:
        sen_wt_hist, sen_tc_hist, sen_et_hist = [], [], []
    if calc_iou:
        iou_wt_hist, iou_tc_hist, iou_et_hist = [], [], []

    # ── 推理循环（保持 train() / BN 实时统计的特性） ──
    for phase in ['eval']:
        loader = loaders[phase]
        for batch_id, (batch_x, batch_y, cls, _) in tqdm(enumerate(loader), total=len(loader), desc="test Batches"):
            batch_x, batch_y = batch_x.cuda(non_blocking=True), batch_y.cuda(non_blocking=True)
            with torch.set_grad_enabled(phase == 'train'):

                # 固定缺失模态
                mask = masks_test[rb]
                mask_tensor = torch.tensor(mask, dtype=torch.float32).view(1, 4, 1, 1, 1)
                mask_tensor = mask_tensor.cuda(non_blocking=True)
                batch_xn = batch_x * mask_tensor
                proj_m, seg_m, c4d_m = model_missing.forward_missing(batch_xn[:, 0:])

            # —— 指标计算 ——
            if calc_dice:
                wt, et, tc = measure_dice_score(seg_m, batch_y, divide=True)
                dice_wt_sum += wt; dice_tc_sum += tc; dice_et_sum += et
                dice_sum    += (wt + et + tc) / 3.0
                dice_wt_hist.append(wt); dice_tc_hist.append(tc); dice_et_hist.append(et)
                dice_hist.append((wt + et + tc) / 3.0)

            if calc_hd95:
                hdwt, hdet, hdtc = measure_hd95(seg_m, batch_y, divide=True)
                hd95_wt_sum += hdwt; hd95_tc_sum += hdtc; hd95_et_sum += hdet
                hd95_sum    += (hdwt + hdet + hdtc) / 3.0
                hd95_wt_hist.append(hdwt); hd95_tc_hist.append(hdtc); hd95_et_hist.append(hdet)
                hd95_hist.append((hdwt + hdet + hdtc) / 3.0)

            if calc_sen:
                swt, set_, sct = measure_sensitivity_score(seg_m, batch_y, divide=True)
                sen_wt_hist.append(swt); sen_tc_hist.append(sct); sen_et_hist.append(set_)

            if calc_iou:
                iwt, iet, ict = measure_iou_score(seg_m, batch_y, divide=True)
                iou_wt_hist.append(iwt); iou_tc_hist.append(ict); iou_et_hist.append(iet)

    # —— 统计 μ±σ ——  
    def fmt(hist):
        return f"{np.mean(hist):.4f}±{np.std(hist):.4f}"

    NA = ''
    row = [
        mask_name[rb],
        fmt(dice_wt_hist) if calc_dice else NA,
        fmt(dice_tc_hist) if calc_dice else NA,
        fmt(dice_et_hist) if calc_dice else NA,
        fmt(hd95_wt_hist) if calc_hd95 else NA,
        fmt(hd95_tc_hist) if calc_hd95 else NA,
        fmt(hd95_et_hist) if calc_hd95 else NA,
        fmt(sen_wt_hist)  if calc_sen  else NA,
        fmt(sen_tc_hist)  if calc_sen  else NA,
        fmt(sen_et_hist)  if calc_sen  else NA,
        fmt(iou_wt_hist)  if calc_iou  else NA,
        fmt(iou_tc_hist)  if calc_iou  else NA,
        fmt(iou_et_hist)  if calc_iou  else NA,
    ]

    # —— 统一日志输出 ——  
    logger.critical(
        f"{row[0]}  "
        f"Dice(WT/TC/ET) {row[1]}, {row[2]}, {row[3]} | "
        f"HD95 {row[4]}, {row[5]}, {row[6]} | "
        f"Sen {row[7]}, {row[8]}, {row[9]} | "
        f"IoU {row[10]}, {row[11]}, {row[12]}"
    )

    # —— CSV 追加 ——  
    if csv_path:
        header = [
            'mask',
            'Dice WT', 'Dice TC', 'Dice ET',
            'HD95 WT', 'HD95 TC', 'HD95 ET',
            'Sensitivity WT', 'Sensitivity TC', 'Sensitivity ET',
            'IoU WT', 'IoU TC', 'IoU ET'
        ]
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        file_exists = os.path.isfile(csv_path)
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(header)
            writer.writerow(row)
        logger.info(f"Row appended to {csv_path}")

    print(f"[{mask_name[rb]}] 指标计算并写入 CSV 完成。")

train_val(model, d_style, loaders, optimizer, scheduler, losses, losses_kd, epoch, best_dice, pretrain_full_path)
for i in range(0,15):
    test_val(model, loaders, i)
print('Training process is finished')