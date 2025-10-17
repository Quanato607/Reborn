import yaml
from data import make_data_loaders_divide
from models import build_model, build_AdaMMKD
from models.discriminator import get_style_discriminator
from solver import make_optimizer_double
from losses import get_losses, bce_loss, Dice, get_current_consistency_weight, get_losses_kd
import os
import torch
import torch.optim as optim
from utils import *
from tqdm import tqdm
import os, csv, numpy as np 
from strategy.EMA import EMA
from strategy.GAPO import GAPO

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
task_name = 'retreat_1_reborn_dot_1'
dataset = task_name.split('_')[0]
seed = task_name.split('_')[1]
p = task_name.split('_')[-1]
full_proportion = float(task_name.split('_')[-1])/10
config = yaml.load(open(os.path.join('./configs', (dataset + '_' + seed + 'd' + p + '.yml'))), Loader=yaml.FullLoader)
init_env('1')
loaders = make_data_loaders_divide(config)
model_full, model_missing = build_AdaMMKD(inp_dim1 = 4, inp_dim2 = 4)
model_full    = model_full.cuda()
model_missing = model_missing.cuda()
d_style       = get_style_discriminator(num_classes = 128).cuda()
optimizer_d_style = optim.Adam(d_style.parameters(), lr=float(config['lr']), betas=(0.9, 0.99))
log_dir = os.path.join(config['path_to_log'], task_name)
gapo = GAPO(num_tasks=2, full_proportion=full_proportion).cuda()  # 2个任务：鲁棒性损失 和 知识蒸馏损失
optimizer, scheduler = make_optimizer_double(config, model_full, model_missing, gapo=gapo)
losses = get_losses(config)
losses_kd = get_losses_kd(config)
weight_content = float(config['weight_content'])
weight_missing = float(config['weight_mispath'])
weight_full    = 1 - float(config['weight_mispath'])
epoch = 0
saved_model_path = log_dir+'/model_best.pth'
latest_model_path = log_dir+'/model_last.pth'
test_csv = log_dir+'/results.csv'
if os.path.exists(latest_model_path):
    model_full, model_missing, d_style, optimizer, epoch, best_dice, \
        ema_model_full, ema_model_missing = load_old_model_double(
            model_full, model_missing, d_style, optimizer, latest_model_path, use_ema=True
        )
else:
    best_dice = 0.0
    ema_model_full = EMA(model_full)
    ema_model_missing = EMA(model_missing)

if not os.path.exists(log_dir):
    os.makedirs(log_dir) 

logger = get_logging(os.path.join(log_dir, 'train&valid.log'))

continue_training = False
pretrain_full_path = None

criteria = Dice() 
    
def train_val(
    model_full, model_missing, d_style, loaders, optimizer, scheduler, losses,
    ema_model_full, ema_model_missing, epoch_init=1, best_dice=0.0
):

    n_epochs = int(config['epochs'])
    logger.info(f'Start from epoch {epoch_init} to {n_epochs}')

    for epoch in range(epoch_init, n_epochs):
        weight_consistency = get_current_consistency_weight(epoch)
        scheduler.step()
        train_loss = 0.0
        dice_wt=0.0
        dice_et=0.0
        dice_tc=0.0
        hd95_wt=0.0
        hd95_et=0.0
        hd95_tc=0.0
        hd95=0.0
        dice=0.0

        for phase in ['train', 'eval']:
            if phase == 'eval' and epoch % 5 != 0:
                continue

            loader = loaders[phase]
            total = len(loader)

            for batch_id, batch in tqdm(enumerate(loader), total=total, desc=f"{phase}"):

                if phase == 'train':
                    (x_m, y_m, cls_m, mask, mask_), (x_f, y_f, cls_f, _, _) = batch
                    x_m, y_m, cls_m = x_m.cuda(non_blocking=True), y_m.cuda(non_blocking=True), cls_m.cuda(non_blocking=True)
                    x_f, y_f, cls_f = x_f.cuda(non_blocking=True), y_f.cuda(non_blocking=True), cls_f.cuda(non_blocking=True)
                    mask_tensor = mask.cuda(non_blocking=True).to(x_m.dtype)
                    model_full.train()
                    model_missing.train()
                    d_style.train()
                elif phase == 'eval':
                    model_full.eval()
                    model_missing.eval()
                    with torch.no_grad():
                        (x, y, cls, mask, mask_) = batch
                        x, y, cls = x.cuda(non_blocking=True), y.cuda(non_blocking=True), cls.cuda(non_blocking=True)
                        rb = random.randint(0, 14)
                        mask = masks_test[rb]
                        mask_tensor = (
                            torch.tensor(mask, dtype=torch.float32)
                            .view(1, 4, 1, 1, 1)
                            .cuda(non_blocking=True)
                        )
                        batch_xn = x * mask_tensor
                        seg_m, style_m, content_m, Gs_m, logit_m, cls_probs_m= model_missing(batch_xn[:,0:], mask)

                        wt,et,tc=measure_dice_score(seg_m, y, divide=True)
                    dice+=(wt+et+tc)/3.0
                    dice_wt+=wt
                    dice_et+=et
                    dice_tc+=tc

                    hdwt,hdet,hdtc= measure_hd95(seg_m, y, divide=True)
                    hd95+=(hdwt+hdet+hdtc)/3.0
                    hd95_wt+=hdwt
                    hd95_et+=hdet
                    hd95_tc+=hdtc

                    continue

                with torch.set_grad_enabled(phase == 'train'):
                    
                    # === 原始训练流程: 1. 缺失模态鲁棒性 loss ===
                    seg_m, style_m, content_m, Gs_m, logit_m, cls_probs_m = model_missing(x_m[:, 0:], mask_)
                    loss_dict1 = losses['adamm_co_loss_missing'](config, y_m, seg_m, cls_probs_m, epoch, cls_m)
                    
                    # === 原始训练流程: 2. KD / 全模态引导 loss ===
                    batch_xn = x_f * mask_tensor
                    seg_f, style_f, content_f, Gs_f, logit_f, cls_probs_f = model_full(x_f[:,0:])
                    seg_m, style_m, content_m, Gs_m, logit_m, cls_probs_m = model_missing(batch_xn[:,0:], mask_)
                    loss_dict2 = losses['adamm_co_loss'](config, seg_f, content_f, cls_probs_f, y_f, seg_m, content_m, cls_probs_m, style_f, style_m, epoch, cls_f) 
                    
                    d_style.train()
                    # labels for style adversarial training
                    source_label = 0
                    target_label = 1

                    optimizer.zero_grad()
                    optimizer_d_style.zero_grad()
                    
                    # only train. Don't accumulate grads in disciminators
                    for param in d_style.parameters():
                        param.requires_grad = False

                    # 3a) 更新 model_full —— 冻结 missing，仅对 full 反传 KD
                    for p in model_missing.parameters():
                        p.requires_grad = False
                    # 仅让与 model_full 相关的图参与反传
                    loss_full = loss_dict2['loss_Co']
                    loss_full.backward(retain_graph=True)
                    for p in model_missing.parameters():
                        p.requires_grad = True

                    params_m = [p for p in model_missing.parameters() if p.requires_grad]
                    grads1 = torch.autograd.grad(
                        loss_dict1['loss_Co'], params_m, retain_graph=True, allow_unused=True
                    )
                    grads2 = torch.autograd.grad(
                        loss_dict2['loss_Co'], params_m, retain_graph=True, allow_unused=True
                    )

                    # 把 None 用同形状的 0 張量替换
                    grads1 = [g if g is not None else torch.zeros_like(p) for g, p in zip(grads1, params_m)]
                    grads2 = [g if g is not None else torch.zeros_like(p) for g, p in zip(grads2, params_m)]
                    
                    # 1) 先用“当前” α/λ 做一次融合
                    blended_grads = gapo.forward(grads1, grads2)

                    # 3) 把融合后的梯度写回 model_missing（一定要 detach）
                    for p, g in zip(params_m, blended_grads):
                        if p.grad is None:
                            p.grad = torch.zeros_like(p)
                        p.grad.add_(g.detach())
                    
                    # === 原来的对抗训练流程 ===
                    ##################### adversarial training ot fool the discriminator ######################
                    df_src_main = style_f
                    df_trg_main = style_m
                    d_df_out_main = d_style(df_trg_main)
                    loss_adv_df_trg_main = bce_loss(d_df_out_main, source_label)
                    loss = 0.0002 * loss_adv_df_trg_main
                    if phase == 'train':
                        loss.backward()                    
                    
                    ####################### Train discriminator networks ######################################
                    # enable training mode on discriminator networks
                    for param in d_style.parameters():
                        param.requires_grad = True
                    df_src_main = df_src_main.detach()
                    d_df_out_main = d_style(df_src_main)
                    loss_d_feature_main = bce_loss(d_df_out_main, source_label)
                    if phase == 'train':
                        loss_d_feature_main.backward()
                    
                    ####################### train with target ##################################################
                    df_trg_main = df_trg_main.detach()
                    d_df_out_main = d_style(df_trg_main)
                    loss_d_feature_main = bce_loss(d_df_out_main, target_label)

                    if phase == 'train':  
                        loss_d_feature_main.backward()

                # 打印或保存模型
                if phase == 'train':
                    # === 原始优化器 step ===
                    optimizer.step()
                    optimizer_d_style.step()
                    
                    # === EMA 更新 ===
                    ema_model_full.update()
                    ema_model_missing.update()
                    train_loss += loss_dict1['loss_Co'].item() + loss_dict2['loss_Co'].item()
                    if (batch_id + 1) % 20 == 0:
                        print(f'Epoch {epoch+1}>> itteration {batch_id+1}>> training loss>> {train_loss/(batch_id+1)}')


            if phase == 'train':
                logger.info(f'Epoch {epoch+1} overall training loss>> {train_loss/(batch_id+1)}')
                state = {}
                state['model_full'] = model_full.state_dict()
                state['model_missing'] = model_missing.state_dict()
                state['d_style'] = d_style.state_dict()
                state['optimizer'] = optimizer.state_dict()
                state['epochs'] = epoch
                state['dice'] = dice
                state['best_dice'] = dice
                # >>> 新增：保存 EMA <<<
                state['ema_model_full'] = ema_state_dict(ema_model_full)
                state['ema_model_missing'] = ema_state_dict(ema_model_missing)

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
                    state['model_full'] = model_full.state_dict()
                    state['model_missing'] = model_missing.state_dict()
                    state['d_style'] = d_style.state_dict()
                    state['optimizer'] = optimizer.state_dict()
                    state['epochs'] = epoch
                    state['dice'] = dice
                    state['best_dice'] = dice
                    # >>> 新增：保存 EMA <<<
                    state['ema_model_full'] = ema_state_dict(ema_model_full)
                    state['ema_model_missing'] = ema_state_dict(ema_model_missing)

                    file_name = log_dir + '/model_best.pth'
                    torch.save(state, file_name)
                    best_dice = dice

def pretrain(
    model_full, model_missing, d_style, loaders, optimizer, losses_kd,
    num_epochs=10, device='cuda', log_dir='./', epoch_to_load=None
):
    """
    同时预训练全模态模型和缺失模态模型：
    - 使用全模态数据模拟15种缺失模态组合
    - 对全模态模型和缺失模态模型进行KD + 对抗训练
    - 预训练阶段不使用 EMA 更新
    - 支持检查是否已存在预训练权重，若存在则跳过训练，且仅加载最后一轮的预训练权重
    """
    model_full.train()   # 全模态模型训练
    model_missing.train()  # 缺失模态模型训练
    d_style.train()  # 判别器训练

    # 检查是否已有预训练权重（只检查最后一轮的权重）
    if epoch_to_load is not None:
        pretrain_model_path = os.path.join(log_dir, f"pretraind_epoch_{epoch_to_load}.pth")
        if os.path.exists(pretrain_model_path):
            print(f"Found pre-trained weights for epoch {epoch_to_load}, loading...")
            checkpoint = torch.load(pretrain_model_path)
            model_full.load_state_dict(checkpoint['model_full'])
            model_missing.load_state_dict(checkpoint['model_missing'])
            d_style.load_state_dict(checkpoint['d_style'])
            # optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"Pre-trained model loaded successfully from {pretrain_model_path}")
            return  # 直接返回，跳过训练

    # 定义缺失模态的组合
    all_masks = [
        [False, False, False, True], [False, True, False, False], [False, False, True, False], [True, False, False, False],
        [False, True, False, True], [False, True, True, False], [True, False, True, False], [False, False, True, True],
        [True, False, False, True], [True, True, False, False], [True, True, True, False], [True, False, True, True],
        [True, True, False, True], [False, True, True, True], [True, True, True, True]
    ]

    # 进行预训练
    for epoch in range(num_epochs):
        train_loss = 0.0

        for batch_id, batch in tqdm(enumerate(loaders['pretrain']), total=len(loaders['pretrain']), desc=f"Pretrain Epoch {epoch+1}"):

            # 获取全模态数据
            (x_f, y_f, cls, _, _) = batch[1]
            x_f, y_f, cls = x_f.to(device), y_f.to(device), cls.to(device)

            optimizer.zero_grad()
            optimizer_d_style.zero_grad()

            # 遍历15种缺失模态组合
            for mask in all_masks:
                mask_tensor = torch.tensor(mask, dtype=torch.float32).view(1, 4, 1, 1, 1).to(device)
                batch_xn = x_f * mask_tensor

                seg_f, style_f, content_f, Gs_f, logit_f, cls_probs_f = model_full(x_f[:,0:])
                seg_m, style_m, content_m, Gs_m, logit_m, cls_probs_m = model_missing(batch_xn[:,0:], mask)
                loss_dict2 = losses['adamm_co_loss'](config, seg_f, content_f, cls_probs_f, y_f, seg_m, content_m, cls_probs_m, style_f, style_m, epoch, cls) 
                loss_dict2['loss_Co'].backward(retain_graph=True)  # 保留计算图以便后续对抗训练
                train_loss += loss_dict2['loss_Co'].item()

                # === 对抗训练 ===
                source_label = 0
                target_label = 1

                # 1. 通过 discriminator 欺骗判别器
                for param in d_style.parameters():
                    param.requires_grad = False
                d_df_out_main = d_style(style_m)
                loss_adv_df_trg_main = bce_loss(d_df_out_main, source_label)
                (0.0002 * loss_adv_df_trg_main).backward()

                # 2. 训练判别器
                for param in d_style.parameters():
                    param.requires_grad = True

                # source
                d_df_out_main = d_style(style_f.detach())
                loss_d_feature_main = bce_loss(d_df_out_main, source_label)
                loss_d_feature_main.backward()

                # target
                d_df_out_main = d_style(style_m.detach())
                loss_d_feature_main = bce_loss(d_df_out_main, target_label)
                loss_d_feature_main.backward()

            # 更新参数
            optimizer.step()
            optimizer_d_style.step()

            if (batch_id + 1) % 20 == 0:
                print(f"Pretrain Epoch {epoch+1}, Batch {batch_id+1}, avg loss: {train_loss/(batch_id+1)}")

        print(f"Pretrain Epoch {epoch+1} finished, avg loss: {train_loss/len(loaders['pretrain'])}")

        # 保存最后一轮模型权重
        if epoch == num_epochs - 1:
            pretrain_model_path = os.path.join(log_dir, f"pretraind_epoch_{epoch+1}.pth")
            print(f"Saving pre-trained model to {pretrain_model_path}")
            torch.save({
                'model_full': model_full.state_dict(),
                'model_missing': model_missing.state_dict(),
                'd_style': d_style.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, pretrain_model_path)

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
    for phase in ['test']:
        loader = loaders[phase]
        print('len(loaders["test"]) =', len(loaders['test']))
        for batch_id, (batch_x, batch_y, cls, _, _) in tqdm(enumerate(loader), total=len(loader), desc="test Batches"):
            batch_x, batch_y = batch_x.cuda(non_blocking=True), batch_y.cuda(non_blocking=True)
            with torch.set_grad_enabled(False):
                with use_ema(model_missing, ema_model_missing):
                    # 固定缺失模态
                    mask = masks_test[rb]
                    mask_tensor = torch.tensor(mask, dtype=torch.float32).view(1, 4, 1, 1, 1)
                    mask_tensor = mask_tensor.cuda(non_blocking=True)
                    batch_xn = batch_x * mask_tensor
                    seg_m, _, _, _, _, cls_prob_m = model_missing(batch_xn[:, 0:], mask)
            gate = (cls_prob_m > 0.5).float()
            gate = gate.view(gate.size(0), gate.size(1), 1, 1, 1)
            seg_m = seg_m[:, :3] * gate
            # gate = (cls_prob_m > 0.5).float()
            # gate = gate.view(gate.size(0), gate.size(1), 1, 1, 1)
            # seg_m = seg_m[:, :3] * gate

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

# ===== 预训练 =====
pretrain(
    model_full, model_missing, d_style, loaders, optimizer, losses_kd,
    num_epochs=10, device='cuda', log_dir=log_dir, epoch_to_load=10  # 设置epoch_to_load为None表示没有预训练文件时从头开始
)

train_val(model_full, model_missing, d_style, loaders, optimizer, scheduler, losses, ema_model_full, ema_model_missing, epoch, best_dice)
print('Training process is finished')

if os.path.exists(saved_model_path):
    model_full, model_missing, d_style, optimizer, epoch, best_dice, ema_model_full, ema_model_missing = load_old_model_double(model_full, model_missing, d_style, optimizer, saved_model_path)
for i in range(0,15):
    test_val(model_missing, loaders, i)
    print('Testing process is finished')