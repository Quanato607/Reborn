import yaml
from data import make_data_loaders_divide
from models import build_model
from models.discriminator import get_style_discriminator
from solver import make_optimizer_double
from losses import get_losses, bce_loss, Dice, get_current_consistency_weight, get_losses_kd
import os
import torch
import torch.optim as optim
from utils import *
from tqdm import tqdm
import os, csv, numpy as np 

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
task_name = 'brats_1_smunet_meta_dot1'
dataset = task_name.split('_')[0]
config = yaml.load(open(os.path.join('./configs', (dataset + '.yml'))), Loader=yaml.FullLoader)
init_env('7')
loaders = make_data_loaders_divide(config)
model_full, model_missing = build_model(inp_dim1 = 4, inp_dim2 = 4)
model_full    = model_full.cuda()
model_missing = model_missing.cuda()
d_style       = get_style_discriminator(num_classes = 128).cuda()
optimizer_d_style = optim.Adam(d_style.parameters(), lr=float(config['lr']), betas=(0.9, 0.99))
log_dir = os.path.join(config['path_to_log'], task_name)
optimizer, scheduler = make_optimizer_double(config, model_full, model_missing)
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
    model_full, model_missing, d_style, optimizer, epoch, best_dice = load_old_model_double(model_full, model_missing, d_style, optimizer, latest_model_path)
else:
    best_dice = 0.0

if not os.path.exists(log_dir):
    os.makedirs(log_dir) 

logger = get_logging(os.path.join(log_dir, 'train&valid.log'))

continue_training = False
pretrain_full_path = None

criteria = Dice() 
    
def train_val(model_full, model_missing, d_style, loaders, optimizer, scheduler, losses, epoch_init=1, best_dice=0.0, pretrain_full_path=None):

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
            if phase == 'eval' and epoch%5!=0 :
                continue

            loader = loaders[phase]
            total  = len(loader)

            for batch_id, batch in tqdm(enumerate(loader), total=total, desc=f"{phase}"):
                if phase == 'train':
                    (x_m, y_m, cls_m, mask), (x_f, y_f, cls_f, _) = batch
                    x_m, y_m, cls_m = x_m.cuda(non_blocking=True), y_m.cuda(non_blocking=True), cls_m.cuda(non_blocking=True)
                    x_f, y_f, cls_f = x_f.cuda(non_blocking=True), y_f.cuda(non_blocking=True), cls_f.cuda(non_blocking=True)
                    mask_tensor = mask.cuda(non_blocking=True).to(x_m.dtype)
                elif phase == 'eval':
                    model_full.eval()
                    model_missing.eval()
                    with torch.no_grad():
                        (x, y, cls, mask) = batch
                        x, y, cls = x.cuda(non_blocking=True), y.cuda(non_blocking=True), cls.cuda(non_blocking=True)
                        rb = random.randint(0, 14)
                        mask = masks_test[rb]
                        mask_tensor = (
                            torch.tensor(mask, dtype=torch.float32)
                            .view(1, 4, 1, 1, 1)
                            .cuda(non_blocking=True)
                        )
                        batch_xn = x * mask_tensor
                        seg_m, style_m, content_m = model_missing(batch_xn[:,0:])

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
                    # 缺失模态的鲁棒性损失
                    seg_m, _, _ = model_missing(x_m[:,0:])

                    loss_dict = losses['co_loss'](config, seg_m, y_m)
                    if phase == 'train':
                        (loss_dict['loss_Co']).backward()
                        train_loss += loss_dict['loss_Co'].item()

                    # 全模态样本下的缺失模态模型引导
                    batch_xn = x_f * mask_tensor

                    seg_f, style_f, content_f = model_full(x_f[:,0:])
                    seg_m, style_m, content_m = model_missing(batch_xn[:,0:])
                    loss_dict = losses_kd['co_loss'](config, seg_f, content_f, y_f, seg_m, content_m, style_f, style_m, epoch)
                   
                    d_style.train()
                    optimizer_d_style = optim.Adam(d_style.parameters(), lr = float(config['lr']), betas=(0.9, 0.99))
                    # labels for style adversarial training
                    source_label = 0
                    target_label = 1

                    optimizer.zero_grad()
                    optimizer_d_style.zero_grad()
                    
                    # only train. Don't accumulate grads in disciminators
                    for param in d_style.parameters():
                        param.requires_grad = False

                    if phase == 'train':
                        (loss_dict['loss_Co']).backward(retain_graph=True)
                        train_loss += loss_dict['loss_Co'].item()
                    
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


                if phase == 'train':
                    optimizer.step()
                    optimizer_d_style.step()
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
                    file_name = log_dir+'/model_best.pth'
                    torch.save(state, file_name)
                    best_dice = dice

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
        print('len(loaders["eval"]) =', len(loaders['eval']))
        for batch_id, (batch_x, batch_y, cls, mask) in tqdm(enumerate(loader), total=len(loader), desc="test Batches"):
            batch_x, batch_y = batch_x.cuda(non_blocking=True), batch_y.cuda(non_blocking=True)
            with torch.set_grad_enabled(phase == 'eval'):

                # 固定缺失模态
                mask = masks_test[rb]
                mask_tensor = torch.tensor(mask, dtype=torch.float32).view(1, 4, 1, 1, 1)
                mask_tensor = mask_tensor.cuda(non_blocking=True)
                batch_xn = batch_x * mask_tensor
                seg_m, _, _ = model_missing(batch_xn[:, 0:])

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

# train_val(model_full, model_missing, d_style, loaders, optimizer, scheduler, losses, epoch, 0.0, pretrain_full_path)
# print('Training process is finished')

if os.path.exists(saved_model_path):
    model_full, model_missing, _, optimizer, epoch, best_dice = load_old_model_double(model_full, model_missing, d_style, optimizer, saved_model_path)
for i in range(0,15):
    test_val(model_missing, loaders, i)
    print('Testing process is finished')