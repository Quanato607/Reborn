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
task_name = 'brats_1_smunet'
dataset = task_name.split('_')[0]
config = yaml.load(open(os.path.join('./configs', (dataset + '.yml'))), Loader=yaml.FullLoader)
init_env('2')
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

                    wt,et,tc=measure_dice_score(seg_m, y)
                    dice+=(wt+et+tc)/3.0
                    dice_wt+=wt
                    dice_et+=et
                    dice_tc+=tc

                    hdwt,hdet,hdtc= measure_hd95(seg_m, y)
                    hd95+=(hdwt+hdet+hdtc)/3.0
                    hd95_wt+=hdwt
                    hd95_et+=hdet
                    hd95_tc+=hdtc

                    continue
                    
                
                with torch.set_grad_enabled(phase == 'train'):

                    seg_m, _, _ = model_missing(x_m[:,0:])
                    loss_dict = losses['co_loss'](config, seg_m, y_m)
                    if phase == 'train':
                        (loss_dict['loss_Co']).backward()
                        train_loss += loss_dict['loss_Co'].item()


                if phase == 'train':
                    optimizer.step()
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
                    state['model_full'] = model_full.state_dict()
                    state['model_missing'] = model_missing.state_dict()
                    state['d_style'] = d_style.state_dict()
                    state['optimizer'] = optimizer.state_dict()
                    state['epochs'] = epoch
                    state['dice'] = dice
                    state['best_dice'] = best_dice
                    file_name = log_dir+'/model_best.pth'
                    torch.save(state, file_name)
                    best_dice = dice

train_val(model_full, model_missing, d_style, loaders, optimizer, scheduler, losses, epoch, 0.0, pretrain_full_path)
print('Training process is finished')