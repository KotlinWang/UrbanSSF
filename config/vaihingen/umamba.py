from torch.utils.data import DataLoader
from geoseg.losses import *
from geoseg.datasets.vaihingen_dataset import *
# from geoseg.models.UNetFormer import UNetFormer
# from geoseg.models.baseline.BaseLine import Ablation
# from geoseg.models.baseline.new_baseline import UNet
# from geoseg.models.baseline.baseline import UNetFormer
# from geoseg.models.umamba_v2.umamba_v2 import UMamba
# from geoseg.models.umamba_v3.umamba_v3 import UMamba
# from geoseg.models.umamba_v9.umamba_v9 import UMamba
# from geoseg.models.baseline.baseline_SRT_CSR_SFE.baseline_srt_csr_sfe import UMamba
# from geoseg.models.urbanssf_t.urbanssf_t import UMamba
from geoseg.models.urbanssf_s_69.umamba_s import UMamba
from catalyst.contrib.nn import Lookahead
from catalyst import utils

# training hparam
max_epoch = 100
ignore_index = len(CLASSES)
train_batch_size = 8
val_batch_size = 8
lr = 6e-4
weight_decay = 0.01
backbone_lr = 6e-5
backbone_weight_decay = 0.01
num_classes = len(CLASSES)
classes = CLASSES

weights_name = "urbanssf_s"
# weights_name = "urbanssf_t"

weights_path = "model_weights/vaihingen/{}".format(weights_name)
test_weights_name = "urbanssf_s"
log_name = 'vaihingen/{}'.format(weights_name)
monitor = 'val_F1'
monitor_mode = 'max'
save_top_k = 1
save_last = True
check_val_every_n_epoch = 1
pretrained_ckpt_path = None # the path for the pretrained model weight
gpus = 'auto'  # default or gpu ids:[0] or gpu nums: 2, more setting can refer to pytorch_lightning
resume_ckpt_path = None  # whether continue training with the checkpoint, default None

#  define the network
# net = UNetFormer(num_classes=num_classes)
# net = Ablation(num_classes=num_classes)
# net = UNet(in_channels=3, num_classes=num_classes)

net = UMamba(num_classes=num_classes)

# define the loss
# loss = UnetFormerLoss(ignore_index=ignore_index)
loss = UMambaLoss(ignore_index=ignore_index)
use_aux_loss = True

# define the dataloader

train_dataset = VaihingenDataset(data_root='/home/kotlin/DATASET/vaihingen/train', mode='train',
                                 mosaic_ratio=0.25, transform=train_aug)

val_dataset = VaihingenDataset(data_root='/home/kotlin/DATASET/vaihingen/test',transform=val_aug)
test_dataset = VaihingenDataset(data_root='/home/kotlin/DATASET/vaihingen/test',
                                transform=val_aug)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=train_batch_size,
                          num_workers=8,
                          pin_memory=True,
                          shuffle=True,
                          drop_last=True)

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=val_batch_size,
                        num_workers=8,
                        shuffle=False,
                        pin_memory=True,
                        drop_last=False)

# define the optimizer
layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
net_params = utils.process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
# lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch)

