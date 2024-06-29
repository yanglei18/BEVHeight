# Copyright (c) Megvii Inc. All rights reserved.
from argparse import ArgumentParser, Namespace

import os
import torch
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint
from utils.backup_files import backup_codebase

from exps.kitti.bev_height_lss_r101_384_1280_256x256 import \
    BEVHeightLightningModel as BaseBEVHeightLightningModel

H = 384
W = 1280
final_dim = (384, 1280)
img_conf = dict(img_mean=[123.675, 116.28, 103.53],
                img_std=[58.395, 57.12, 57.375],
                to_rgb=True)
model_type = 2 # 0: BEVDepth, 1: BEVHeight, 2: BEVHeight++

return_depth = True
data_root = "data/kitti/"
gt_label_path = "data/kitti/training/label_2"
bev_dim = 160 if model_type==2 else 80
 
backbone_conf = {
    'x_bound': [0, 102.4, 0.4],
    'y_bound': [-51.2, 51.2, 0.4],
    'z_bound': [-5, 3, 8],
    'd_bound': [1.0, 102.0, 0.5],
    'h_bound': [-2.0, 3.0, 80],
    'model_type': model_type,
    'final_dim':
    final_dim,
    'output_channels':
    80,
    'downsample_factor':
    16,
    'img_backbone_conf':
    dict(
        type='ResNet',
        depth=101,
        frozen_stages=0,
        out_indices=[0, 1, 2, 3],
        norm_eval=False,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet101'),
    ),
    'img_neck_conf':
    dict(
        type='SECONDFPN',
        in_channels=[256, 512, 1024, 2048],
        upsample_strides=[0.25, 0.5, 1, 2],
        out_channels=[128, 128, 128, 128],
    ),
    'height_net_conf':
    dict(in_channels=512, mid_channels=512)
}

bev_backbone = dict(
    type='ResNet',
    in_channels = bev_dim,
    depth=18,
    num_stages=3,
    strides=(1, 2, 2),
    dilations=(1, 1, 1),
    out_indices=[0, 1, 2],
    norm_eval=False,
    base_channels= bev_dim * 2,
)

bev_neck = dict(type='SECONDFPN',
                in_channels=[bev_dim, bev_dim * 2, bev_dim * 4, bev_dim * 8],
                upsample_strides=[1, 2, 4, 8],
                out_channels=[64, 64, 64, 64])

CLASSES = [
    'car',
    # 'truck',
    # 'construction_vehicle',
    # 'bus',
    # 'trailer',
    # 'barrier',
    # 'motorcycle',
    # 'bicycle',
    # 'pedestrian',
    # 'traffic_cone',
]

TASKS = [
    dict(num_class=1, class_names=['car']),
    # dict(num_class=2, class_names=['truck', 'construction_vehicle']),
    # dict(num_class=2, class_names=['bus', 'trailer']),
    # dict(num_class=1, class_names=['barrier']),
    # dict(num_class=2, class_names=['motorcycle', 'bicycle']),
    # dict(num_class=2, class_names=['pedestrian', 'traffic_cone']),
    # dict(num_class=1, class_names=['bicycle']),
]

common_heads = dict(reg=(2, 2),
                    height=(1, 2),
                    dim=(3, 2),
                    rot=(2, 2),
                    vel=(2, 2))

bbox_coder = dict(
    type='CenterPointBBoxCoder',
    post_center_range=[0.0, -51.2, -10.0, 102.4, 51.2, 10.0],
    max_num=500,
    score_threshold=0.1,
    out_size_factor=4,
    voxel_size=[0.1, 0.1, 8],
    pc_range=[0, -51.2, -5, 102.4, 51.2, 3],
    code_size=9,
)

train_cfg = dict(
    point_cloud_range=[0, -51.2, -5, 102.4, 51.2, 3],
    grid_size=[1024, 1024, 1],
    voxel_size=[0.1, 0.1, 8],
    out_size_factor=4,
    dense_reg=1,
    gaussian_overlap=0.1,
    max_objs=500,
    min_radius=2,
    code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5],
)

test_cfg = dict(
    post_center_limit_range=[0.0, -51.2, -10.0, 102.4, 51.2, 10.0],
    max_per_img=500,
    max_pool_nms=False,
    min_radius=[4, 12, 10, 1, 0.85, 0.175],
    score_threshold=0.1,
    out_size_factor=4,
    voxel_size=[0.1, 0.1, 8],
    nms_type='circle',
    pre_max_size=1000,
    post_max_size=83,
    nms_thr=0.2,
)

head_conf = {
    'bev_backbone_conf': bev_backbone,
    'bev_neck_conf': bev_neck,
    'tasks': TASKS,
    'common_heads': common_heads,
    'bbox_coder': bbox_coder,
    'train_cfg': train_cfg,
    'test_cfg': test_cfg,
    'in_channels': 256,  # Equal to bev_neck output_channels.
    'loss_cls': dict(type='GaussianFocalLoss', reduction='mean'),
    'loss_bbox': dict(type='L1Loss', reduction='mean', loss_weight=0.25),
    'gaussian_overlap': 0.1,
    'min_radius': 2,
}

class BEVHeightPlusLightningModel(BaseBEVHeightLightningModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data_root = data_root
        self.class_names = CLASSES
        self.backbone_conf = backbone_conf
        self.head_conf = head_conf
        self.return_depth = return_depth
        
def main(args: Namespace) -> None:
    if args.seed is not None:
        pl.seed_everything(args.seed)
    print(args)
    
    model = BEVHeightPlusLightningModel(**vars(args))
    checkpoint_callback = ModelCheckpoint(dirpath='./outputs/bev_height_plus_lss_r101_384_1280_256x256/checkpoints', filename='{epoch}', every_n_epochs=5, save_last=True, save_top_k=-1)
    trainer = pl.Trainer.from_argparse_args(args, callbacks=[checkpoint_callback])
    if args.evaluate:
        for ckpt_name in os.listdir(args.ckpt_path):
            model_pth = os.path.join(args.ckpt_path, ckpt_name)
            trainer.test(model, ckpt_path=model_pth)
    else:
        backup_codebase(os.path.join('./outputs/bev_height_plus_lss_r101_384_1280_256x256', 'backup'))
        '''
        if os.path.exists("pretrain_ckpt/bevheight_plus_pretrain_car.ckpt"):
            model = BEVHeightPlusLightningModel.load_from_checkpoint("pretrain_ckpt/bevheight_plus_pretrain_car.ckpt")
        '''
        trainer.fit(model)
        
def run_cli():
    parent_parser = ArgumentParser(add_help=False)
    parent_parser = pl.Trainer.add_argparse_args(parent_parser)
    parent_parser.add_argument('-e',
                               '--evaluate',
                               dest='evaluate',
                               action='store_true',
                               help='evaluate model on validation set')
    parent_parser.add_argument('-b', '--batch_size_per_device', type=int)
    parent_parser.add_argument('--seed',
                               type=int,
                               default=0,
                               help='seed for initializing training.')
    parent_parser.add_argument('--ckpt_path', type=str)
    parser = BEVHeightPlusLightningModel.add_model_specific_args(parent_parser)
    parser.set_defaults(
        profiler='simple',
        deterministic=False,
        max_epochs=50,
        accelerator='ddp',
        num_sanity_val_steps=0,
        gradient_clip_val=5,
        limit_val_batches=0,
        enable_checkpointing=True,
        precision=32,
        default_root_dir='./outputs/bev_height_plus_lss_r101_384_1280_256x256')
    args = parser.parse_args()
    main(args)

if __name__ == '__main__':
    run_cli()
