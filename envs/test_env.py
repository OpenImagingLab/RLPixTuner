import os
import shutil
import datetime
import cv2
import yaml
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import sys

from config import cfg


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default='train_val', help="train, train and val, val")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--epochs", type=int, default=400, help="epochs")
    parser.add_argument("--patience", type=int, default=20, help="early stopping patience")
    parser.add_argument("--lr", type=float, default=3e-5, help="learning rate")
    parser.add_argument("--scheduler_step_size", type=int, default=20, help="scheduler_step_size")
    parser.add_argument("--scheduler_lr_gamma", type=float, default=0.5, help="scheduler_lr_gamma")
    parser.add_argument("--imgsz", type=int, default=512, help="image size")
    parser.add_argument("--workers", type=int, default=4, help="workers")

    parser.add_argument("--agent_lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--value_lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--lr_decay", type=bool, default=True, help="learning rate")
    parser.add_argument("--ddpg_gamma", type=float, default=0.99, help="learning rate")

    parser.add_argument('--weights', type=str, default='../pretrained_models/yolov3-torch/yolov3-spp.pt',
                        help='yolov3 pretrained path')
    parser.add_argument('--yolo_cfg', type=str, default='yolov3/models/yolov3-spp.yaml', help='model.yaml path')
    parser.add_argument('--hyp', type=str, default='yolov3/data/hyps/hyp.scratch-low.yaml', help='hyperparameters path')

    parser.add_argument("--save_path", type=str, default='dynamic-isp-train-whole-0917',
                        help="save path at experiments/save_path/")
    parser.add_argument("--data_name", type=str, default='isp_diff', choices=['coco', 'lod', 'oprd', 'isp_diff'],
                        help="train data: coco, lod")
    parser.add_argument('--data_cfg', type=str, default='yolov3/data/coco-2017.yaml', help='dataset.yaml path')
    parser.add_argument("--add_noise", type=bool, default=True, help="add_noise")
    parser.add_argument("--use_linear", action='store_true', default=False, help="use linear noise distribution")
    parser.add_argument("--bri_range", type=float, default=None, nargs='*',
                        help="brightness range, (low, high), 0.0~1.0")
    parser.add_argument("--noise_level", type=float, default=None, help="noise_level, 0.001~0.012")

    parser.add_argument('--use_truncated', type=bool, default=True, help='use_truncated')
    parser.add_argument('--resume', type=str, default=None, help='resume model weights')

    parser.add_argument('--model_weights', type=str,
                        default='experiments/coco-dynamic-isp-train-whole-bri-20-100-0920/ckpt/DynamicISP_iter_50000.pth',
                        help='isp model weight')
    parser.add_argument("--val_save_path", type=str, default='experiments/coco-dynamic-isp-train-whole-bri-20-100-0920')
    parser.add_argument("--val_batch_size", type=int, default=200)
    parser.add_argument("--steps", type=int, default=5, help="useless")
    parser.add_argument('--dataset_dir', type=str, default=None, help='dataset dir, with train and val')
    parser.add_argument("--mem_traj_keep_len", type=int, default=None, help="steps")
    parser.add_argument("--replay_size", type=int, default=128, help="replay memory size")
    parser.add_argument('--agent', type=str, default='original',
                        choices=['original',
                                 'custom_isp_single',
                                 'custom_isp_single_rounds',
                                 'custom_isp_single_dpg_wo_grad',
                                 'custom_isp_single_ddpg'], help='agent choice, for selection of RL train script & agent')
    parser.add_argument("--params_init", type=str, default="random",
                        choices=['random', 'fix'])

    parser.add_argument("--max_step", type=int, default=5)

    parser.add_argument("--param_noise_std", type=float, default=0.0)
    parser.add_argument("--use_param_noise_schedule", type=bool, default=False)
    parser.add_argument("--isp_inp_original", type=bool, default=False)
    parser.add_argument("--param_noise_schedule_initial", type=float, default=0.5)


    args = parser.parse_args()
    cfg.test_steps = args.max_step
    args.save_path = args.data_name + '-' + args.save_path
    if args.data_name in ("lod", 'oprd'):
        args.add_noise = False
        args.bri_range = None
        args.use_linear = False
    if args.mem_traj_keep_len is None:
        args.mem_traj_keep_len = args.max_step + 1
    cfg.filter_param_noise_std = args.param_noise_std
    cfg.use_param_noise_schedule = args.use_param_noise_schedule

    from stable_baselines3.common.env_checker import check_env
    from envs.isp_env import ISPEnv
    from isp_blocks import ISPBlocks

    isp_blocks = ISPBlocks(cfg, is_blackbox=True)
    isp_blocks.init_filters(cfg.custom_isp)
    env = ISPEnv(cfg, data_path=args.dataset_dir + '/train', isp_blocks=isp_blocks, max_step=3)
    check_env(env)
    env.reset()
    for i in range(4):
        env.print_status()
        env.step(np.random.rand(env.action_space_dim))
        print(f"---------step {i}------------------------------------------------------")
    env.print_status()
    env.reset()
    print("------------------clear------------------------------------------------------")
    env.print_status()
