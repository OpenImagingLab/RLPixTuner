import os
import shutil
import datetime
import cv2
import yaml
import random
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import sys

from config import cfg
from isp.filters import *


def set_random_seed(seed=1, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    import argparse
    import wandb

    set_random_seed(42)

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
    parser.add_argument("--ema_rate", type=float, default=0.995, help="learning rate")

    parser.add_argument('--weights', type=str, default='../pretrained_models/yolov3-torch/yolov3-spp.pt',
                        help='yolov3 pretrained path')
    parser.add_argument('--yolo_cfg', type=str, default='yolov3/models/yolov3-spp.yaml', help='model.yaml path')
    parser.add_argument('--hyp', type=str, default='yolov3/data/hyps/hyp.scratch-low.yaml', help='hyperparameters path')

    parser.add_argument("--save_path", type=str, default='dynamic-isp-train-whole-0917',
                        help="save path at experiments/save_path/")
    parser.add_argument("--data_name", type=str, default='isp_diff', choices=['coco', 'lod', 'oprd', 'isp_diff'],
                        help="train data: coco, lod")
    parser.add_argument('--data_cfg', type=str, default='yolov3/data/coco-2017.yaml', help='dataset.yaml path')
    parser.add_argument("--add_noise", type=str2bool, default=True, help="add_noise")
    parser.add_argument("--use_linear", action='store_true', default=False, help="use linear noise distribution")
    parser.add_argument("--bri_range", type=float, default=None, nargs='*',
                        help="brightness range, (low, high), 0.0~1.0")
    parser.add_argument("--noise_level", type=float, default=None, help="noise_level, 0.001~0.012")

    parser.add_argument('--use_truncated', type=str2bool, default=True, help='use_truncated')
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
                                 'custom_isp_single_ddpg'],
                        help='agent choice, for selection of RL train script & agent')
    parser.add_argument("--params_init", type=str, default="random",
                        choices=['random', 'fix'])

    parser.add_argument("--max_step", type=int, default=5)
    parser.add_argument("--timestep", type=int, default=100000)
    parser.add_argument("--obs_stack_ori", type=str2bool, default=False)
    parser.add_argument("--obs_stack_stop", type=str2bool, default=True)
    parser.add_argument("--obs_stack_step", type=str2bool, default=True)
    parser.add_argument("--obs_history_action", type=str2bool, default=False)
    parser.add_argument("--obs_img_mean_rgb", type=str2bool, default=False)
    parser.add_argument("--obs_inp_laplacian", type=int, default=0, choices=[0, 1, 2, 3], help="0 no laplacian; 1 first layer; 2 last layer; 3 all pyramid")
    parser.add_argument("--joint_obs", type=str2bool, default=True)
    parser.add_argument("--truncate_param", type=str2bool, default=False)
    parser.add_argument("--truncate_retouch_mean", type=str2bool, default=False)
    parser.add_argument("--gradient_clip", type=str2bool, default=True)
    parser.add_argument("--eval_use_best_img", type=str2bool, default=False)
    parser.add_argument("--use_exposure_cnn", type=str2bool, default=False)
    parser.add_argument("--use_hdrnet_cnn", type=str2bool, default=False)
    parser.add_argument("--concat_hist", type=str2bool, default=False)
    parser.add_argument("--concat_hist_type", type=str, default="rgb", choices=["rgb", "yuv", "None"])

    parser.add_argument("--vec_env", type=str2bool, default=False)

    parser.add_argument("--param_noise_std", type=float, default=0.1)
    parser.add_argument("--use_param_noise_schedule", type=str2bool, default=False)
    parser.add_argument("--isp_inp_original", type=str2bool, default=True)
    parser.add_argument("--param_noise_schedule_initial", type=float, default=0.5)
    parser.add_argument("--wandb_proj", type=str, default='rlisp-sb3')
    parser.add_argument("--isp", type=str, default="wb", choices=["wb", "exp", "cont", "exp-wb-cont", "dataset"])
    parser.add_argument("--net_arch", type=str, default="default", choices=["small", "big", "large", "default", "plus", "max", "ultra"])
    parser.add_argument("--loss_type", type=str, default="l2", choices=["l2", "l1", "psnr", "default"])
    parser.add_argument("--reward_scale", type=float, default=1.0)
    parser.add_argument("--target_policy_noise", type=float, default=0.2)
    parser.add_argument("--wb_param_dist_mode", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--vis_buffer_dist", type=bool, default=True)
    parser.add_argument("--train_freq", type=int, default=-1)
    parser.add_argument("--env_img_sz", type=int, default=256)
    parser.add_argument("--eval_save_freq", type=int, default=2000)
    parser.add_argument("--cnn_output_dim", type=int, default=256)

    args = parser.parse_args()

    wandb_config = {"isp": args.isp,
                    "dataset_dir": args.dataset_dir,
                    "agent_lr": args.agent_lr,
                    "value_lr": args.value_lr,
                    "batch_size": args.batch_size,
                    "replay_size": args.replay_size,
                    "max_step": args.max_step,
                    "param_noise_std": args.param_noise_std,
                    "truncate_param": args.truncate_param,
                    "truncate_retouch_mean": args.truncate_retouch_mean,
                    "ddpg_gamma": args.ddpg_gamma,
                    "ema_rate": args.ema_rate,
                    "isp_inp_original": args.isp_inp_original,
                    "net_arch": args.net_arch,
                    "loss_type": args.loss_type,
                    "reward_scale": args.reward_scale,
                    "target_policy_noise": args.target_policy_noise,
                    "concat_hist": args.concat_hist,
                    "concat_hist_type": args.concat_hist_type,
                    "train_freq": args.train_freq,
                    "env_img_sz": args.env_img_sz,
                    "eval_use_best_img": args.eval_use_best_img,
                    "obs_stack_stop": args.obs_stack_stop,
                    "obs_stack_step": args.obs_stack_step,
                    "obs_history_action": args.obs_history_action,
                    "obs_img_mean_rgb": args.obs_img_mean_rgb,
                    "obs_inp_laplacian": args.obs_inp_laplacian,
                    "use_exposure_cnn": args.use_exposure_cnn,
                    "use_hdrnet_cnn": args.use_hdrnet_cnn,
                    "cnn_output_dim": args.cnn_output_dim,
                    }
    wandb_path = os.path.join(os.getcwd(), 'wandb', '__test', args.save_path)
    os.makedirs(wandb_path)
    wandb_run = wandb.init(project=args.wandb_proj, sync_tensorboard=True, config=wandb_config, dir=wandb_path)
    torch.autograd.set_detect_anomaly(True)

    cfg.test_steps = args.max_step
    args.save_path = args.data_name + '-' + args.save_path
    if args.mem_traj_keep_len is None:
        args.mem_traj_keep_len = args.max_step + 1
    if args.isp == "wb":
        cfg.custom_isp = [ImprovedWhiteBalanceFilter]
    elif args.isp == "exp-wb-cont":
        cfg.custom_isp = [ExposureFilter, ImprovedWhiteBalanceFilter, ContrastFilter]
    elif args.isp == "dataset":
        dataset_name = os.path.basename(args.dataset_dir)
        if dataset_name.startswith("exp_out"):
            cfg.custom_isp = [ExposureFilter]
        elif dataset_name.startswith("sat_out"):
            cfg.custom_isp = [SaturationFilter]
        elif dataset_name.startswith("exp_wb_sat_chs_sharp"):
            cfg.custom_isp = [ExposureFilter, ImprovedWhiteBalanceFilter, SaturationFilter, ContrastFilter,
                              HighlightFilter, ShadowFilter, SharpenFilter]
        elif dataset_name.startswith("exp_wb_sat_chs"):
            cfg.custom_isp = [ExposureFilter, ImprovedWhiteBalanceFilter, SaturationFilter, ContrastFilter, HighlightFilter, ShadowFilter]
        elif dataset_name.startswith("exp_wb_sat"):
            cfg.custom_isp = [ExposureFilter, ImprovedWhiteBalanceFilter, SaturationFilter]
        else:
            raise NotImplementedError("unsupported isp pipeline from dataset")
    else:
        raise NotImplementedError("unsupported isp pipeline")
    cfg.filter_param_noise_std = args.param_noise_std
    cfg.use_param_noise_schedule = args.use_param_noise_schedule
    cfg.use_exposure_cnn = args.use_exposure_cnn
    cfg.use_hdrnet_cnn = args.use_hdrnet_cnn
    cfg.change_param_dist_cfg["ImprovedWhiteBalanceFilter"] = args.wb_param_dist_mode

    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.logger import configure
    from envs.isp_env import ISPEnv
    from isp_blocks import ISPBlocks

    isp_blocks = ISPBlocks(cfg, is_blackbox=True)
    isp_blocks.init_filters(cfg.custom_isp)
    env_args_dict = {
        'args': args,
        'is_train': True,
        'data_path': args.dataset_dir + '/train',
        'image_size': args.env_img_sz,
        'isp_blocks': isp_blocks,
        'max_step': args.max_step,
        'obs_stack_ori': args.obs_stack_ori,
        'obs_stack_step': args.obs_stack_step,
        'obs_stack_stop': args.obs_stack_stop,
        'obs_history_action': args.obs_history_action,
        'obs_img_mean_rgb': args.obs_img_mean_rgb,
        'obs_inp_laplacian': args.obs_inp_laplacian,
        'joint_obs': args.joint_obs,
        'truncate_param': args.truncate_param,
        'truncate_retouch_mean': args.truncate_retouch_mean,
        'isp_inp_original': args.isp_inp_original,
        'loss_type': args.loss_type,
        'reward_scale': args.reward_scale,
    }
    if not args.vec_env:
        env = ISPEnv(cfg, **env_args_dict)
        check_env(env)
        env.reset()
    else:
        from dataset import LoadImagesAndLabelsRAWReplay_target
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
        train_dataset = LoadImagesAndLabelsRAWReplay_target(
            args.dataset_dir + '/train',
            args.env_img_sz,
            batch_size=1,
            augment=False,  # augmentation
            hyp=None,  # hyperparameters
            rect=False,  # rectangular batches
        )
        env_args_dict['specified_dataset'] = train_dataset
        def make_env():
            def _init():
                env = ISPEnv(cfg, **env_args_dict)
                check_env(env)  # Optional: checks if the environment follows the Gym interface
                env.reset()
                return env
            return _init
        env_fns = [make_env() for _ in range(5)]
        env = DummyVecEnv(env_fns)

    eval_env = ISPEnv(cfg,
                      args=args,
                      is_train=False,
                      data_path=args.dataset_dir + '/val',
                      image_size=args.env_img_sz,
                      isp_blocks=isp_blocks,
                      max_step=args.max_step,
                      obs_stack_ori=args.obs_stack_ori,
                      obs_stack_step=args.obs_stack_step,
                      obs_stack_stop=args.obs_stack_stop,
                      obs_history_action=args.obs_history_action,
                      obs_img_mean_rgb=args.obs_img_mean_rgb,
                      obs_inp_laplacian=args.obs_inp_laplacian,
                      joint_obs=args.joint_obs,
                      truncate_param=args.truncate_param,
                      truncate_retouch_mean=args.truncate_retouch_mean,
                      isp_inp_original=args.isp_inp_original,
                      loss_type=args.loss_type,
                      reward_scale=args.reward_scale,
                      eval_use_best_img=args.eval_use_best_img,
                      save_freq=args.eval_save_freq,
                      )
    eval_env.reset()

    log_pth = os.getcwd() + '/experiments/__sb3_saved_eval/' + args.save_path
    new_logger = configure(log_pth, ["stdout", "csv", "log", "json", "tensorboard"])

    from stable_baselines3 import TD3
    from envs.custom_td3 import CustomTD3
    from envs.custom_callbacks import SaveActionDistributionCallback
    from stable_baselines3.common.callbacks import EvalCallback, CallbackList
    from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

    eval_callback = EvalCallback(eval_env,
                                 best_model_save_path=log_pth+"/eval/",
                                 log_path=log_pth+"/eval/",
                                 eval_freq=1000,  # mostly 500
                                 deterministic=True,
                                 n_eval_episodes=len(eval_env.dataset),
                                 render=False)
    callbacks = eval_callback
    if args.vis_buffer_dist:
        vis_save_dir = os.path.join('experiments/__sb3_saved_eval', args.save_path, "dist")
        os.makedirs(vis_save_dir, exist_ok=True)
        vis_buffer_callback = SaveActionDistributionCallback(
            save_freq=1000, action_dim=env.action_space_dim, save_dir=vis_save_dir
        )
        callbacks = CallbackList([eval_callback, vis_buffer_callback])

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=args.param_noise_std * np.ones(n_actions))
    train_freq = (1, "episode") if (args.train_freq == -1) else args.train_freq
    if args.net_arch == "default":
        policy_kwargs = {
            "features_extractor_kwargs": {"normalized_image": True},
        }
    else:
        policy_kwargs = {
            "features_extractor_kwargs": {"normalized_image": True, "cnn_output_dim": args.cnn_output_dim},
            "net_arch": {"small": [256, 256], "big": [400, 300], "large": [256, 256, 256], "plus": [512, 512, 512], "max": [256, 256, 256, 256], "ultra": [512, 512, 512, 512]}[args.net_arch],
        }
        if args.concat_hist:
            from envs.custom_style_torch_layers import CombinedExtractor
            policy_kwargs["features_extractor_class"] = CombinedExtractor
            policy_kwargs["features_extractor_kwargs"] = {"normalized_image": True, "cnn_output_dim": args.cnn_output_dim,
                                                          "concat_hist": True, "hist_type": args.concat_hist_type}

    model = CustomTD3("CnnPolicy" if args.joint_obs else "MultiInputPolicy",  
                      env,
                      learning_rate=args.agent_lr,
                      buffer_size=args.replay_size,
                      learning_starts=100,
                      batch_size=args.batch_size,
                      tau=1-args.ema_rate,
                      gamma=args.ddpg_gamma,
                      train_freq=train_freq,
                      action_noise=action_noise,
                      policy_kwargs=policy_kwargs,
                      verbose=1,
                      actor_lr=args.agent_lr,
                      critic_lr=args.value_lr,
                      gradient_clip=args.gradient_clip,
                      target_policy_noise=args.target_policy_noise,
                      target_noise_clip=args.target_policy_noise * 0.25,
                      )
    model.set_logger(new_logger)
    print("start learning")
    model.learn(total_timesteps=args.timestep, log_interval=50, callback=callbacks)
