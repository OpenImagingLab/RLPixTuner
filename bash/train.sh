#!/bin/bash

export PYTHONPATH=.;
export WANDB_MODE=online;

CUDA_VISIBLE_DEVICES=0 python envs/run_sb3.py --save_path "experiments" --agent custom_isp_single_ddpg --model_path "/input/your/path/to/dataset" --dataset_dir "/path/to/your/dataset" --agent_lr 5e-4 --value_lr 5e-5 --isp_inp_original False --param_noise_std 0.02 --target_policy_noise 0.04 --max_step 10 --replay_size 16384 --joint_obs False --truncate_param True --truncate_retouch_mean True --epochs 1000 --train_freq -1 --ema_rate 0.993 --ddpg_gamma 0.90 --loss_type psnr --reward_scale 0.01 --wb_param_dist_mode 1 --isp dataset --eval_use_best_img False --obs_stack_step True --obs_stack_stop True --timestep 800000 --use_exposure_cnn "False" --use_hdrnet_cnn True --cnn_output_dim 256 --concat_hist False --concat_hist_type "yuv" --obs_inp_laplacian 3
