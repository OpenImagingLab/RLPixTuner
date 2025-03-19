#!/bin/bash

export PYTHONPATH=.;
export WANDB_MODE=online;

CUDA_VISIBLE_DEVICES=0 python envs/run_sb3_style.py --save_path "experiments" --agent custom_isp_single_ddpg --model_path "/input/your/path/to/model" --dataset_dir "/input/your/path/to/dataset" --isp_inp_original False --param_noise_std 0.02 --target_policy_noise 0.04 --max_step 10 --replay_size 16384 --joint_obs False --truncate_param True --truncate_retouch_mean True --epochs 1000 --train_freq -1 --ema_rate 0.993 --ddpg_gamma 0.90 --loss_type style --loss_type_content vgg --loss_type_style gram  --loss_type_cont_target first --loss_type_hist yuv --reward_scale 1 --wb_param_dist_mode 1 --env_img_sz 512 --isp style-0 --eval_use_best_img False --obs_stack_step True --obs_stack_stop True --timestep 500000 --use_exposure_cnn True --obs_history_action False --obs_img_mean_rgb False --concat_hist False --concat_hist_type yuv
