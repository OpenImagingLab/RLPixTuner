from util import Dict
from isp.filters import *


cfg = Dict()

cfg.print_log_freq = 10
cfg.val_freq = 1000
cfg.save_model_freq = 1000
cfg.print_freq = 100
cfg.summary_freq = 100
cfg.show_img_num = 2

cfg.parameter_lr_mul = 1
cfg.value_lr_mul = 1  # TODO 10
cfg.critic_lr_mul = 1

cfg.agent_lr = 3e-4
cfg.value_lr = 3e-4
cfg.lr_decay = True

###########################################################################
# Dataset -- explicitly for l2 loss based
###########################################################################
cfg.train_path = "/mnt/data/sail_3090/wujiarui/data/distilled_jpg_pair/" + "train"
cfg.val_path = "/mnt/data/sail_3090/wujiarui/data/distilled_jpg_pair/" + "val"


###########################################################################
###########################################################################
# Here is a list of parameters. Instead of hard coding them in the script, I summarize them here.
# You do not need to modify most of them except the dataset part (see bottom), unless for good reasons.
###########################################################################
###########################################################################

#--------------------------------------------------------------------------

###########################################################################
# Filter Parameters
###########################################################################
# cfg.filters = [
#     ExposureFilter, GammaFilter, ColorFilter,  # CCMFilter, SharpenUSMFilter, DenoiseFilter, SharpenUSMFilter,
#     ToneFilter, ContrastFilter, SaturationPlusFilter, WNBFilter, ImprovedWhiteBalanceFilter
# ]
# cfg.filters = [
#     (ExposureFilter, GammaFilter, CCMFilter, SharpenFilter), DenoiseFilter, # SharpenUSMFilter, ColorFilter
#     ToneFilter, ContrastFilter, SaturationPlusFilter, WNBFilter, ImprovedWhiteBalanceFilter
# ]
cfg.filters = [
    ExposureFilter, GammaFilter, ImprovedWhiteBalanceFilter,
    SaturationPlusFilter, ToneFilter, ContrastFilter, WNBFilter, ColorFilter
]
cfg.custom_isp = [
    ImprovedWhiteBalanceFilter
]

# agent
cfg.isp_inp_last_step = True
cfg.params_random_init = True

# value
cfg.value_inp_skip_last_params = False
cfg.change_param_dist_cfg = {
    "ImprovedWhiteBalanceFilter": 0, "ExposureFilter": 0, "ContrastFilter": 0,
}
# param sel noise
cfg.use_param_noise_schedule = False
cfg.filter_param_noise_std = 0.2
cfg.param_noise_stds = [0.4, 0.3, 0.2, 0.1, 0.0, 0.0, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
cfg.param_noise_scales = [1., 0.5, 0.25, 0.1, 0.01, 0., 0., 0., 0., 0., 0., 0., 0., 0., ]
# Gamma = 1/x ~ x
cfg.curve_steps = 8
cfg.gamma_range = 3
# cfg.exposure_range = 3.5
cfg.exposure_range = 2.0
cfg.wb_range = 1.1
cfg.wb_param_norm = False
cfg.color_curve_range = (0.90, 1.10)
cfg.lab_curve_range = (0.90, 1.10)
cfg.tone_curve_range = (0.5, 2)
cfg.usm_sharpen_range = (0.0, 2.0)  # wikipedia recommended sigma 0.5-2.0; amount 0.5-1.5
cfg.sharpen_range = (0.0, 10.0)
cfg.ccm_range = (-2.0, 2.0)
cfg.denoise_range = (0.0, 1.0)

# Masking is DISABLED
cfg.masking = False
cfg.minimum_strength = 0.3
cfg.maximum_sharpness = 1
cfg.clamp = False


###########################################################################
# RL Parameters
###########################################################################
cfg.critic_logit_multiplier = 100  # 0.05 20
cfg.ddpg_gamma = 0.99
cfg.discount_factor = 1.0
# Each time the agent reuse a filter, a penalty is subtracted from the reward. Set to 0 to disable.
cfg.filter_usage_penalty = 1.0
# Use temporal difference error (thereby the value network is used) or directly a single step award (greedy)?
cfg.use_TD = True
# During test, do we use random walk or pick the action with maximized prob.?
cfg.test_random_walk = False
# Replay memory
cfg.replay_memory_size = 128
# Note, a trajectory will be killed either after achieving this value (by chance) or submission
# Thus exploration will lead to kills as well.
cfg.maximum_trajectory_length = 7
cfg.over_length_keep_prob = 0.5
cfg.all_reward = 1.0
# Append input image with states?
cfg.img_include_states = True
# with prob. cfg.exploration, we randomly pick one action during training
cfg.exploration = 0.05
# Action entropy penalization
cfg.exploration_penalty = 0.05
cfg.early_stop_penalty = 1.0
cfg.detect_loss_weight = 1.0

###########################################################################
# CNN Parameters
###########################################################################
cfg.use_exposure_cnn = False
cfg.use_hdrnet_cnn = False
cfg.source_img_size = 512
cfg.base_channels = 32
cfg.dropout_keep_prob = 0.5
cfg.agent_input_target = True
cfg.value_input_target = True
# G and C use the same feed dict?
cfg.share_feed_dict = True
cfg.shared_feature_extractor = True
cfg.fc1_size = 128
cfg.bnw = False
# number of filters for the first convolutional layers for all networks
#                      (stochastic/deterministic policy, critic, value)
cfg.feature_extractor_dims = 4096

###########################################################################
# GAN Parameters
###########################################################################
# For WGAN only
cfg.use_penalty = True
# LSGAN or WGAN? (LSGAN is not supported now, so please do not change this)
cfg.gan = 'w'

##################################
# Generator
##################################
cfg.giters = 1

##################################
# Critic & Value Networks
##################################
cfg.gradient_penalty_lambda = 10
# max iter step, note the one step indicates that a Citers updates of critic and one update of generator
cfg.citers = 5
cfg.critic_initialization = 10
# the upper bound and lower bound of parameters in critic
# when using gradient penalty, clamping is disabled
cfg.clamp_critic = 0.01

# EMD output filter size
cfg.median_filter_size = 101

# Noise defined here is not actually used
cfg.z_type = 'uniform'
cfg.z_dim_per_filter = 16

# TODO - cfg actually used
cfg.num_state_dim = 3 + len(cfg.filters)
cfg.z_dim = 3 + len(cfg.filters) * cfg.z_dim_per_filter
cfg.test_steps = 5

cfg.real_img_size = 512
cfg.real_img_channels = 1 if cfg.bnw else 3

###########################################################################
# Training
###########################################################################
# cfg.supervised = False
# multiplier = 2
# cfg.max_iter_step = int(5000 * multiplier)

cfg.num_samples = 8
cfg.img_channels = 1 if cfg.bnw else 3


##################################
# Debugging Outputs
##################################
cfg.vis_draw_critic_scores = True
cfg.vis_step_test = False
cfg.realtime_vis = False
cfg.write_image_interval = 10000
cfg.debug_step_params = True