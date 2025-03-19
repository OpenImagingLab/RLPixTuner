import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn
import random
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union
import VGG
from tools.hist_loss import histogram_matching_loss

from util import Dict, save_params, save_img, save_params_steps
from isp_blocks import ISPBlocks
from dataset import LoadImagesAndLabelsRAWReplay_target
from criterion import PSNR, structural_similarity_index_measure
import matplotlib.pyplot as plt

env_debug = True

l1loss = nn.L1Loss(reduction="none")


def compute_l2_batch(outputs, targets):
    return torch.mean((targets - outputs) ** 2, dim=(1, 2, 3))


def compute_l2(outputs, targets):
    return torch.mean((targets - outputs) ** 2)


def compute_l1_batch(outputs, targets):
    return torch.mean(l1loss(outputs, targets), dim=(1, 2, 3))


def compute_l1(outputs, targets):
    return torch.mean(l1loss(outputs, targets))


def compute_psnr_batch(outputs, targets):
    value = torch.mean((targets - outputs) ** 2, dim=(1, 2, 3))
    return -10 * torch.log10(value)


def scale_action(action, low, high):
    abto1 = 2.0 * ((action - low) / (high - low)) - 1.0
    _1toab = low + (0.5 * (action + 1.0) * (high - low))


def get_img_specs(images, add_rgb=False):
    lum = (images[:, 0, :, :] * 0.27 + images[:, 1, :, :] * 0.67 + images[:, 2, :, :] * 0.06 + 1e-5)[:, None, :, :]
    luminance = torch.mean(lum, dim=(1, 2, 3))
    contrast = torch.var(lum, dim=(1, 2, 3))
    mean_r = torch.mean(images[:, 0, :, :], dim=(1, 2))
    mean_g = torch.mean(images[:, 1, :, :], dim=(1, 2))
    mean_b = torch.mean(images[:, 2, :, :], dim=(1, 2))
    i_max, _ = torch.max(torch.clip(images, min=0.0, max=1.0), dim=1)
    i_min, _ = torch.min(torch.clip(images, min=0.0, max=1.0), dim=1)
    sat = (i_max - i_min) / (torch.minimum(i_max + i_min, 2.0 - i_max - i_min) + 1e-2)
    saturation = torch.mean(sat, dim=[1, 2])
    repetition = 1
    if add_rgb:
        state_feature = torch.cat(
            [torch.tile(luminance[:, None], [1, repetition]),
             torch.tile(mean_r[:, None], [1, repetition]),
             torch.tile(mean_g[:, None], [1, repetition]),
             torch.tile(mean_b[:, None], [1, repetition]),
             torch.tile(contrast[:, None], [1, repetition]),
             torch.tile(saturation[:, None], [1, repetition])], dim=1)
    else:
        state_feature = torch.cat(
            [torch.tile(luminance[:, None], [1, repetition]),
             torch.tile(contrast[:, None], [1, repetition]),
             torch.tile(saturation[:, None], [1, repetition])], dim=1)
    return state_feature  # shape batch_sz x n


class ISPStyleEnv(gym.Env):
    def __init__(self,
                 cfg=None,
                 args=None,
                 is_train=True,
                 data_path=None,
                 image_size=256,
                 obs_img_shape=(3, 64, 64),
                 batch_size=1,
                 max_step=10,
                 isp_blocks: ISPBlocks = None,
                 input_img_specs=True,
                 specified_dataset=None,
                 obs_stack_ori=False,
                 obs_stack_step=True,
                 obs_stack_stop=True,
                 obs_history_action=False,
                 obs_img_mean_rgb=False,
                 joint_obs=True,
                 truncate_param=False,
                 truncate_retouch_mean=False,
                 isp_inp_original=True,
                 loss_type="style",
                 loss_type_content="vgg",
                 loss_type_style="gram",
                 loss_type_cont_target="last",
                 loss_type_hist="None",
                 loss_coeff_style=100,
                 loss_coeff_hist=100,
                 reward_scale=1.0,
                 action_space_decay: Optional[float] = None,
                 highres_eval=False,
                 eval_use_best_img=False,
                 save_freq=1000,
                 device='cuda'):
        super(ISPStyleEnv, self).__init__()

        self.is_train = is_train
        self.is_val = not self.is_train
        self.batch_size = batch_size
        self.cfg = cfg
        self.device = device
        self.input_img_specs = input_img_specs
        self.max_step = max_step
        self.obs_stack_ori = obs_stack_ori
        self.obs_stack_step = obs_stack_step
        self.obs_stack_stop = obs_stack_stop
        self.obs_history_action = obs_history_action
        self.obs_img_mean_rgb = obs_img_mean_rgb
        self.truncate_param = truncate_param
        self.truncate_retouch_mean = truncate_retouch_mean
        self.isp_inp_original = isp_inp_original
        self.reward_scale = reward_scale
        self.eval_use_best_img = eval_use_best_img

        self.dataset = LoadImagesAndLabelsRAWReplay_target(
            data_path,
            image_size,
            batch_size=batch_size,
            augment=False,  # augmentation
            hyp=None,  # hyperparameters
            rect=False,  # rectangular batches
            highres_eval=highres_eval
        ) if specified_dataset is None else specified_dataset
        self.isp_blocks = isp_blocks

        # state obs variable
        self.images = None
        self.original_images = None
        self.targets = None
        self.steps = 0
        self.params = []  
        self.obs_steps = self.steps  

        self.rewards = None
        self.terminated = False
        self.truncated = False
        self.images_pth_list = []
        self.down_sample = nn.AdaptiveAvgPool2d((obs_img_shape[1], obs_img_shape[2]))
        self.rewards_range = 100.0

        self.obs_target_specs = True  # combined to a single image
        self.loss_type = loss_type  # ['l1','l2', 'xxx']
        self.loss_type_style = loss_type_style  # gram adain
        self.loss_type_content = loss_type_content  # vgg l2 l1 None
        self.loss_type_cont_target = loss_type_cont_target  # last first
        self.loss_type_hist = loss_type_hist  # rgb yuv None
        self.loss_coeff_style = loss_coeff_style
        self.loss_coeff_hist = loss_coeff_hist
        self.joint_obs = False

        self.image_steps = []
        self.params_steps = []
        self.actions_steps = []
        self.psnr_steps = []
        self.ssim_steps = []
        self.score_steps = []
        self.reward_steps = []

        action_dim = np.sum([filter.num_filter_parameters for filter in self.isp_blocks.filters])
        self.action_space = spaces.Box(low=-1, high=1, shape=(action_dim,), dtype=np.float32)
        self.action_space_dim = action_dim

        obs_img_dim = 3 + 3 + 3 + int(self.obs_target_specs)*3 + int(self.obs_img_mean_rgb)*6
        obs_vec_dim = action_dim + int(obs_stack_stop) + int(obs_stack_step) + int(self.obs_history_action)*self.max_step*(action_dim+1)
        if self.obs_stack_ori:
            obs_img_dim += 3
        self.observation_space = spaces.Dict({
            'image': spaces.Box(low=0, high=1, shape=(obs_img_dim, obs_img_shape[1], obs_img_shape[2]), dtype=np.float32),
            'vector': spaces.Box(low=-1, high=1, shape=(obs_vec_dim,), dtype=np.float32),
        })

        vgg = VGG.vgg
        vgg.load_state_dict(torch.load('envs/checkpoints/vgg_normalised.pth'))
        vgg = nn.Sequential(*list(vgg.children())[:31])
        self.vgg_net = VGG.Net(vgg).to(self.device)

        self.num_timestep = 0
        self.train_reset_cnt = 0
        self.eval_fname2step = {}
        self.save_freq = save_freq  # actual_freq = save_freq / len(eval_dataset)
        if self.is_val:
            self.base_dir = os.path.join('experiments/__sb3_saved_eval', args.save_path)
            os.makedirs(self.base_dir, exist_ok=True)
            self.log_dir = os.path.join(self.base_dir, "logs")
            os.makedirs(self.log_dir, exist_ok=True)
            self.ckpt_dir = os.path.join(self.base_dir, "ckpt")
            os.makedirs(self.ckpt_dir, exist_ok=True)
            self.image_dir = os.path.join(self.base_dir, "images")
            os.makedirs(self.image_dir, exist_ok=True)
        self.base_dir = os.path.join('experiments/__sb3_saved_eval', args.save_path)
        self.debug_dir = os.path.join(self.base_dir, "debug")
        os.makedirs(self.debug_dir, exist_ok=True)

    def _get_obs(self):
        images = self.down_sample(self.images)
        specs = get_img_specs(images, add_rgb=self.obs_img_mean_rgb)[0]
        image = images[0]
        targets = self.down_sample(self.targets)
        specs_target = get_img_specs(targets, add_rgb=self.obs_img_mean_rgb)[0]
        target = targets[0]
        stop = torch.tensor([self.terminated and self.truncated], device=self.device)
        step = torch.tensor([self.obs_steps], device=self.device)

        params_list = []
        for idx in range(len(self.params)):
            a = self.isp_blocks.filters[idx].range_l
            b = self.isp_blocks.filters[idx].range_r
            params_list.append(2.0 * ((self.params[idx][0] - a) / (b - a)) - 1.0)
        params = torch.cat([para.view(-1) for para in params_list])
        params_history = []
        if self.obs_history_action:
            assert not self.joint_obs, "not supported joint obs + stack action history"
            para_zeros = torch.zeros(self.action_space_dim + 1, device=self.device)
            for idx_steps in range(len(self.params_steps)):
                params_per_step = self.params_steps[idx_steps]
                params_list = []
                for idx in range(len(params_per_step)):
                    a = self.isp_blocks.filters[idx].range_l
                    b = self.isp_blocks.filters[idx].range_r
                    params_list.append(2.0 * ((self.params[idx][0] - a) / (b - a)) - 1.0)
                params_this_step = torch.cat([para.view(-1) for para in params_list])
                params_this_step = torch.cat([params_this_step, torch.tensor([self.reward_steps[idx_steps]], device=params_this_step.device)], dim=0)
                params_history.append(params_this_step)
            params_history = params_history + [para_zeros for _ in range(self.max_step - len(self.params_steps))]

        if self.obs_stack_ori:
            ori_img = self.down_sample(self.original_images)[0]
            image = torch.cat([image, ori_img])

        vec_list = []
        vec_list = vec_list + [stop] if self.obs_stack_stop else vec_list
        vec_list = vec_list + [step] if self.obs_stack_step else vec_list
        vec_list = vec_list + [params]
        if self.obs_history_action:
            vec_list = vec_list + params_history
        img_list = [image, target, specs[:, None, None] + image[0:1, :, :] * 0]
        img_list = img_list + [specs_target[:, None, None] + target[0:1, :, :] * 0] if self.obs_target_specs else img_list
        obs = {
            'image': torch.cat(img_list).cpu().numpy(),
            'vector': torch.cat(vec_list).cpu().numpy()
        }
        return obs

    def reset(self, seed=42):
        if self.is_train:
            im_list, label_list, path_list, _ = self.dataset.get_next_batch(self.batch_size)
            is_last = False
            self.train_reset_cnt += 1
            if self.train_reset_cnt % 100 == 0:
                print("!!! [train_reset_cnt] ", self.train_reset_cnt)
        else:
            self.num_timestep += 1
            im_list, label_list, path_list, _, is_last = self.dataset.get_next_batch_det(self.batch_size)
        self.images = torch.from_numpy(np.stack(im_list, 0)).to(self.device)
        self.original_images = self.images.clone()
        self.targets = torch.from_numpy(np.stack(label_list, 0)).to(self.device)
        self.steps = 0
        self.params, _ = self.isp_blocks.get_random_filter_params(device=self.device)
        self.obs_steps = self.steps

        self.rewards = torch.zeros(1)
        self.terminated = False
        self.truncated = False
        self.images_pth_list = path_list

        self.image_steps = []
        self.params_steps = []
        self.actions_steps = []
        self.psnr_steps = []
        self.ssim_steps = []
        self.score_steps = []
        self.reward_steps = []

        info = {"is_last": is_last}
        return self._get_obs(), info

    def print_status(self):
        print("images" + " shape ->", self.images.shape)
        print(self.images)
        print("original_images" + " shape ->", self.original_images.shape)
        print(self.original_images)
        print("targets" + " shape ->", self.targets.shape)
        print(self.targets)
        print("steps" + " shape ->")
        print(self.steps)
        print("params[0]" + " shape ->", self.params[0].shape)
        print(self.params)
        print("rewards" + " shape ->", self.rewards.shape)
        print(self.rewards)
        print("terminated" + " shape ->")
        print(self.terminated)
        print("images_pth_list" + " shape ->")
        print(self.images_pth_list)
        print("obs shape ->", self._get_obs().shape)
        print(self._get_obs())
        from stable_baselines3.common.preprocessing import is_image_space

    def action_to_tensor_list(self, action, dimensions):
        assert sum(dimensions) == len(action)
        assert len(dimensions) == len(self.isp_blocks.filters)
        tensor_list = []
        start = 0
        for idx, dim in enumerate(dimensions):
            chunk = action[start:start + dim]
            # scale action from (-1,1) to (a, b)
            para = torch.tensor(chunk, dtype=torch.float32, device=self.device).unsqueeze(0)
            a = self.isp_blocks.filters[idx].range_l
            b = self.isp_blocks.filters[idx].range_r
            para = a + (b - a) * (para + 1) / 2
            tensor_list.append(para)
            start += dim
        return tensor_list

    def get_extreme(self, score_list):
        if self.loss_type in ["psnr"]:
            return max(score_list)
        else:
            return - min(score_list)

    def compute_style_score(self, input, output, style):
        if self.loss_type_cont_target == "first":
            cont_target = self.original_images
        elif self.loss_type_cont_target == "last":
            cont_target = input
        else:
            raise NotImplementedError("no loss_type_cont_target")

        score = None
        eval_score = 0
        if self.loss_type_style == 'gram':
            l_cont, l_style = self.vgg_net.loss_gram(output, cont_target, style)
            score = self.loss_coeff_style * l_style
            eval_score = 100 * l_style
        elif self.loss_type_style == 'adain':
            l_cont, l_style = self.vgg_net.loss_adain(output, cont_target, style)
            score = 0.1 * l_style
        else:
            raise NotImplementedError("no loss_type_style")

        if self.loss_type_hist == 'yuv':
            hist_y, hist_uv = histogram_matching_loss(output[0], style[0])
            score += self.loss_coeff_hist * (hist_y + 0.5 * hist_uv)
            eval_score += 100 * (hist_y + 0.5 * hist_uv)

        if self.loss_type_content != "None":
            if self.loss_type_content == "l2":
                score += 2 * compute_l2(output, cont_target)
            elif self.loss_type_content == "vgg":
                score += 0.5 * l_cont
            else:
                raise NotImplementedError("no loss_type_cont")

        return score, eval_score  # tensor of size (1)

    def step(self, action):
        images_old = self.images.clone()
        param_dims = [filter.num_filter_parameters for filter in self.isp_blocks.filters]
        param_list = self.action_to_tensor_list(action, param_dims)
        self.actions_steps.append(action.tolist())

        if self.isp_inp_original:
            self.images = self.isp_blocks.run(self.original_images, param_list)
        else:
            self.images = self.isp_blocks.run(self.images, param_list)
        self.steps = self.steps + 1
        self.params = param_list
        # self.params_steps.append(param_list)
        self.image_steps.append(self.images[0])
        self.obs_steps = self.steps

        if self.loss_type == 'l2':
            self.rewards = (compute_l2_batch(images_old, self.targets) -
                            compute_l2_batch(self.images, self.targets))
            score = compute_l2_batch(self.images, self.targets)[0].item()
        elif self.loss_type == 'l1':
            self.rewards = (compute_l1_batch(images_old, self.targets) -
                            compute_l1_batch(self.images, self.targets))
            score = compute_l1_batch(self.images, self.targets)[0].item()
        elif self.loss_type == 'psnr':
            self.rewards = (compute_psnr_batch(self.images, self.targets) -
                            compute_psnr_batch(images_old, self.targets))
            score = compute_psnr_batch(self.images, self.targets)[0].item()
        elif self.loss_type == 'style':
            images_before_old = self.image_steps[-3].unsqueeze(0) if len(self.image_steps) > 2 else self.original_images
            old_score, _ = self.compute_style_score(images_before_old, images_old, self.targets)
            cur_score, eval_score = self.compute_style_score(images_old, self.images, self.targets)
            self.rewards = (old_score - cur_score).unsqueeze(0)
            score = eval_score.item()
        else:
            raise NotImplementedError("not supported loss type")

        self.rewards = self.rewards * self.reward_scale
        self.terminated = (self.steps >= self.max_step)
        self.params_steps.append(param_list)
        self.reward_steps.append(self.rewards[0].clamp(min=-self.rewards_range, max=self.rewards_range).item())
        retouch_mean = torch.mean(self.images, dim=(1, 2, 3))[0]
        self.truncated = False
        if self.truncate_param:
            eps = 1e-5
            self.truncated = np.all(np.isclose(action, 1, atol=eps) | np.isclose(action, -1, atol=eps))
            self.truncated = bool(self.truncated)
            if self.truncated and self.is_train:
                print(f"| [train] truncate-para at step {self.num_timestep}, ", end="")
            if self.truncated and self.is_val:
                print(f"| [val] truncate-para at step {self.num_timestep}, ", end="")
        if self.truncate_retouch_mean and (retouch_mean < 0.01 or retouch_mean > 0.9):
            self.truncated = True
            if self.truncated and self.is_train:
                print(f"| [train] truncate-mean at step {self.num_timestep}, ", end="")
            if self.truncated and self.is_val:
                print(f"| [val] truncate-mean at step {self.num_timestep}, ", end="")

        if self.is_train:
            reward = self.rewards[0].clamp(min=-self.rewards_range, max=self.rewards_range).item()
            terminated = self.terminated
            truncated = self.truncated
            info = {"score": score, "psnr": score}
        else:
            psnr = PSNR()
            psnr_loss = psnr(self.images, self.targets).clamp(max=100000.0).item()
            ssim_loss = structural_similarity_index_measure(self.images, self.targets).item()
            self.psnr_steps.append(psnr_loss)
            self.ssim_steps.append(ssim_loss)
            self.score_steps.append(score)
            if self.terminated:
                _, fullflname = os.path.split(self.images_pth_list[0])
                fname, _ = os.path.splitext(fullflname)
                last_timestep = self.eval_fname2step.get(fname)
                if last_timestep is None:
                    if len(self.eval_fname2step) < 20:
                        self.eval_fname2step[fname] = self.num_timestep
                        last_timestep = self.num_timestep
                if env_debug or ((last_timestep is not None) and (last_timestep // self.save_freq != self.num_timestep // self.save_freq)):
                    score_str = "{:.2f}".format(score).replace(".", "-")
                    save_img(self.image_steps[self.score_steps.index(min(self.score_steps))], self.images_pth_list[0], self.image_dir, f"{self.num_timestep}_out_{score_str}")
                    save_img(self.original_images[0], self.images_pth_list[0], self.image_dir, f"{self.num_timestep}_in")
                    save_img(self.targets[0], self.images_pth_list[0], self.image_dir, f"{self.num_timestep}_target")
                    if len(self.isp_blocks.filters) > 1:
                        save_params_steps(self.params_steps, self.images_pth_list[0], self.image_dir, f"{self.num_timestep}_para")
                    else:
                        save_params(self.params_steps, self.images_pth_list[0], self.image_dir, f"{self.num_timestep}_para")
                    _, fullflname = os.path.split(self.images_pth_list[0])
                    fname, ext = os.path.splitext(fullflname)
                    if int(fname.split("-")[0].replace("ExpertC", "")) < 5000:
                        visualize_results(self.original_images[0], self.targets[0], self.image_steps, self.actions_steps,
                                          self.psnr_steps, self.score_steps, self.images_pth_list[0], self.image_dir, f"{self.num_timestep}_traj")
            if self.eval_use_best_img and self.is_val:
                if psnr_loss < max(self.psnr_steps):
                    best_img_idx = self.psnr_steps.index(max(self.psnr_steps))
                    self.images = self.image_steps[best_img_idx].unsqueeze(0)
                    self.params = self.params_steps[best_img_idx]
                    self.obs_steps = best_img_idx + 1

            reward = 100 + self.get_extreme(self.score_steps)
            terminated = self.terminated
            truncated = self.truncated
            info = {"score": score, "step": self.steps, "psnr": score, }
        return self._get_obs(), reward, terminated, truncated, info



def visualize_results(input_image, target_image, trajectory_images, parameters, psnr_values, score_values, img_path, save_path, prefix=None):
    print('save traj')
    num_steps = len(trajectory_images)
    num_cols = min(num_steps, 5)  # Maximum of 5 columns for trajectory images per row
    num_rows = (num_steps + num_cols - 1) // num_cols  # Calculate the number of rows needed

    # Create subplots with two rows and up to 5 columns per row
    fig, axs = plt.subplots(num_rows + 1, num_cols, figsize=(5 * num_cols, 5 * (num_rows + 1)))

    # Plot input and target images in the first row
    axs[0, 0].imshow(input_image.detach().cpu().permute(1, 2, 0))
    axs[0, 0].set_title('Input Image')
    axs[0, 1].imshow(target_image.detach().cpu().permute(1, 2, 0))
    axs[0, 1].set_title('Target Image')

    # Plot trajectory images with parameters and PSNR/SSIM values
    for i in range(num_steps):
        row = i // num_cols + 1  # Calculate the row index
        col = i % num_cols  # Calculate the column index

        axs[row, col].imshow(trajectory_images[i].detach().cpu().permute(1, 2, 0))
        axs[row, col].set_title(
            f'Trajectory Image {i + 1}\nParameters: {", ".join([f"{param:.2f}" for param in parameters[i]])}\nPSNR: {psnr_values[i]:.3f}, SCORE: {score_values[i]:.3f}')

        # Plot PSNR and SSIM values for each trajectory image
        axs[row, col].text(0, -20, f'PSNR: {psnr_values[i]:.3f}', fontsize=10, ha='left')
        axs[row, col].text(0, -10, f'SCORE: {score_values[i]:.3f}', fontsize=10, ha='left')

    # Remove empty subplots
    for i in range(num_steps, num_rows * num_cols):
        row = i // num_cols + 1  # Calculate the row index
        col = i % num_cols  # Calculate the column index
        axs[row, col].remove()

    # Remove x and y ticks for all subplots
    for ax in axs.flatten():
        ax.axis('off')

    # Adjust layout
    plt.tight_layout()

    # Save the visualization image
    _, fullflname = os.path.split(img_path)
    fname, ext = os.path.splitext(fullflname)
    os.makedirs(os.path.join(save_path, fname), exist_ok=True)
    path = os.path.join(save_path, fname, fname + ('' if prefix is None else f'_{prefix}.png'))
    plt.savefig(path)
    plt.close()