import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import gymnasium as gym
import numpy as np

from stable_baselines3.common import type_aliases
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor, is_vecenv_wrapped, sync_envs_normalization

from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import EvalCallback
import numpy as np
import os
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback
from envs.isp_env import ISPEnv


class SaveActionDistributionCallback(BaseCallback):
    def __init__(self,
                 save_freq: int = 1000,
                 action_dim: int = 1,
                 save_dir: str = ""):
        super().__init__()
        self.save_freq = save_freq
        self.action_dim = action_dim
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, "action"), exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, "psnr"), exist_ok=True)
        self.steps = []
        self.distributions = [[] for _ in range(action_dim)]
        self.n_env = 0

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0:
            actions = self.model.replay_buffer.actions
            psnrs = self.model.replay_buffer.psnrs
            for dim in range(self.action_dim):
                self.distributions[dim] = actions[:, self.n_env, dim]
            self.steps.append(self.num_timesteps)
            self.save_distributions(self.num_timesteps)
            self.save_psnrs(self.num_timesteps, psnrs)
            print("saving buffer action distribution")
        return True

    def save_distributions(self, step):
        for dim, distribution in enumerate(self.distributions):
            plt.hist(distribution, bins=200)
            plt.title(f'Action Dimension {dim} Distribution at step {step}')
            plt.xlabel('Action Value')
            plt.ylabel('Frequency')
            plt.savefig(os.path.join(self.save_dir, "action", f'step_{step}_action_dim_{dim}_distribution.png'))
            plt.close()

    def save_psnrs(self, step, psnrs):
        mean_psnr = np.mean(psnrs)
        psnr_str = "{:.2f}".format(mean_psnr).replace(".", "-")
        plt.hist(psnrs, bins=200)
        plt.title(f'buffer PSNR (mean={mean_psnr}) Distribution at step {step}')
        plt.xlabel('PSNR Value')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(self.save_dir, "psnr", f'step_{step}_psnr_dist_mean_{psnr_str}.png'))
        plt.close()


class CustomEvalCallback(EvalCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _on_step(self) -> bool:
        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    ) from e

            # Reset success rate buffer
            self._is_success_buffer = []

            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )

            if self.log_path is not None:
                assert isinstance(episode_rewards, list)
                assert isinstance(episode_lengths, list)
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = float(mean_reward)

            if self.verbose >= 1:
                print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose >= 1:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose >= 1:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                self.best_mean_reward = float(mean_reward)
                # Trigger callback on new best model, if needed
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            self.logger.record("eval/best_mean_reward", self.best_mean_reward)

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training


def evaluate_policy(
    model: "type_aliases.PolicyPredictor",
    model_fine: "type_aliases.PolicyPredictor",
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
    reward_threshold: Optional[float] = None,
    return_episode_rewards: bool = False,
    warn: bool = True,
    psnr_thres: float = 40.0,
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
    is_monitor_wrapped = False
    # Avoid circular import
    from stable_baselines3.common.monitor import Monitor

    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])  # type: ignore[list-item, return-value]

    is_monitor_wrapped = is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]

    n_envs = env.num_envs  # todo! by default, num_env=1
    episode_rewards = []
    episode_lengths = []

    episode_counts = np.zeros(n_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")

    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    observations = env.reset()
    states = None
    episode_starts = np.ones((env.num_envs,), dtype=bool)
    while (episode_counts < episode_count_targets).any():
        actions, states = model.predict(
            observations,  # type: ignore[arg-type]
            state=states,
            episode_start=episode_starts,
            deterministic=deterministic,
        )
        new_observations, rewards, dones, infos = env.step(actions)
        indices = rewards > current_rewards
        current_rewards[indices] = rewards[indices]
        current_lengths += 1
        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:
                # unpack values so that the callback can access the local variables
                reward = rewards[i]
                done = dones[i]
                info = infos[i]
                episode_starts[i] = done

                if callback is not None:
                    callback(locals(), globals())

                assert isinstance(env, ISPEnv)
                # todo transit condition must be PSNR rating

                if dones[i]:
                    # todo [start] use fine policy for a new round -- should we change the obs space
                    # actually nothing will change for the environment, its just changing model in different stage
                    dones[i] = False
                    while not dones[i]:
                        observations = new_observations

                    # todo [end] use fine policy
                    # episode_rewards.append(current_rewards[i])
                    episode_rewards.append(current_rewards[i])  # todo : here is for custom eval where episode reward is PSNR avg
                    episode_lengths.append(current_lengths[i])
                    episode_counts[i] += 1
                    current_rewards[i] = 0
                    current_lengths[i] = 0

        observations = new_observations
        if render:
            env.render()

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    if return_episode_rewards:
        return episode_rewards, episode_lengths
    return mean_reward, std_reward