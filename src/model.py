from stable_baselines3 import DQN
from stable_baselines3.common.buffers import ReplayBufferSamples
import torch as th
import torch.nn.functional as F
import numpy as np

def make_dqn_model(env, dataset_size, device="cpu", verbose=0, double_dqn: bool = False):
    class DoubleDQN(DQN):
        def __init__(self, *args, double_dqn: bool = True, **kwargs):
            super().__init__(*args, **kwargs)
            self.double_dqn = double_dqn

        def train(self, gradient_steps: int, batch_size: int = 100) -> None:
            self.policy.set_training_mode(True)
            self._update_learning_rate(self.policy.optimizer)

            losses = []
            for _ in range(gradient_steps):
                replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

                discounts = replay_data.discounts if hasattr(replay_data, 'discounts') and replay_data.discounts is not None else self.gamma * th.ones_like(replay_data.rewards)

                with th.no_grad():
                    next_q_values = self.q_net_target(replay_data.next_observations)
                    if self.double_dqn:
                        next_actions = self.q_net(replay_data.next_observations).argmax(dim=1, keepdim=True)
                        next_q_values = next_q_values.gather(1, next_actions).squeeze(1)
                    else:
                        next_q_values, _ = next_q_values.max(dim=1)
                    
                    next_q_values = next_q_values.reshape(-1, 1)
                    
                    target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

                current_q_values = self.q_net(replay_data.observations).gather(1, replay_data.actions.long())

                loss = F.smooth_l1_loss(current_q_values, target_q_values)
                losses.append(loss.item())

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += gradient_steps
            self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
            self.logger.record("train/loss", np.mean(losses))

    policy_kwargs = dict(
        net_arch=[256, 256, 128],
        activation_fn=th.nn.ReLU,
    )

    if dataset_size > 100_000:
        hyperparams = {
            "learning_rate": 1e-3,
            "buffer_size": 100_000,
            "learning_starts": 1_000,
            "batch_size": 256,
            "train_freq": 8,
            "gradient_steps": 4,
            "exploration_initial_eps": 1.0,
            "exploration_fraction": 0.2,
            "exploration_final_eps": 0.05,
            "target_update_interval": 500,
            "policy_kwargs": policy_kwargs,
            "tau": 1.0,
        }
    else:
        hyperparams = {
            "learning_rate": 7e-4,
            "buffer_size": 100_000,
            "learning_starts": 2_000,
            "batch_size": 128,
            "train_freq": 4,
            "gradient_steps": 4,
            "exploration_initial_eps": 1.0,
            "exploration_fraction": 0.3,
            "exploration_final_eps": 0.05,
            "target_update_interval": 750,
            "policy_kwargs": policy_kwargs,
            "tau": 1.0,
        }

    return DoubleDQN(
        "MlpPolicy",
        env,
        verbose=verbose,
        device=device,
        double_dqn=double_dqn,
        **hyperparams,
    )