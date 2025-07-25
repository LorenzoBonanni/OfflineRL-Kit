import numpy as np
import torch

from typing import Optional, Union, Tuple, Dict


class ReplayBuffer:
    def __init__(
        self,
        buffer_size: int,
        obs_shape: Tuple,
        obs_dtype: torch.dtype,
        action_dim: int,
        action_dtype: torch.dtype,
        device: str = "cpu"
    ) -> None:
        self._max_size = buffer_size
        self.obs_shape = obs_shape
        self.obs_dtype = obs_dtype
        self.action_dim = action_dim
        self.action_dtype = action_dtype

        self._ptr = 0
        self._size = 0
        self.device = torch.device(device)

        self.observations = torch.zeros(*((self._max_size,) + self.obs_shape), dtype=obs_dtype, device=self.device)
        self.next_observations = torch.zeros(*((self._max_size,) + self.obs_shape), dtype=obs_dtype, device=self.device)
        self.actions = torch.zeros((self._max_size, self.action_dim), dtype=action_dtype, device=self.device)
        self.rewards = torch.zeros((self._max_size, 1), dtype=torch.float32, device=self.device)
        self.terminals = torch.zeros((self._max_size, 1), dtype=torch.float32, device=self.device)


    def add(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        terminal: torch.Tensor
    ) -> None:
        # Copy to avoid modification by reference
        self.observations[self._ptr] = obs.clone()
        self.next_observations[self._ptr] = next_obs.clone()
        self.actions[self._ptr] = action.clone()
        self.rewards[self._ptr] = reward.clone()
        self.terminals[self._ptr] = terminal.clone()

        self._ptr = (self._ptr + 1) % self._max_size
        self._size = min(self._size + 1, self._max_size)
    
    def add_batch(
        self,
        obss: torch.Tensor,
        next_obss: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        terminals: torch.Tensor
    ) -> None:
        batch_size = len(obss)
        indexes = np.arange(self._ptr, self._ptr + batch_size) % self._max_size

        self.observations[indexes] = obss.clone()
        self.next_observations[indexes] = next_obss.clone()
        self.actions[indexes] = actions.clone()
        self.rewards[indexes] = rewards.clone()
        self.terminals[indexes] = terminals.clone()

        self._ptr = (self._ptr + batch_size) % self._max_size
        self._size = min(self._size + batch_size, self._max_size)
    
    def load_dataset(self, dataset: Dict[str, np.ndarray]) -> None:
        observations = torch.tensor(dataset["observations"], dtype=self.obs_dtype, device=self.device)
        next_observations = torch.tensor(dataset["next_observations"], dtype=self.obs_dtype, device=self.device)
        actions = torch.tensor(dataset["actions"], dtype=self.action_dtype, device=self.device)
        rewards = torch.tensor(dataset["rewards"], dtype=torch.float32, device=self.device).reshape(-1, 1)
        terminals = torch.tensor(dataset["terminals"], dtype=torch.float32, device=self.device).reshape(-1, 1)

        self.observations = observations
        self.next_observations = next_observations
        self.actions = actions
        self.rewards = rewards
        self.terminals = terminals

        self._ptr = len(observations)
        self._size = len(observations)
     
    def normalize_obs(self, eps: float = 1e-3) -> Tuple[np.ndarray, np.ndarray]:
        mean = self.observations.mean(0, keepdims=True)
        std = self.observations.std(0, keepdims=True) + eps
        self.observations = (self.observations - mean) / std
        self.next_observations = (self.next_observations - mean) / std
        obs_mean, obs_std = mean, std
        return obs_mean, obs_std

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:

        batch_indexes = np.random.randint(0, self._size, size=batch_size)
        
        return {
            "observations": torch.tensor(self.observations[batch_indexes]),
            "actions": torch.tensor(self.actions[batch_indexes]),
            "next_observations": torch.tensor(self.next_observations[batch_indexes]),
            "terminals": torch.tensor(self.terminals[batch_indexes]),
            "rewards": torch.tensor(self.rewards[batch_indexes])
        }
    
    def sample_all(self) -> Dict[str, np.ndarray]:
        return {
            "observations": self.observations[:self._size].clone(),
            "actions": self.actions[:self._size].clone(),
            "next_observations": self.next_observations[:self._size].clone(),
            "terminals": self.terminals[:self._size].clone(),
            "rewards": self.rewards[:self._size].clone()
        }