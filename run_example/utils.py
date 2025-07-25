# Create a custom wrapper
import d4rl
import gym
import h5py
import numpy as np


class CustomDatasetWrapper(gym.Wrapper):
    def __init__(self, env, dataset_path, d4rl_name):
        super().__init__(env)
        self.dataset_path = dataset_path
        self.d4rl_name = d4rl_name

    def get_dataset(self):
        with h5py.File(self.dataset_path, 'r') as f:
            dataset = {
                'observations': np.array(f['observations']),
                'actions': np.array(f['actions']),
                'rewards': np.array(f['rewards']),
                'terminals': np.array(f['terminals']),
                'timeouts': np.array(f.get('timeouts', np.zeros_like(f['terminals'])))
            }

        # Add next_observations if not present
        if 'next_observations' not in f:
            dataset['next_observations'] = np.concatenate([
                dataset['observations'][1:],
                dataset['observations'][-1:]
            ], axis=0)
        else:
            dataset['next_observations'] = np.array(f['next_observations'])

        return dataset

    def get_normalized_score(self, mean_reward):
        return d4rl.get_normalized_score(self.d4rl_name, mean_reward)
