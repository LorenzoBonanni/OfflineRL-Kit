import gym
import h5py
import numpy as np
import wandb
import d4rl
import torch

from offlinerlkit.modules.actor_module import ActorProb

class CustomDatasetWrapper(gym.Wrapper):
    def __init__(self, env, dataset_path, d4rl_name):
        super().__init__(env)
        self.dataset_path = dataset_path
        self.d4rl_name = d4rl_name

    def get_dataset(self):
        with h5py.File(self.dataset_path, 'r') as f:
            print(f"Baseline Performance: {f['metadata'].attrs['eval_avg_return']:.3f}±{f['metadata'].attrs.get('eval_std_return', 0.0):.3f} on {f['metadata'].attrs['eval_episodes']} episodes")
            print("Deterministic Policy: ", f['metadata'].attrs.get('deterministic_policy', None))
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
    
    def get_normalized_score(self, score):
        # ref_min_score = d4rl.infos.REF_MIN_SCORE[self.d4rl_name]
        # ref_max_score = d4rl.infos.REF_MAX_SCORE[self.d4rl_name]
        # return (score - ref_min_score) / (ref_max_score - ref_min_score)
        return score

def run_evaluation(actor, env_id, device, num_eval_episodes=10, seed=42):
    actor.eval()
    """Run evaluation episodes and return average return"""
    eval_env = gym.make(env_id)
    eval_env = gym.wrappers.RecordEpisodeStatistics(eval_env)
    eval_env.action_space.seed(seed)

    
    episode_returns = []
    
    print(f"Running {num_eval_episodes} evaluation episodes...")
    
    for episode in range(num_eval_episodes):
        obs = eval_env.reset(seed=seed + episode)
        episode_return = 0.0
        done = False
        s = 1
        while not done:
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                actions = actor(obs_tensor)
                if isinstance(actor, ActorProb):
                    action, _ = actions.mode()
                    action = action.cpu().numpy()[0]
                else:
                    action = actions.cpu().numpy()[0]
            
            obs, reward, terminated, info  = eval_env.step(action)

            episode_return += reward
            done = terminated
            s+=1
        
        episode_returns.append(episode_return)
        print(f"Episode {episode + 1}: Return = {episode_return:.2f}")
        print("Steps taken:", s)
    
    avg_return = np.mean(episode_returns)
    std_return = np.std(episode_returns)
    
    print(f"\nEvaluation Results:")
    print(f"Average Return: {avg_return:.2f} ± {std_return:.2f}")
    wandb.log({
        "final/average_return": avg_return,
        "final/std_return": std_return
    })
    print(f"Min Return: {np.min(episode_returns):.2f}")
    print(f"Max Return: {np.max(episode_returns):.2f}")
    print("-" * 50)

    # norm_return = eval_env.get_normalized_score(avg_return) * 100.0
    # print(f"Normalized Average Return: {norm_return:.2f}")
    eval_env.close()
    return avg_return, episode_returns