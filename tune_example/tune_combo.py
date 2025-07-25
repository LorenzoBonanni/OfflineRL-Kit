import argparse
import os
import sys
import random

import gym

import numpy as np
import torch
import ray
from ray import tune
import d4rl



from offlinerlkit.nets import MLP
from offlinerlkit.modules import ActorProb, Critic, TanhDiagGaussian, EnsembleDynamicsModel
from offlinerlkit.dynamics import EnsembleDynamics
from offlinerlkit.policy.model_based.combo import COMBOPolicy
from offlinerlkit.utils.scaler import StandardScaler
from offlinerlkit.utils.termination_fns import get_termination_fn
from offlinerlkit.utils.load_dataset import qlearning_dataset
from offlinerlkit.buffer import ReplayBuffer
from offlinerlkit.utils.logger import Logger, make_log_dirs
from offlinerlkit.policy_trainer import MBPolicyTrainer
from offlinerlkit.policy import MOPOPolicy
from utils import CustomDatasetWrapper


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo-name", type=str, default="combo")
    parser.add_argument("--task", type=str, default="hopper-medium-v2")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--hidden-dims", type=int, nargs='*', default=[256, 256, 256])
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--auto-alpha", default=True)
    parser.add_argument("--target-entropy", type=int, default=None)
    parser.add_argument("--alpha-lr", type=float, default=1e-4)
    parser.add_argument("--fraction", type=float, default=1)

    parser.add_argument("--cql-weight", type=float, default=5.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-q-backup", type=bool, default=False)
    parser.add_argument("--deterministic-backup", type=bool, default=True)
    parser.add_argument("--with-lagrange", type=bool, default=False)
    parser.add_argument("--lagrange-threshold", type=float, default=10.0)
    parser.add_argument("--cql-alpha-lr", type=float, default=3e-4)
    parser.add_argument("--num-repeat-actions", type=int, default=10)
    parser.add_argument("--uniform-rollout", type=bool, default=False)
    parser.add_argument("--rho-s", type=str, default="mix", choices=["model", "mix"])

    parser.add_argument("--dynamics-lr", type=float, default=1e-3)
    parser.add_argument("--dynamics-hidden-dims", type=int, nargs='*', default=[200, 200, 200, 200])
    parser.add_argument("--dynamics-weight-decay", type=float, nargs='*', default=[2.5e-5, 5e-5, 7.5e-5, 7.5e-5, 1e-4])
    parser.add_argument("--n-ensemble", type=int, default=7)
    parser.add_argument("--n-elites", type=int, default=5)
    parser.add_argument("--rollout-freq", type=int, default=1000)
    parser.add_argument("--rollout-batch-size", type=int, default=50000)
    parser.add_argument("--rollout-length", type=int, default=5)
    parser.add_argument("--model-retain-epochs", type=int, default=5)
    parser.add_argument("--real-ratio", type=float, default=0.5)
    parser.add_argument("--load-dynamics-path", type=str, default=None)

    parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--step-per-epoch", type=int, default=1000)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    return parser.parse_args()



def run_exp(args_for_exp):
    # set config
    global args
    args_for_exp = vars(args)
    for k, v in config.items():
        args_for_exp[k] = v
    args_for_exp = argparse.Namespace(**args_for_exp)

    # create env and dataset
    if 'custom' in args_for_exp.task:
        name_dict = {
            'halfcheetah_custom': ('HalfCheetah-v2', 'halfcheetah-medium-v2'),
        }

        env_name, d4rl_name = name_dict[args_for_exp.task]
        env = gym.make(env_name)
        env = CustomDatasetWrapper(
            env,
            f'{os.environ.get("D4RL_DATASET_DIR")}/datasets/{args_for_exp.task}_dataset.hdf5',
            d4rl_name
        )
    else:
        env = gym.make(args_for_exp.task)


    # seed
    random.seed(args_for_exp.seed)
    np.random.seed(args_for_exp.seed)
    torch.manual_seed(args_for_exp.seed)
    torch.cuda.manual_seed_all(args_for_exp.seed)
    torch.backends.cudnn.deterministic = True
    env.seed(args_for_exp.seed)

    # create policy model
    actor_backbone = MLP(input_dim=np.prod(args_for_exp.obs_shape), hidden_dims=args_for_exp.hidden_dims)
    critic1_backbone = MLP(input_dim=np.prod(args_for_exp.obs_shape) + args_for_exp.action_dim, hidden_dims=args_for_exp.hidden_dims)
    critic2_backbone = MLP(input_dim=np.prod(args_for_exp.obs_shape) + args_for_exp.action_dim, hidden_dims=args_for_exp.hidden_dims)
    dist = TanhDiagGaussian(
        latent_dim=getattr(actor_backbone, "output_dim"),
        output_dim=args_for_exp.action_dim,
        unbounded=True,
        conditioned_sigma=True,
        max_mu=args_for_exp.max_action
    )
    actor = ActorProb(actor_backbone, dist, args_for_exp.device)
    critic1 = Critic(critic1_backbone, args_for_exp.device)
    critic2 = Critic(critic2_backbone, args_for_exp.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args_for_exp.actor_lr)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args_for_exp.critic_lr)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args_for_exp.critic_lr)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(actor_optim, args_for_exp.epoch)


    if args_for_exp.auto_alpha:
        target_entropy = args_for_exp.target_entropy if args_for_exp.target_entropy \
            else -np.prod(env.action_space.shape)

        args_for_exp.target_entropy = target_entropy

        log_alpha = torch.zeros(1, requires_grad=True, device=args_for_exp.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args_for_exp.alpha_lr)
        alpha = (target_entropy, log_alpha, alpha_optim)
    else:
        alpha = args_for_exp.alpha

    # create dynamics
    load_dynamics_model = True if args_for_exp.load_dynamics_path else False
    dynamics_model = EnsembleDynamicsModel(
        obs_dim=np.prod(args_for_exp.obs_shape),
        action_dim=args_for_exp.action_dim,
        hidden_dims=args_for_exp.dynamics_hidden_dims,
        num_ensemble=args_for_exp.n_ensemble,
        num_elites=args_for_exp.n_elites,
        weight_decays=args_for_exp.dynamics_weight_decay,
        device=args_for_exp.device
    )
    dynamics_optim = torch.optim.Adam(
        dynamics_model.parameters(),
        lr=args_for_exp.dynamics_lr
    )
    scaler = StandardScaler()
    termination_fn = get_termination_fn(task=args_for_exp.task)
    dynamics = EnsembleDynamics(
        dynamics_model,
        dynamics_optim,
        scaler,
        termination_fn,
        penalty_coef=args_for_exp.penalty_coef
    )

    if args_for_exp.load_dynamics_path:
        dynamics.load(args_for_exp.load_dynamics_path)

    # create policy
    policy = COMBOPolicy(
        dynamics,
        actor,
        critic1,
        critic2,
        actor_optim,
        critic1_optim,
        critic2_optim,
        action_space=env.action_space,
        tau=args.tau,
        gamma=args.gamma,
        alpha=alpha,
        cql_weight=args.cql_weight,
        temperature=args.temperature,
        max_q_backup=args.max_q_backup,
        deterministic_backup=args.deterministic_backup,
        with_lagrange=args.with_lagrange,
        lagrange_threshold=args.lagrange_threshold,
        cql_alpha_lr=args.cql_alpha_lr,
        num_repeart_actions=args.num_repeat_actions,
        uniform_rollout=args.uniform_rollout,
        rho_s=args.rho_s
    )

    # create buffer
    real_buffer = ReplayBuffer(
        buffer_size=len(dataset["observations"]),
        obs_shape=args_for_exp.obs_shape,
        obs_dtype=torch.float32,
        action_dim=args_for_exp.action_dim,
        action_dtype=torch.float32,
        device=args_for_exp.device
    )
    real_buffer.load_dataset(dataset)
    fake_buffer = ReplayBuffer(
        buffer_size=args_for_exp.rollout_batch_size*args_for_exp.rollout_length*args_for_exp.model_retain_epochs,
        obs_shape=args_for_exp.obs_shape,
        obs_dtype=torch.float32,
        action_dim=args_for_exp.action_dim,
        action_dtype=torch.float32,
        device=args_for_exp.device
    )

    # log
    record_params = list(config.keys())
    if "seed" in record_params:
        record_params.remove("seed")
    log_dirs = make_log_dirs(
        args_for_exp.task,
        args_for_exp.algo_name,
        args_for_exp.seed,
        vars(args_for_exp),
        record_params=record_params
    )
    # key: output file name, value: output handler type
    output_config = {
        "consoleout_backup": "stdout",
        "policy_training_progress": "csv",
        "dynamics_training_progress": "csv",
        "tb": "tensorboard"
    }
    logger = Logger(log_dirs, output_config)
    logger.log_hyperparameters(vars(args_for_exp))

    # create policy trainer
    policy_trainer = MBPolicyTrainer(
        policy=policy,
        eval_env=env,
        real_buffer=real_buffer,
        fake_buffer=fake_buffer,
        logger=logger,
        rollout_setting=(args_for_exp.rollout_freq, args_for_exp.rollout_batch_size, args_for_exp.rollout_length),
        epoch=args_for_exp.epoch,
        step_per_epoch=args_for_exp.step_per_epoch,
        batch_size=args_for_exp.batch_size,
        real_ratio=args_for_exp.real_ratio,
        eval_episodes=args_for_exp.eval_episodes
    )

    # train
    if not load_dynamics_model:
        dynamics.train(real_buffer.sample_all(), logger)
    
    result = policy_trainer.train()
    tune.report(**result)


if __name__ == "__main__":
    ray.init()
    # load default args
    args = get_args()

    config = {}
    seeds = list(range(3))
    config["rollout_length"] = tune.grid_search([1, 5])
    config["penalty_coef"] = tune.grid_search([0.5, 1.0, 5.0])
    config["seed"] = tune.grid_search(seeds)

    analysis = tune.run(
        run_exp,
        name="tune_mopo",
        config=config,
        resources_per_trial={
            "gpu": 0.5
        }
    )