import numpy as np
import torch
import omegaconf

import mbrl.env.reward_fns as reward_fns
import mbrl.env.termination_fns as termination_fns
import mbrl.models as models
import mbrl.planning as planning
import mbrl.util.common as common_util
import mbrl.util as util
import mbrl.models.gaussian_mlp

from utils.envs import get_env

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

env_name = 'safe_cartpole' # 'multi' 'safe_cartpole' 'cartpole'
seed = 0

env = get_env(env_name)
env.reset(seed)
obs_shape = env.observation_space.shape
act_shape = env.action_space.shape

trial_length = 200
num_trials = 1
ensemble_size = 5
# Everything with "???" indicates an option with a missing value.
# Our utility functions will fill in these details using the 
# environment information
cfg_dict = {
    # dynamics model configuration
    "dynamics_model": {
        "_target_": "mbrl.models.GaussianMLP",
        "device": device,
        "num_layers": 3,
        "ensemble_size": ensemble_size,
        "hid_size": 200,
        "in_size": "???",
        "out_size": "???",
        "deterministic": False,
        "propagation_method": "fixed_model",
        # can also configure activation function for GaussianMLP
        "activation_fn_cfg": {
            "_target_": "torch.nn.LeakyReLU",
            "negative_slope": 0.01
        }
    },
    # options for training the dynamics model
    "algorithm": {
        "learned_rewards": False,
        "target_is_delta": True,
        "normalize": True,
    },
    # these are experiment specific options
    "overrides": {
        "trial_length": trial_length,
        "num_steps": num_trials * trial_length,
        "model_batch_size": 32,
        "validation_ratio": 0.05
    }
}
cfg = omegaconf.OmegaConf.create(cfg_dict)

# Create a 1-D dynamics model for this environment
dynamics_model = common_util.create_one_dim_tr_model(cfg, obs_shape, act_shape)

generator = torch.Generator(device=device)
generator.manual_seed(seed)
reward_fn = reward_fns.cartpole
term_fn = termination_fns.cartpole

model_env = models.ModelEnv(env, dynamics_model, term_fn, reward_fn, generator=generator)

replay_buffer = common_util.create_replay_buffer(cfg, obs_shape, act_shape)


common_util.rollout_agent_trajectories(
    env,
    trial_length * num_trials, # exploration steps
    planning.RandomAgent(env),
    {}, # keyword arguments to pass to agent.act()
    replay_buffer=replay_buffer,
    trial_length=trial_length
)

print("# samples stored", replay_buffer.num_stored)

dataset_train, dataset_val = common_util.get_basic_buffer_iterators(
    replay_buffer,
    batch_size=cfg.overrides.model_batch_size,
    val_ratio=cfg.overrides.validation_ratio,
    ensemble_size=ensemble_size,
    shuffle_each_epoch=True,
    bootstrap_permutes=False,  # build bootstrap dataset using sampling with replacement
)

dynamics_model.load(f'models/{env_name}/')

model_in, target = dynamics_model._process_batch(dataset_train[0])
pred_mean, _ = dynamics_model.forward(model_in, use_propagation=False)

print(1)