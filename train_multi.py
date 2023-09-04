import numpy as np
import torch
import omegaconf

#===================================================
from envs.multi_agent.environment import MultiAgentEnv
from envs.multi_agent.descentralized_safe import Scenario


import mbrl.env.cartpole_continuous as cartpole_env
import mbrl.env.reward_fns as reward_fns
import mbrl.env.termination_fns as termination_fns
import mbrl.models as models
import mbrl.planning as planning
import mbrl.util.common as common_util
import mbrl.util as util
import mbrl.models.gaussian_mlp

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


scenario = Scenario()
world = scenario.make_world()
# Environment Setup
env = MultiAgentEnv(world,
                    scenario.reset_world,
                    scenario.reward,
                    scenario.observation,
                    info_callback=None,
                    done_callback = scenario.done,
                    constraint_callback = scenario.constraints,
                    shared_viewer = True)


seed = 0
env.reset(seed)
rng = np.random.default_rng(seed=0)
generator = torch.Generator(device=device)
generator.manual_seed(seed)
obs_shape = env.observation_space.shape
act_shape = env.action_space.shape

# This functions allows the model to evaluate the true rewards given an observation 
reward_fn = reward_fns.cartpole
# This function allows the model to know if an observation should make the episode end
term_fn = termination_fns.cartpole



trial_length = 1000
num_trials = 500
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

# Create a gym-like environment to encapsulate the model
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

dynamics_model.update_normalizer(replay_buffer.get_all())  # update normalizer stats


# Create a trainer for the model
model_trainer = models.ModelTrainer(dynamics_model, optim_lr=1e-3, weight_decay=5e-5)

dataset_train, dataset_val = common_util.get_basic_buffer_iterators(
    replay_buffer,
    batch_size=cfg.overrides.model_batch_size,
    val_ratio=cfg.overrides.validation_ratio,
    ensemble_size=ensemble_size,
    shuffle_each_epoch=True,
    bootstrap_permutes=False,  # build bootstrap dataset using sampling with replacement
)

train_losses = []
val_scores = []

def train_callback(_model, _total_calls, _epoch, tr_loss, val_score, _best_val):
    train_losses.append(tr_loss)
    val_scores.append(val_score.mean().item())   # this returns val score per ensemble model

model_trainer.train(
    dataset_train, 
    dataset_val=dataset_val, 
    num_epochs=50, 
    patience=50, 
    callback=train_callback,
    silent=True)


x = np.hstack((replay_buffer.obs, replay_buffer.action))
y = np.hstack((replay_buffer.next_obs, replay_buffer.action))

x_tensor = torch.from_numpy(x).float().to(device)
x_tensor = dynamics_model.input_normalizer.normalize(x_tensor)
y_tensor = torch.from_numpy(y).float().to(device)
y_tensor = dynamics_model.input_normalizer.normalize(y_tensor)

def get_mean_var(x, denormalize = False):
    x_tensor = torch.from_numpy(x).float().to(device)
    x_tensor = dynamics_model.input_normalizer.normalize(x_tensor)
    with torch.no_grad():
        mean, logvar = dynamics_model(x_tensor, use_propagation = False)
        mean = torch.mean(mean, axis = 0)
        var = torch.mean(torch.exp(logvar), axis = 0)
        if denormalize:
            mean = dynamics_model.input_normalizer.denormalize(torch.hstack((mean, torch.Tensor([[0]]).cuda())))
            var = dynamics_model.input_normalizer.denormalize(torch.hstack((var, torch.Tensor([[0]]).cuda())))
    return mean, var


save_dir = 'models/multi/'
import os
os.makedirs(save_dir, exist_ok=True)
dynamics_model.save(save_dir)
print(1)