import omegaconf
import numpy as np
import os
import torch 

from .utils.model_config import get_base_config
from .utils.rollout import cost_rollout_agent_trajectories

import mbrl.env.reward_fns as reward_fns
import mbrl.env.termination_fns as termination_fns
import mbrl.models as models
import mbrl.planning as planning
import mbrl.util.common as common_util


class CDM():
    # cost dynamics model
    
    def __init__(self, env, buffer_size = 1e6, N=5, device = 'cpu') -> None:
        cfg_dict = get_base_config()

        self.env = env
        self.obs_shape = env.observation_space.shape
        self.action_shape = env.action_space.shape
        self.obs_action_shape = (env.observation_space.shape[0] + env.action_space.shape[0], )
        self.cost_shape = env.constraint_space.shape

        cfg_dict["dynamics_model"]["device"] = device
        cfg_dict["dynamics_model"]["ensemble_size"] = N
        cfg_dict['dynamics_model']['out_size'] = self.cost_shape[0] * self.action_shape[0]

        cfg_dict["overrides"]["trial_length"] = env.spec.max_episode_steps  # ep length
        cfg_dict["overrides"]["num_steps"] = buffer_size

        self.cfg_dict = cfg_dict
        self.cfg = omegaconf.OmegaConf.create(self.cfg_dict)
    
        # Create a 1-D dynamics model for this environment
        self.dynamics_model = common_util.create_one_dim_tr_model(self.cfg, self.cost_shape, self.obs_action_shape)
        self.dynamics_model.model.set_shapes(self.obs_shape, self.action_shape, self.cost_shape)
        
        # self.model_env = models.ModelEnv(env, self.dynamics_model, term_fn, reward_fn, generator=generator)
        self.replay_buffer = common_util.create_replay_buffer(self.cfg, self.cost_shape, self.obs_action_shape)
    

    def train(self, n_epochs=20):

        model_trainer = models.ModelTrainer(self.dynamics_model, optim_lr=1e-3, weight_decay=5e-5)

        dataset_train, dataset_val = common_util.get_basic_buffer_iterators(
            self.replay_buffer,
            batch_size = self.cfg.overrides.model_batch_size,
            val_ratio = self.cfg.overrides.validation_ratio,
            ensemble_size = self.cfg.dynamics_model.ensemble_size,
            shuffle_each_epoch=True,
            bootstrap_permutes=False,  # build bootstrap dataset using sampling with replacement
        )

        model_trainer.train(dataset_train, dataset_val=dataset_val, num_epochs=n_epochs, patience=50, 
                            silent=False)

    def save(self, save_dir):
        # saves the dynamics model in the provided directory
        os.makedirs(save_dir, exist_ok = True)
        self.dynamics_model.save(save_dir)

    def load(self, load_dir):
        self.dynamics_model.load(load_dir)

    def add_to_buffer(self, cost, obs, action, next_cost, reward, terminated, truncated):
        # manually add to buffer
        self.replay_buffer.add(cost, np.concatenate([obs, action]), next_cost, reward, terminated, truncated)
    

    def fill_buffer(self, steps=None):
        # fill buffer with randomly exploring the environment

        if steps is None:
            steps = self.cfg.overrides.num_steps

        cost_rollout_agent_trajectories(
            self.env,
            steps, # exploration steps
            planning.RandomAgent(self.env),
            {}, # keyword arguments to pass to agent.act()
            replay_buffer=self.replay_buffer,
            trial_length=self.cfg.overrides.trial_length
        )

        print("# samples stored", self.replay_buffer.num_stored)
    
    @torch.no_grad
    def predict(self, cost, obs, action):
        # predicts cost for next time step based on current cost, obs and action
        model_in = self.dynamics_model._get_model_input(cost, np.concatenate((obs, action)))
        pred_mean, _ = self.dynamics_model.model._default_forward(model_in, use_propagation=False, return_g=True)
        pred_mean = torch.mean(pred_mean, axis=0)* 3 @ action
        pred_mean = cost + pred_mean.squeeze(0).detach().cpu().numpy()
        
        return pred_mean