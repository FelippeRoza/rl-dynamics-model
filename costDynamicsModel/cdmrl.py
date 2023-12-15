import omegaconf
import numpy as np
import os
import torch 
import json

from .utils.model_config import get_base_config
from .utils.rollout import cost_rollout_agent_trajectories

import mbrl.env.reward_fns as reward_fns
import mbrl.env.termination_fns as termination_fns
import mbrl.models as models
import mbrl.planning as planning
import mbrl.util.common as common_util


class CDM():
    # cost dynamics model
    
    def __init__(self, env, buffer_size = 1e6, N=5, device = 'cpu', linearized=True, stack_length = 1) -> None:
        cfg_dict = get_base_config()

        self.env = env
        self.obs_shape = env.observation_space.shape
        self.action_shape = env.action_space.shape
        self.obs_action_shape = (env.observation_space.shape[0]*stack_length + env.action_space.shape[0]*stack_length, )
        self.cost_shape = env.constraint_space.shape
        self.linearized = linearized
        self.stack_length = stack_length

        cfg_dict["dynamics_model"]["device"] = device
        cfg_dict["dynamics_model"]["ensemble_size"] = N

        if linearized:
            cfg_dict['dynamics_model']['out_size'] = self.cost_shape[0] * self.action_shape[0]
        else:
            cfg_dict['dynamics_model']['_target_'] = "mbrl.models.GaussianMLP"
            cfg_dict['dynamics_model']['deterministic'] = True
            

        cfg_dict["overrides"]["trial_length"] = env.spec.max_episode_steps  # ep length
        cfg_dict["overrides"]["num_steps"] = int(buffer_size)

        self.cfg_dict = cfg_dict
        self.cfg = omegaconf.OmegaConf.create(self.cfg_dict)
    
        # Create a 1-D dynamics model for this environment
        self.dynamics_model = common_util.create_one_dim_tr_model(self.cfg, self.cost_shape, self.obs_action_shape)
        if linearized:
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

        def train_callback(_model, _total_calls, _epoch, tr_loss, val_score, _best_val):
            print(f'epoch: {_epoch} training loss:{tr_loss} validation score: {val_score.mean().item()}')
 
 
        model_trainer.train(dataset_train, dataset_val=dataset_val, num_epochs=n_epochs, patience=50, 
                            silent=False, callback=train_callback)

    def save(self, save_dir):
        # saves the dynamics model in the provided directory
        os.makedirs(save_dir, exist_ok = True)
        self.dynamics_model.save(save_dir)
        with open(os.path.join(save_dir, 'config.json'), 'w') as f:
            json.dump(self.cfg_dict, f, indent=2)


    def load(self, load_dir):
        self.dynamics_model.load(load_dir)

    def add_to_buffer(self, cost, obs, action, next_cost, reward, terminated, truncated):
        # manually add to buffer
        self.replay_buffer.add(cost, np.concatenate([obs, action]), next_cost, reward, terminated, truncated)
    
    def get_empty_stack(self):
        return np.zeros((self.replay_buffer.action.shape[1],))
    
    def stack_frame(self, frame_stack, new_frame):
        # roll frame_stack and add new frame to the end
        assert new_frame.shape == (self.obs_shape[0] + self.action_shape[0], ), \
                f"new frame shape should be {(self.obs_shape[0] + self.action_shape[0], )}"
        frame_stack = np.roll(frame_stack, -new_frame.shape[0])
        frame_stack[-new_frame.shape[0]:] = new_frame
        return frame_stack

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
    def forward_mean_std(self, cost, input_stack):
        # returns the mean and std of the ensemble predictions

        if type(cost) is list:
            cost = np.array(cost)
        model_in = self.dynamics_model._get_model_input(cost, input_stack)
        model_in = model_in.float().unsqueeze(0)
        pred_mean, _ = self.dynamics_model.model._default_forward(model_in, use_propagation=False, return_g=True)
        pred_var, pred_mean = torch.var_mean(pred_mean, axis=0)
        pred_std = np.sqrt(pred_var.squeeze(0).detach().cpu().numpy())
        
        pred_mean = pred_mean.squeeze(0).detach().cpu().numpy()
        
        return pred_mean, pred_std 
    
    @torch.no_grad
    def predict(self, cost, input_stack, action):
        # predicts cost for next time step based on current cost, obs and action
        
        mean, std  = self.forward_mean_std(cost, input_stack)
        
        if self.linearized:
            return cost + mean @ action

        else:
            return cost + mean
