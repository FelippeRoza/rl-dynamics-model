'''Register environments.'''

from .registration import register

register(idx='cartpole',
         entry_point='envs.gym_control.cartpole:CartPole',
         config_entry_point='envs.gym_control:cartpole.yaml')

# register(idx='quadrotor',
#          entry_point='envs.gym_pybullet_drones.quadrotor:Quadrotor',
#          config_entry_point='envs.gym_pybullet_drones:quadrotor.yaml')
