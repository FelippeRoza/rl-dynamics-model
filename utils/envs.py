
import mbrl.env.cartpole_continuous as cartpole_env

#============ Safe control gym envs ==================
from functools import partial
from envs.gym_control.registration import make

#============ Multi agent envs =======================
from envs.multi_agent.environment import MultiAgentEnv
from envs.multi_agent.descentralized_safe import Scenario


def get_env(env_name):

    if env_name == 'cartpole':
        env = cartpole_env.CartPoleEnv(render_mode="rgb_array")
    elif env_name == 'safe_cartpole':
        env_func = partial(make,
                        'cartpole', info_in_reset=True)
        env = env_func()
    elif env_name == 'multi':
        scenario = Scenario()
        world = scenario.make_world()
        env = MultiAgentEnv(world,
                        scenario.reset_world,
                        scenario.reward,
                        scenario.observation,
                        info_callback=None,
                        done_callback = scenario.done,
                        constraint_callback = scenario.constraints,
                        shared_viewer = True)
    else:
        raise ValueError('Unsupported env.')

    return env