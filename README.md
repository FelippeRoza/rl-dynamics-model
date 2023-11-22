# Rl Cost Dynamics Model

Model based on the mbrl-lib to train cost dynamics model for safe RL environments.
Compatible with the continuousSafetyGym environments.


## Installation

```
git clone https://github.com/FelippeRoza/rl-dynamics-model.git
cd rl-dynamics-model
pip install -e .
```

install the continuousSafetyGym:

```
git clone https://github.com/FelippeRoza/continuous-safety-gym.git
cd continuous-safety-gym
pip install -e .
```


## Train model


```
from costDynamicsModel import CDM
import continuousSafetyGym
import gymnasium as gym

env = gym.make('ContSafetyBallReach-v1', render_mode = 'rgb_array')
cdm = CDM(env, buffer_size=1_000_000)

cdm.fill_buffer()   # explores environment with random actions until buffer is full
cdm.train()         # dynamics model training
cdm.save('models')  # creates folder 'models' and save the dynamics model
```

## Test model

```
from costDynamicsModel import CDM
import continuousSafetyGym
import gymnasium as gym
import numpy as np


env = gym.make('ContSafetyBallReach-v1', render_mode = 'rgb_array')
cdm = CDM(env)
cdm.load('models')

for i in range(100):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    cost = np.array(info['cost'])
    predicted_cost = cdm.predict(cost, observation, action)


    if terminated or truncated:
        observation, info = env.reset()
```