import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import sys
sys.path.append('./deepq')
sys.path.append('./utils')

import pickle
import gym
import numpy as np
from cartpole import CartPoleEnv

from deepq.policies import MlpPolicy, LnMlpPolicy
from deepq.dqn2 import DQN


with_render = False
path_logs = f'./output/'
model_name = 'cp_model'

task = 0
gravitys = [9.8, 9.8*4]
simulations = ['masscart3.0_masspoles2.0', 'gravity39.2']

env = CartPoleEnv(gravity=gravitys[task])
state = env.reset()

num_episodes = 200
seq_states_episodes = []

for j in range(1,11):
    state = env.reset()
    model = DQN.load(f'{path_logs}/{simulations[task]}/{model_name}_{j}',env=env)
    for i in range(num_episodes):
        print(f'Trial {j} - Episode {i}')    
        done = False
        state = env.reset()
        sequence_states = [state]
        while not done:
            action, _ = model.predict(state)
            state, reward, done, _ = env.step(action)
            sequence_states.append(state)
        seq_states_episodes.append(sequence_states)

with open(f'{path_logs}/seq_states_T0.pkl', 'wb') as f:
    pickle.dump(seq_states_episodes, f)