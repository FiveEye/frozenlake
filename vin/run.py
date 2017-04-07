from importlib import reload
import gym
import random
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Lambda, Reshape, Permute
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling3D, MaxPooling2D
from keras.layers.merge import concatenate
import keras.backend as K
#import matplotlib.pyplot as plt

from Agent import Agent

env = gym.make('FrozenLake8x8-v0')

NUM_EPISODES = 12000
agent = Agent(env)
jList = []
rList = []
for i in range(NUM_EPISODES):
    observation = env.reset()
    agent.init(observation)
    terminal = False
    rAll = 0
    j = 0
    while not terminal:
        j += 1
        state = observation
        action = agent.get_action(state)
        observation, reward, terminal, _ = env.step(action)
        agent.run(state, action, reward, terminal, observation)
        rAll += reward
        if terminal:
            break
            
    print("i=",i,reward)
    env.render()
    agent.print_map_info()
    jList.append(j)
    rList.append(rAll)
print("Percent of succesful episodes: " + str(sum(rList)/NUM_EPISODES) + "%")
            
