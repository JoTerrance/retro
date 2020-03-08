import gym # openAi gym
import retro
from gym import envs
import numpy as np 
import datetime
import keras 
import tensorflow as tf
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from time import sleep

print("OK")

env = retro.make('SonicTheHedgehog-Genesis')
env.reset()
env.render()


rew_tot=0
obs= env.reset()
env.render()
for _ in range(6):
    action = env.action_space.sample() #take step using random action from possible actions (actio_space)
    obs, rew, done, info = env.step(action) 
    rew_tot = rew_tot + rew
    env.render()
#Print the reward of these random action
print("Reward: %r" % rew_tot)   



# action space has 6 possible actions, the meaning of the actions is nice to know for us humans but the neural network will figure it out
print(env.action_space)
NUM_ACTIONS = env.action_space.n
print("Possible actions: [0..%a]" % (NUM_ACTIONS-1))

print(env.observation_space)
print()


# Value iteration algorithem
NUM_ACTIONS = env.action_space.n
NUM_STATES = env.observation_space.shape[0]*env.observation_space.shape[1]*env.observation_space.shape[2]
V = np.zeros([NUM_STATES]) # The Value for each state
Pi = np.zeros([NUM_STATES], dtype=int)  # Our policy with we keep updating to get the optimal policy
gamma = 0.9 # discount factor
significant_improvement = 0.01

def best_action_value(s):
    # finds the highest value action (max_a) in state s
    best_a = None
    best_value = float('-inf')

    # loop through all possible actions to find the best current action
    for a in range (0, NUM_ACTIONS):
        env.env.s = s
        s_new, rew, done, info = env.step(a) #take the action
        v = rew + gamma * V[s_new]
        if v > best_value:
            best_value = v
            best_a = a
    return best_a

iteration = 0
while True:
    # biggest_change is referred to by the mathematical symbol delta in equations
    biggest_change = 0
    for s in range (0, NUM_STATES):
        old_v = V[s]
        action = best_action_value(s) #choosing an action with the highest future reward
        env.env.s = s # goto the state
        s_new, rew, done, info = env.step(action) #take the action
        V[s] = rew + gamma * V[s_new] #Update Value for the state using Bellman equation
        Pi[s] = action
        biggest_change = max(biggest_change, np.abs(old_v - V[s]))
    iteration += 1
    if biggest_change < significant_improvement:
        print (iteration,' iterations done')
        break

# Let's see how the algorithm solves the taxi game
rew_tot=0
obs= env.reset()
env.render()
done=False
while done != True: 
    action = Pi[obs]
    obs, rew, done, info = env.step(action) #take step using selected action
    rew_tot = rew_tot + rew
    env.render()
#Print the reward of these actions
print("Reward: %r" % rew_tot)  
