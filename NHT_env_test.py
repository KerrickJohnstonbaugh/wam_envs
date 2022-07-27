import wam_envs
import wam_envs.action_interfaces.NHT_WAMWipe
import gym
import numpy as np

k=3
my_NHT_env = gym.make('NHT_WAMWipe-v1', action_dim=k)
my_NHT_env.reset()

qdot = np.zeros(k)
my_NHT_env.step(qdot)
