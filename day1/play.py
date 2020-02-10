import gym
from gym.utils.play import play, PlayPlot

import numpy as np

observations = []
actions = []
i = 0

def callback(obs_t, obs_tp1, action, rew, done, info):
    observations.append(obs_t)
    actions.append(action)
    global i
    i += 1
    if i%250 == 0:
        np.save('obs', np.array(observations))
        np.save('actions', np.array(actions))


env = gym.make("Pong-ram-v4")
play(env, callback=callback, fps=10)
