import gym

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class Net(nn.Module):

    def __init__(self, input_shape):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_shape, 164)
        self.fc2 = nn.Linear(164, 64)
        self.fc3 = nn.Linear(64, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


env = gym.make("Pong-ram-v4")
observation = env.reset()

model_path = 'model_weights'

model = Net(observation.shape[0])
model.load_state_dict(torch.load(model_path))
model.eval()

take_action = lambda x: 0 if x==0 else x+1

for _ in range(1000):
  env.render()
  obs_input = torch.Tensor(observation)
  out = model(obs_input).detach().numpy()

  action = take_action(np.argmax(out))

  observation, reward, done, info = env.step(action)

  print('observation', observation)
  print('reward', reward)
  print('done', done)
  print('action', action)
  print('info', info)
  
  if done:
    observation = env.reset()
    print('done')
env.close()
