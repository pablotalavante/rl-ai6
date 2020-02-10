import gym
env = gym.make("Acrobot-v1")
observation = env.reset()

for _ in range(1000):
  env.render()
  print(env.action_space)
  action = env.action_space.sample() # your agent here (this takes random actions)
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
