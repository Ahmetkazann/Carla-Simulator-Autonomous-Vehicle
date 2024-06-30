
from stable_baselines3 import PPO #PPO
from typing import Callable
import os
from CarlaenvCalisan import CarlaEnv
import time


print('Starting...')

models_dir = f"models/{int(time.time())}/"
logdir = f"logs/{int(time.time())}/"

if not os.path.exists(models_dir):
	os.makedirs(models_dir)

if not os.path.exists(logdir):
	os.makedirs(logdir)

print('Connecting to GYM Custom ENV...')

env = CarlaEnv()
print('Env action space:',env.action_space)
env.reset()
print('Reseting Environment Before Training...')
#model = PPO('MlpPolicy', env, verbose=1,learning_rate=0.001, tensorboard_log=logdir)
model = PPO('MultiInputPolicy', env, verbose=1,learning_rate=0.001, tensorboard_log=logdir)
#ValueError: You must use `MultiInputPolicy` when working with dict observation space, not MlpPolicy
print("Training")
TIMESTEPS = 500_000 # how long is each training iteration - individual steps
iters = 0
while iters<10:  # how many training iterations you want
	iters += 1
	print('Iteration ', iters,' is to commence...')
	model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO" )  # step function is abstract in model.learn method
	print('Iteration ', iters,' has been trained')
	model.save(f"{models_dir}/{TIMESTEPS*iters}")