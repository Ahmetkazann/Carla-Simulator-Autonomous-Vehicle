
from stable_baselines3 import PPO #PPO
from typing import Callable
import os
from myscript_9 import CarlaEnv
import time


print('This is the start of training script')

print('setting folders for logs and models')
models_dir = f"models/{int(time.time())}/"
logdir = f"logs/{int(time.time())}/"

if not os.path.exists(models_dir):
	os.makedirs(models_dir)

if not os.path.exists(logdir):
	os.makedirs(logdir)

print('connecting to env..')

env = CarlaEnv()
print('Env action space:',env.action_space)
env.reset()
print('Env has been reset as part of launch')
model = PPO('MlpPolicy', env, verbose=1,learning_rate=0.001, tensorboard_log=logdir)

TIMESTEPS = 500_000 # how long is each training iteration - individual steps
iters = 0
while iters<10:  # how many training iterations you want
	iters += 1
	print('Iteration ', iters,' is to commence...')
	model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO" )  # step function is abstract in model.learn method
	print('Iteration ', iters,' has been trained')
	model.save(f"{models_dir}/{TIMESTEPS*iters}")