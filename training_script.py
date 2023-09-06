import os
import time
from time import sleep
from termcolor import colored
import matplotlib.pyplot as plt

from stable_baselines3 import TD3, PPO, A2C
from stable_baselines3.td3.policies import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

from environment import TSPEasyEnv, TSPMediumEnv, TSPHardEnv, TSP33Env, TSPEasyBatteryEnv, TSPMediumBatteryEnv, TSPHardBatteryEnv, TSP33BatteryEnv

# Set the parameters for the implementation
MAX_TIMESTEPS = 750000000  # Maximum number of steps to perform
SAVE_TIMESTEPS = 150000000 # save model every SAVE_TIMESTEPS step

# refer here directories for model saving and logs
models_dir = "models/PPOEasyBatterybattery"
logs_dir = "logs_aws"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logs_dir):
    os.makedirs(logs_dir) 

# create environment wanted
env = TSPEasyBatteryEnv()

# create model wanted from stable baselines 3 algorithms and set training parameters
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logs_dir, device="auto")
#model.set_parameters(f"{models_dir}/traveling_salesman_aws_20000000_battery.zip")
iteration = 0
# training loop saving models every SAVE_TIMESTEPS steps
while SAVE_TIMESTEPS * iteration < MAX_TIMESTEPS:
    iteration += 1
    model.learn(total_timesteps=SAVE_TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO_Easybattery")
    model.save(f"{models_dir}/traveling_salesman_aws_{SAVE_TIMESTEPS * iteration}_easy_battery")
env.close()