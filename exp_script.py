import os
import time
from time import sleep
from termcolor import colored
import matplotlib.pyplot as plt

from stable_baselines3 import TD3, PPO, A2C
from stable_baselines3.td3.policies import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

from environment import TSPEasyEnv, TSPMediumEnv, TSPHardEnv, TSP33Env, TSPEasyBatteryEnv, TSP33BatteryEnv, TSPMediumBatteryEnv

# refer here directories for model saving and logs
models_dir = "models/PPO_TSPEasyBattery"
logs_dir = "logs"

# create environment wanted
env = TSPEasyBatteryEnv()

# load model from model saving directory
model = PPO.load(f"{models_dir}/traveling_salesman_aws_10000000_easy_2.zip")

# running loop to check model predictions
for episode in range(10):
    obs = env.reset()
    done = False
    while not done:
        env.render()
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        time.sleep(0.5)
env.close()