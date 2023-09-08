import ray
from ray import tune
from environment import TSPEnv
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from ray.air import CheckpointConfig
import time


# #initialisation de Ray pour l'entrainement
# ray.init(ignore_reinit_error=True)

# #On donne nos paramètres de configuration pour l'entrainement 
# tune_config = {
#     # select environment wanted : TSPEnv or TSPBatteryEnv
#     "env": TSPEnv,
#     #env_config est la configuration de notre environnement : nombre d'agent, d'objectif, ect
#     "env_config": {
#         "map_quad":(2,2),
#         "n_orders":4,   
#         "max_time":50,
#         "randomized_orders":False,
#         "implementation":"simple"
#     },
#     "framework": "torch",  # ou "tf" pour TensorFlow
#     #nombre d'agent qui seront entrainer en parallèle
#     "num_workers": 1,
#     "num_learner_workers" : 0,

#     #Pour entrainer avec des GPU mettre "num_cpus" a 1 puis décomemnter la ligne " "num_gpus_per_worker": 1," et remplacer 2 pour le nombre de gpu voulu par workers
#     "num_gpus": 0,
#     #"num_gpus_per_worker": 2,

#     #commenter cette ligne si num_gpus = 1 
#     "num_cpus_per_worker": 5,

#     #si 2 worker et 2 cpu ou gpu par worker alors 4 cpu ou gpu seront utilisé

#     "model": {
#         "fcnet_hiddens": [64, 64],  # Architecture du réseau de neurones (couches cachées)
#     },
#     "optimizer": {
#         "learning_rate": 0.001,  # Taux d'apprentissage
#     },
# }
# # stop = Condition de stop, checkpoint_config = choose when to save training with checkpoint_frequency or check_point_at_end
# analysis = tune.run("PPO", name="PPO", config=tune_config,stop={"timesteps_total": 3000000}, checkpoint_config=CheckpointConfig(checkpoint_at_end=True))

# refer cheskpoint path from ray_results
algo = PPO.from_checkpoint("C:\\Users\\gregg\\ray_results\\PPO\\PPO_TSPEnv_7a4a9_00000_0_2023-09-07_20-58-05\\checkpoint_000750")

env_config = {
        "map_quad":(2,2),
        "n_orders":4,   
        "max_time":50,
        "randomized_orders":True,
        "implementation":"simple"
    }

env = TSPEnv(env_config=env_config)

done = False
obs = env.reset()

while not done:
    env.render()
    # check method compute_single_action to be sure it is a prediction from model
    action = algo.compute_single_action(obs)
    obs, reward, done, info = env.step(action)
    time.sleep(0.5)