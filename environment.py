import gymnasium as gym
import ray
from gymnasium import spaces
import numpy as np
from ray import tune
import random
import math
from TSP_view_2D import TSPView2D
from mobile import Vehicle


class TSPEnv(gym.Env):
    def render(self, mode="human", close=False):

        if self.tsp_view is None:
            self.tsp_view = TSPView2D(self.n_orders, self.map_quad, grid_size=25)

        return self.tsp_view.update(
            self.agt_at_restaurant,
            self.restaurant_x,
            self.restaurant_y,
            self.o_delivery,
            self.o_x,
            self.o_y,
            self.agt_x,
            self.agt_y,
            mode,
        )

    def __init__(self, env_config):

        map_quad = env_config["map_quad"]
        n_orders = env_config["n_orders"]
        max_time = env_config["max_time"]
        randomized_orders = env_config["randomized_orders"]
        self.agent = Vehicle(implementation=env_config["implementation"])

        self.tsp_view = None
        self.map_quad = map_quad

        self.o_y = []
        self.o_x = []

        self.randomized_orders = randomized_orders

        self.n_orders = n_orders
        self.restaurant_x = 0
        self.restaurant_y = 0

        self.agt_x = None
        self.agt_y = None

        self.o_delivery = []
        self.o_time = []
        self.agt_at_restaurant = None
        self.agt_time = None

        self.max_time = max_time

        self.map_min_x = -map_quad[0]
        self.map_max_x = +map_quad[0]
        self.map_min_y = -map_quad[1]
        self.map_max_y = +map_quad[1]

        # agent x,
        agt_x_min = [self.map_min_x]
        agt_x_max = [self.map_max_x]
        # agent y,
        agt_y_min = [self.map_min_y]
        agt_y_max = [self.map_max_y]
        # n_orders for x positions of orders,
        o_x_min = [self.map_min_x for i in range(n_orders)]
        o_x_max = [self.map_max_x for i in range(n_orders)]
        # n_orders for y positions of orders,
        o_y_min = [self.map_min_y for i in range(n_orders)]
        o_y_max = [self.map_max_y for i in range(n_orders)]

        # whether delivered or not, 0 not delivered, 1 delivered
        o_delivery_min = [0] * n_orders
        o_delivery_max = [1] * n_orders

        # whether agent is at restaurant or not
        agt_at_restaurant_max = 1
        agt_at_restaurant_min = 0

        # Time since orders have been placed
        o_time_min = [0] * n_orders
        o_time_max = [max_time] * n_orders

        # Time since start
        agt_time_min = 0
        agt_time_max = max_time

        self.observation_space = spaces.Box(
            low=np.array(
                agt_x_min
                + agt_y_min
                + o_x_min
                + o_y_min
                + [0]
                + [0]
                + o_delivery_min
                + [agt_at_restaurant_min]
                + o_time_min
                + [agt_time_min]
                + [0]
            ),
            high=np.array(
                agt_x_max
                + agt_y_max
                + o_x_max
                + o_y_max
                + [0]
                + [0]
                + o_delivery_max
                + [agt_at_restaurant_max]
                + o_time_max
                + [agt_time_max]
                + [self.max_time]
            ),
            dtype=np.int16,
        )

        # Action space, UP, DOWN, LEFT, RIGHT
        self.action_space = spaces.Discrete(4)

    def reset(self):

        self.restaurant_x = 0
        self.restaurant_y = 0
        self.agt_x = self.restaurant_x
        self.agt_y = self.restaurant_y
        if self.randomized_orders:
            # Enforce uniqueness of orders, to prevent multiple orders being placed on the same points
            # And ensure actual orders in the episode are always == n_orders as expected
            orders = []
            while len(orders) != self.n_orders:
                orders += [self.__receive_order()]
                orders = list(set(orders))
        else:
            orders = [(-2, -2), (1, 1), (2, 0), (0, -2)]
        self.o_x = [x for x, y in orders]
        self.o_y = [y for x, y in orders]
        self.o_delivery = [0] * self.n_orders
        self.o_time = [0] * self.n_orders
        self.agt_at_restaurant = 1
        self.agt_time = 0

        return self.__compute_state()

    def step(self, action):

        done = False
        reward_before_action = self.__compute_reward()
        self.__play_action(action)
        reward = self.__compute_reward() - reward_before_action

        # If agent completed the route and returned back to start, give additional reward
        if (np.sum(self.o_delivery) == self.n_orders) and self.agt_at_restaurant:
            done = True
            reward += self.max_time * 0.1

        # If there is timeout, no additional reward
        if self.agt_time >= self.max_time:
            done = True

        info = {}
        return self.__compute_state(), reward, done, info

    def __play_action(self, action):
        # scan or not the area to find goal
        scan = False
        if action == 0:  # UP
            coords = [self.agt_x, min(self.map_max_y, self.agt_y + 1)]
            if coords[0] in self.o_x and coords[1] in self.o_y:
                scan = True
            self.agent.move_to(coords, scan)
        elif action == 1:  # DOWN
            coords = [self.agt_x, max(self.map_max_y, self.agt_y - 1)]
            if coords[0] in self.o_x and coords[1] in self.o_y:
                scan = True
            self.agent.move_to(coords, scan)
        elif action == 2:  # LEFT
            coords = [max(self.map_min_x, self.agt_x - 1), self.agt_y]
            if coords[0] in self.o_x and coords[1] in self.o_y:
                scan = True
            self.agent.move_to(coords, scan)
        elif action == 3:  # RIGHT
            coords = [min(self.map_max_x, self.agt_x + 1), self.agt_y]
            if coords[0] in self.o_x and coords[1] in self.o_y:
                scan = True
            self.agent.move_to(coords, scan)
        else:
            raise Exception("action: {action} is invalid")
        self.agt_x, self.agt_y = self.agent.get_position()[0], self.agent.get_position()[1]

        # Check for deliveries
        for ix in range(self.n_orders):
            if self.o_delivery[ix] == 0:
                if (self.o_x[ix] == self.agt_x) and (self.o_y[ix] == self.agt_y):
                    self.o_delivery[ix] = 1

        # Update the time for the waiting orders
        for ix in range(self.n_orders):
            if self.o_delivery[ix] == 0:
                self.o_time[ix] += 1

        # Update time since agent left restaurant
        self.agt_time += 1

        # Check if agent is at restaurant
        self.agt_at_restaurant = int(
            (self.agt_x == self.restaurant_x) and (self.agt_y == self.restaurant_y)
        )

    def __compute_state(self):
        return (
            [self.agt_x]
            + [self.agt_y]
            + self.o_x
            + self.o_y
            + [self.restaurant_x]
            + [self.restaurant_y]
            + self.o_delivery
            + [self.agt_at_restaurant]
            + self.o_time
            + [self.agt_time]
            + [(self.max_time - self.agt_time)]
        )

    def __receive_order(self):

        # Generate a single order, not at the center (where the restaurant is)
        self.order_x = np.random.choice(
            [i for i in range(self.map_min_x, self.map_max_x + 1) if i != self.restaurant_x], 1
        )[0]
        self.order_y = np.random.choice(
            [i for i in range(self.map_min_y, self.map_max_y + 1) if i != self.restaurant_y], 1
        )[0]

        return self.order_x, self.order_y

    def __compute_reward(self):
        return (
            np.sum(np.asarray(self.o_delivery) * self.max_time / (np.asarray(self.o_time) + 0.0001))
            - self.agt_time
        )



class TSPBatteryEnv(gym.Env):
    def render(self, mode="human", close=False):

        if self.tsp_view is None:
            self.tsp_view = TSPView2D(self.n_orders, self.map_quad, grid_size=25)

        return self.tsp_view.update(
            self.agt_at_restaurant,
            self.restaurant_x,
            self.restaurant_y,
            self.o_delivery,
            self.o_x,
            self.o_y,
            self.agt_x,
            self.agt_y,
            mode,
        )

    def __init__(self, env_config):

        map_quad = env_config["map_quad"]
        n_orders = env_config["n_orders"]
        max_time = env_config["max_time"]
        randomized_orders = env_config["randomized_orders"]
        self.agent = Vehicle(implementation=env_config["implementation"])

        self.tsp_view = None
        self.map_quad = map_quad

        self.o_y = []
        self.o_x = []

        self.randomized_orders = randomized_orders

        self.n_orders = n_orders
        self.restaurant_x = 0
        self.restaurant_y = 0

        self.agt_x = None
        self.agt_y = None

        self.agt_battery = None
        self.battery_eff = None

        self.o_delivery = []
        self.o_time = []
        self.order_distances = []
        self.agt_at_restaurant = None
        self.agt_time = None

        self.max_time = max_time

        self.map_min_x = -map_quad[0]
        self.map_max_x = +map_quad[0]
        self.map_min_y = -map_quad[1]
        self.map_max_y = +map_quad[1]

        # agent x,
        agt_x_min = [self.map_min_x]
        agt_x_max = [self.map_max_x]
        # agent y,
        agt_y_min = [self.map_min_y]
        agt_y_max = [self.map_max_y]

        #agent battery,
        agt_battery_min = [0]
        agt_battery_max = [100]
        # n_orders for x positions of orders,
        o_x_min = [self.map_min_x for i in range(n_orders)]
        o_x_max = [self.map_max_x for i in range(n_orders)]
        # n_orders for y positions of orders,
        o_y_min = [self.map_min_y for i in range(n_orders)]
        o_y_max = [self.map_max_y for i in range(n_orders)]

        # whether delivered or not, 0 not delivered, 1 delivered
        o_delivery_min = [0] * n_orders
        o_delivery_max = [1] * n_orders

        # distance from restaurant to orders in steps
        order_distances_min = [0] * n_orders
        order_distances_max = [self.map_max_x*2] * n_orders

        # whether agent is at restaurant or not
        agt_at_restaurant_max = 1
        agt_at_restaurant_min = 0

        # Time since orders have been placed
        o_time_min = [0] * n_orders
        o_time_max = [max_time] * n_orders

        # Time since start
        agt_time_min = 0
        agt_time_max = max_time

        self.observation_space = spaces.Box(
            low=np.array(
                agt_x_min
                + agt_y_min
                + o_x_min
                + o_y_min
                + [0]
                + [0]
                + o_delivery_min
                + [agt_at_restaurant_min]
                + o_time_min
                + [agt_time_min]
                + [0]
                + agt_battery_min
                + order_distances_min
            ),
            high=np.array(
                agt_x_max
                + agt_y_max
                + o_x_max
                + o_y_max
                + [0]
                + [0]
                + o_delivery_max
                + [agt_at_restaurant_max]
                + o_time_max
                + [agt_time_max]
                + [self.max_time]
                + agt_battery_max
                + order_distances_max
            ),
            dtype=np.int16,
        )

        # Action space, UP, DOWN, LEFT, RIGHT
        self.action_space = spaces.Discrete(4)

    def reset(self):

        self.restaurant_x = 0
        self.restaurant_y = 0
        self.agt_x = self.restaurant_x
        self.agt_y = self.restaurant_y
        self.agt_battery, self.battery_eff = self.agent.reset_battery()
        if self.randomized_orders:
            # Enforce uniqueness of orders, to prevent multiple orders being placed on the same points
            # And ensure actual orders in the episode are always == n_orders as expected
            orders = []
            while len(orders) != self.n_orders:
                orders += [self.__receive_order()]
                orders = list(set(orders))
        else:
            orders = [(-2, -2), (1, 1), (2, 0), (0, -2)]
        self.o_x = [x for x, y in orders]
        self.o_y = [y for x, y in orders]
        self.o_delivery = [0] * self.n_orders
        self.o_time = [0] * self.n_orders
        self.agt_at_restaurant = 1
        self.agt_time = 0
        self.order_distances = [abs(x) + abs(y) for x, y in orders]

        return self.__compute_state()

    def step(self, action):

        done = False
        reward_before_action = self.__compute_reward()
        self.__play_action(action)
        reward = self.__compute_reward() - reward_before_action # - (100 - self.agt_battery)/100

        # If agent completed the route and returned back to start, give additional reward
        if self.agt_at_restaurant:
            done = True
            reward += np.sum(self.o_delivery) * self.max_time - self.__adjust_reward()
            

        # If there is timeout, no additional reward
        if self.agt_time >= self.max_time:
            done = True
        
        if self.agt_battery <= 0 and not self.agt_at_restaurant:
            done = True
            reward -= self.max_time

        info = {}
        return self.__compute_state(), reward, done, info

    def __play_action(self, action):
        # scan or not the area to find goal
        scan = False
        if action == 0:  # UP
            coords = [self.agt_x, min(self.map_max_y, self.agt_y + 1)]
            if coords[0] in self.o_x and coords[1] in self.o_y:
                scan = True
            self.agent.move_to(coords, scan)
        elif action == 1:  # DOWN
            coords = [self.agt_x, max(self.map_max_y, self.agt_y - 1)]
            if coords[0] in self.o_x and coords[1] in self.o_y:
                scan = True
            self.agent.move_to(coords, scan)
        elif action == 2:  # LEFT
            coords = [max(self.map_min_x, self.agt_x - 1), self.agt_y]
            if coords[0] in self.o_x and coords[1] in self.o_y:
                scan = True
            self.agent.move_to(coords, scan)
        elif action == 3:  # RIGHT
            coords = [min(self.map_max_x, self.agt_x + 1), self.agt_y]
            if coords[0] in self.o_x and coords[1] in self.o_y:
                scan = True
            self.agent.move_to(coords, scan)
        else:
            raise Exception("action: {action} is invalid")
        self.agt_x, self.agt_y = self.agent.get_position()[0], self.agent.get_position()[1]

        # Check for deliveries
        for ix in range(self.n_orders):
            if self.o_delivery[ix] == 0:
                if (self.o_x[ix] == self.agt_x) and (self.o_y[ix] == self.agt_y):
                    self.o_delivery[ix] = 1

        # Update the time for the waiting orders
        for ix in range(self.n_orders):
            if self.o_delivery[ix] == 0:
                self.o_time[ix] += 1

        # Update time since agent left restaurant
        self.agt_time += 1

        # Check if agent is at restaurant
        self.agt_at_restaurant = int(
            (self.agt_x == self.restaurant_x) and (self.agt_y == self.restaurant_y)
        )

        #get new battery level
        self.agt_battery = self.agent.get_battery()[0]

    def __compute_state(self):
        return (
            [self.agt_x]
            + [self.agt_y]
            + self.o_x
            + self.o_y
            + [self.restaurant_x]
            + [self.restaurant_y]
            + self.o_delivery
            + [self.agt_at_restaurant]
            + self.o_time
            + [self.agt_time]
            + [(self.max_time - self.agt_time)]
            + [self.agt_battery]
            + self.order_distances
        )

    def __receive_order(self):

        # Generate a single order, not at the center (where the restaurant is)
        self.order_x = np.random.choice(
            [i for i in range(self.map_min_x, self.map_max_x + 1) if i != self.restaurant_x], 1
        )[0]
        self.order_y = np.random.choice(
            [i for i in range(self.map_min_y, self.map_max_y + 1) if i != self.restaurant_y], 1
        )[0]

        return self.order_x, self.order_y

    def __compute_reward(self):
        return (
            np.sum(np.asarray(self.o_delivery) * self.max_time / (np.asarray(self.o_time) + 0.0001))
            - self.agt_time
        )
    
    def __adjust_reward(self):
        factor = 0
        for i in range(len(self.order_distances)):
            if self.o_delivery[i] == 0:
                if self.order_distances[i] * self.battery_eff < self.agt_battery:
                    factor += 1
        return factor * self.max_time
                    

#initialisation de Ray pour l'entrainement
ray.init(ignore_reinit_error=True)

#On donne nos paramètres de configuration pour l'entrainement 
tune_config = {
    # select environment wanted : TSPEnv or TSPBatteryEnv
    "env": TSPEnv,
    #env_config est la configuration de notre environnement : nombre d'agent, d'objectif, ect
    "env_config": {
        "map_quad":(10,10),
        "n_orders":8,   
        "max_time":500,
        "randomized_orders":True,
        # "implementation":"simple"
    },
    "framework": "torch",  # ou "tf" pour TensorFlow
    #nombre d'agent qui seront entrainer en parallèle
    "num_workers": 4,
    "num_learner_workers" : 0,

    #Pour entrainer avec des GPU mettre "num_cpus" a 1 puis décomemnter la ligne " "num_gpus_per_worker": 1," et remplacer 2 pour le nombre de gpu voulu par workers
    "num_gpus": 0,
    #"num_gpus_per_worker": 2,

    #commenter cette ligne si num_gpus = 1 
    "num_cpus_per_worker": 5,

    #si 2 worker et 2 cpu ou gpu par worker alors 4 cpu ou gpu seront utilisé

    "model": {
        "fcnet_hiddens": [64, 64],  # Architecture du réseau de neurones (couches cachées)
    },
    "optimizer": {
        "learning_rate": 0.001,  # Taux d'apprentissage
    },
}
                                            #Condition de stop
analysis = tune.run("PPO", config=tune_config,stop={"timesteps_total": 10000000000000000000000000000000000000000000000000})