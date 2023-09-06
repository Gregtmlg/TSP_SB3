import numpy as np
import math

from bluerov2_bridge.bluerov_node import BlueRov

class VehicleImplementation:
    
    def move_to(self, coordinates):
        pass

    def get_position(self):
        pass

    def get_battery(self):
        pass

    def reset_battery(self):
        pass



class SimpleImplementation(VehicleImplementation):
    """
        This class was created to build the bridge between functions/algorithms used to control 
        Bluerov2 and the Deep Reinforcement Learning script that makes decisions. Use this class
        to have a simplified version without ArduSub simulation.
    """

    def __init__(self):
        self.current_position = np.zeros(3)
        self.battery = self.reset_battery()
        self.battery_efficiency = 0.1

    def move_to(self, coordinates : list, do_scan=False):
        self.current_position[0], self.current_position[1] = coordinates[0], coordinates[1]

    def get_position(self) -> np.ndarray:
        return self.current_position
    
    def get_battery(self) -> int:
        return self.battery
    
    def reset_battery(self):
        self.battery = 100


class ArduSubBluerovImplementation(VehicleImplementation):
    """
        This class was created to build the bridge between functions/algorithms used to control 
        Bluerov2 and the Deep Reinforcement Learning script that makes decisions. Use this class 
        if you are using ArduSub and Unity.
    """

    def __init__(self):
        self.current_position = np.zeros(3)
        self.battery = self.reset_battery()
        self.battery_efficiency = 0.1
        self.bluerov = BlueRov(device='udp:localhost:14550')

    def move_to(self, coordinates, do_scan=False):
        if do_scan:
            self.bluerov.do_scan([coordinates[0] * 10, coordinates[1] * 10, 0])
        else:
            init = [self.current_position[0] * 10, self.current_position[1] * 10, 0]
            goal = [coordinates[0] * 10, coordinates[1] * 10, 0]
            self.bluerov.do_evit(init, goal)

    def get_position(self):        
        return self.current_position