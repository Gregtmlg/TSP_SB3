import numpy as np
import math

# from bluerov_node import BlueRov

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

    def move_to(self, coordinates : list):
        self.current_position[0], self.current_position[1] = coordinates[0], coordinates[1]

    def get_position(self) -> np.ndarray:
        return self.current_position
    
    def get_battery(self) -> int:
        return self.battery
    
    def reset_battery(self):
        self.battery = 100


# class ArduSubBluerovImplementation(BluerovImplementation):
#     """
#         This class was created to build the bridge between functions/algorithms used to control 
#         Bluerov2 and the Deep Reinforcement Learning script that makes decisions. Use this class 
#         if you are using ArduSub and Unity.
#     """

#     def __init__(self):
#         self.current_position = np.zeros((1,3))
#         self.init_position = np.zeros((1,3))
#         self.distance_made = 0
#         self.bluerov = BlueRov(device='udp:localhost:14550')

#     def move_to(self, waypoint, wpnt_is_init=False):
#         if wpnt_is_init:
#             self.init_position = waypoint
#             self.distance_made = 0
#         else:
#             self.__calculate_distance_made(waypoint)
#         is_command_sent=False
#         self.current_position=self.bluerov.get_current_pose()
#         desired_position = [waypoint[0], waypoint[1], -waypoint[2], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
#         while abs(self.current_position[0] - waypoint[0]) > 0.02 and abs(self.current_position[1] - waypoint[1]) > 0.02:
#             if is_command_sent==False:
#                 self.bluerov.set_position_target_local_ned(desired_position)
#                 is_command_sent = True
#             self.bluerov.update()
#             self.current_position=self.bluerov.get_current_pose()
#             self.bluerov.publish()

#     def get_current_position(self):
#         self.bluerov.update()
#         self.current_position=self.bluerov.get_current_pose()        
#         return self.current_position

#     def get_distance_made(self):
#         return self.distance_made
    
#     def __calculate_distance_made(self, waypoint):
#         dx = self.current_position[0] - waypoint[0]
#         dy = self.current_position[1] - waypoint[1]
#         self.distance_made += math.hypot(dx,dy)