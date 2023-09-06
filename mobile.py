from vehicle_implementations import SimpleImplementation, ArduSubBluerovImplementation

class Vehicle:
    def __init__(self, implementation="simple"):
        """
            implementation : choose if you use ArduSub or not ("ardusub" or "simple")
        """
        if implementation == "simple":
            self.implementation = SimpleImplementation()
        elif implementation == "ardusub":
            self.implementation = ArduSubBluerovImplementation()
        else : 
            raise ValueError("Incorrect implementation value. Choose 'ardusub' or 'simple'.")

    def move_to(self, coordinates):
        self.implementation.move_to(coordinates)

    def get_position(self):
        return self.implementation.get_position()

    def get_battery(self):
        return self.implementation.get_battery()
    
    def reset_battery(self):
        return self.implementation.reset_battery()
