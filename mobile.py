from vehicle_implementations import SimpleImplementation # , ArduSubBluerovImplementation

class Vehicle:
    def __init__(self, implementation="simple"):
        """
            implementation : choose if you use a Bluerov2 or not ("bluerov2" or "simple")
        """
        if implementation == "simple":
            self.implementation = SimpleImplementation()
        # elif implementation == "bluerov2":
        #     self.implementation = ArduSubBluerovImplementation()
        else : 
            raise ValueError("Incorrect implementation value. Choose 'bluerov2' or 'simple'.")

    def move_to(self, coordinates, do_scan=False):
        self.implementation.move_to(coordinates, do_scan=do_scan)

    def get_position(self):
        return self.implementation.get_position()

    def get_battery(self):
        return self.implementation.get_battery()
    
    def reset_battery(self):
        return self.implementation.reset_battery()
