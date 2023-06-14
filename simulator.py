from library.simulator_func_mysql import *
         
  
class Simulator:
    def __init__(self):
        self.print_info()  
        self.input_value()

    def print_info(self):  
        # simulate number setting
        self.simul_num = int(input("Please enter the number of the algorithm you want to simulate: "))

        # self.simul_reset Setting!
        #       'y'  :  "Initialize the simulator database corresponding to the number set in self.simul_num and run from the beginning."
        #       'n' : "Continue running the simulator database corresponding to the number set in self.simul_num without initialization."
        #       "Ex) If you have completed the simulator until June 3, 2022, and want to continue testing the simulator consecutively from June 3, 2022."

        option = str(input("Do you want to initialize the simulation database: (y or n) "))
          
        if option == 'y':
            self.simul_reset = 'reset'  
        elif option == 'n':
            self.simul_reset = 'continue' 
        else:
            print("Only 'y' or 'n' (lowercase) can be entered.")
            exit(1)  
  
    def input_value(self):     
        # simulator_func_mysql library 
        simulator_func_mysql(self.simul_num, self.simul_reset, 0)

 
if __name__ == "__main__":
    # simulator 클래스 호출
    Simulator()
    