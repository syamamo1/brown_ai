from touchscreen_helpers.generate_data import create_simulations
from touchscreen_helpers.simulator import *


output = create_simulations(20, 5000)
file = open("output_messing_around.txt", "w")
file.write(str(output))
file.close()
