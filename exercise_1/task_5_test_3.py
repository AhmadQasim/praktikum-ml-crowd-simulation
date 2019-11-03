from tasks import SimulationEnv
import numpy as np
from fire import Fire
import sys


class SimulationTest3(SimulationEnv):
    def __init__(self, p_region_x, p_region_y, p_coord, **kwargs):
        # constants
        self.length = 30
        self.width = 5

        # variables
        self.p_locs = None
        self.p_region_x = p_region_x
        self.p_region_y = p_region_y
        self.p_coord = p_coord
        self.o_locs = None

        self.initialize_obstacles()
        super().__init__(**kwargs)

        self.p_num_per_box = (self.p_region_x*self.p_region_y)/self.p_num

        self.validate_arguments_pred()
        self.initialize_pred()

    def validate_arguments_pred(self):
        if self.p_num_per_box < 1:
            sys.exit("The number of pedestrians should be less then the number of boxes in the region")

    def initialize_pred(self):
        p_locs = list()
        curr_index = 0
        for i in range(self.p_num):

            p_loc = np.array(np.unravel_index(curr_index, (self.p_region_x, self.p_region_y))) + self.p_coord
            print(p_loc)

            p_locs.append(p_loc)
            curr_index += int(self.p_num_per_box)

        self.p_locs = np.array(p_locs)

    def initialize_obstacles(self):
        o_locs = list()
        for i in range(self.length-self.width):
            o_loc_x = self.p_coord[0] - 1
            o_loc_y = self.p_coord[1] + i

            o_locs.append([o_loc_x, o_loc_y])

            o_loc_x = self.p_coord[0] - 1 - i
            o_loc_y = self.p_coord[1] + self.length - self.width

            o_locs.append([o_loc_x, o_loc_y])

        for i in range(self.length + 1):
            o_loc_x = self.p_coord[0] + self.width
            o_loc_y = self.p_coord[1] + i

            o_locs.append([o_loc_x, o_loc_y])

            o_loc_x = self.p_coord[0] + self.width - i
            o_loc_y = self.p_coord[1] + self.length + 1

            o_locs.append([o_loc_x, o_loc_y])

        self.o_locs = np.array(o_locs)


if __name__ == '__main__':
    Fire(SimulationTest3)