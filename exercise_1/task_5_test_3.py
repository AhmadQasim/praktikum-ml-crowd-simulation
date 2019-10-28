from tasks import SimulationEnv
import numpy as np
from fire import Fire


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

        self.initialize_pred()

    def initialize_pred(self):
        p_locs = set()
        for i in range(self.p_num):
            p_loc_x = np.random.randint(self.p_coord[0], self.p_coord[0] + self.p_region_x)
            p_loc_y = np.random.randint(self.p_coord[1], self.p_coord[1] + self.p_region_y)
            p_loc = (p_loc_x, p_loc_y)

            while p_loc in p_locs:
                p_loc_x = np.random.randint(self.p_coord[0], self.p_coord[0] + self.p_region_x)
                p_loc_y = np.random.randint(self.p_coord[1], self.p_coord[1] + self.p_region_y)

                p_loc = (p_loc_x, p_loc_y)

            p_locs.add(p_loc)

        p_locs = list(p_locs)

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