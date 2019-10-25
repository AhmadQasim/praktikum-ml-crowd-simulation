from tasks import SimulationEnv
import numpy as np
from fire import Fire


class SimulationTest5(SimulationEnv):
    def __init__(self, c_locs, p_density, **kwargs):
        super().__init__(**kwargs)
        self.c_locs = np.array(c_locs)
        self.p_density = p_density

        # constants
        self.c_size = 5
        self.c_sum = np.zeros(shape=(1, len(c_locs)))

        # initializations
        self.initialize_pred()

    def simulate(self):
        total_timesteps = 0
        for timestep in range(self.timesteps):
            if self.update_grid():
                if timestep < 180:
                    self.c_check()
                total_timesteps += 1
            else:
                break

        print("Average: ", self.c_sum/180)
        print("Density: ", (self.c_sum/180)/np.square(self.c_size))
        print("Flow: ", (self.c_sum / 180) / np.square(self.c_size) * 1 * 3)

    def c_check(self):
        for i, c_loc in enumerate(self.c_locs):
            c_sum_i = np.sum(self.grid[c_loc[0]:c_loc[0] + self.c_size, c_loc[1]:c_loc[1] + self.c_size] == self.p_code)
            self.c_sum[0, i] += c_sum_i

    def initialize_pred(self):
        left = self.c_locs[np.argmin(self.c_locs[:, 1]), 1] - 180
        right = self.c_locs[np.argmax(self.c_locs[:, 1]), 1] + 5

        p_locs = set()

        for i in range(left, right, 25):
            for p in range(self.p_density):
                p_loc_x = np.random.randint(0, self.grid_size_x)
                p_loc_y = np.random.randint(i, i + 25)

                p_loc = (p_loc_x, p_loc_y)

                while p_loc in p_locs:
                    p_loc_x = np.random.randint(0, self.grid_size_x)
                    p_loc_y = np.random.randint(i, i + 25)

                    p_loc = (p_loc_x, p_loc_y)

                p_locs.add(p_loc)

        p_locs = list(p_locs)

        self.p_locs = np.array(p_locs)

        for p_loc in self.p_locs:
            self.grid[p_loc[0], p_loc[1]] = self.p_code


if __name__ == '__main__':
    Fire(SimulationTest5)
