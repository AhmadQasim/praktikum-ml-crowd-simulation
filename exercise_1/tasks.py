import numpy as np
import fire
import sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib

matplotlib.use("TkAgg")


class SimulationEnv:
    def __init__(self, grid_size, p_locs, t_locs, o_locs, timesteps, p_locs_mode="custom", p_locs_radius=0, p_num=0,
                 mode="normal"):
        # arguments
        self.grid_size = np.array(grid_size)
        self.grid = None
        self.p_locs = np.array(p_locs)
        self.t_locs = np.array(t_locs)
        self.o_locs = np.array(o_locs)
        self.timesteps = np.array(timesteps)
        self.p_locs_mode = p_locs_mode
        self.p_locs_radius = p_locs_radius
        self.p_num = p_num
        self.mode = mode

        # constants
        self.max_grid_size = 1000000
        self.p_code = 1
        self.t_code = 2
        self.o_code = 3
        self.eight_neighbors = np.array([[-1, -1], [0, -1], [1, -1],
                                         [-1, 0], [1, 0],
                                         [-1, 1], [0, 1], [1, 1]])
        self.four_neighbors = np.array([[0, -1],
                                        [-1, 0], [1, 0],
                                        [0, 1]])
        self.neighbors = self.four_neighbors
        self.fig = plt.figure(figsize=(16, 16))
        self.animation = []
        self.p_locs_modes = ['custom', 'circle']
        self.modes = ['normal', 'dijkstra']
        self.r_max = int(np.sqrt(self.grid_size))

        # variables
        self.dijkstra_cost = None

        # initialization
        self.validate_arguments()
        self.initialize_grid()
        self.visualize_grid()

    def simulate(self):
        for timestep in range(self.timesteps - 1):
            self.update_grid()
            self.visualize_grid()
        self.show_animation()

    def update_grid(self):
        for i, p_loc in enumerate(self.p_locs):
            neighbors_p = p_loc - self.neighbors
            neighbors_p[neighbors_p < 0] = 0
            neighbors_p[neighbors_p > self.grid_size] = self.grid_size
            dist = np.zeros(shape=neighbors_p.shape[0])
            for j, neighbor_p in enumerate(neighbors_p):
                dist[j] = np.linalg.norm(neighbor_p - self.t_locs[0])

                # obstacle and pedestrian avoidance
                # dist[j] += self.calculate_cost(neighbor_p, i)
            min_neighbor_p = np.argmin(dist)
            if np.linalg.norm(neighbors_p[min_neighbor_p, :] - self.t_locs[0]) != 0 and not\
                    np. equal(neighbors_p[min_neighbor_p, :], self.o_locs).all(axis=1).any():
                self.grid[p_loc[0], p_loc[1]] = 0
                self.p_locs[i] = neighbors_p[min_neighbor_p, :]
                self.grid[p_loc[0], p_loc[1]] = self.p_code

    def update_grid_dijkstra(self):
        pass

    def calculate_cost(self, neighbor_p, i):
        dist = 0
        for o_loc in self.o_locs:
            r = np.linalg.norm(neighbor_p - o_loc)
            dist += self.r_max / r

        for j, p_loc in enumerate(self.p_locs):
            r = np.linalg.norm(neighbor_p - p_loc)
            if i == j:
                continue
            dist += 1.5 / (r + 1e-5)

        return dist

    def validate_arguments(self):
        if self.grid_size.ndim != 0 or self.grid_size > self.max_grid_size:
            sys.exit("The grid_size should be a single number which is less then 1M")
        if self.p_locs.ndim == 0:
            sys.exit("Pedestrian location is required")
        if self.t_locs.ndim == 0:
            sys.exit("Target location is required")
        if self.o_locs.ndim == 0:
            self.o_locs = np.array([])
        if self.timesteps.ndim != 0:
            sys.exit("The timesteps should be a single number")
        if self.p_locs_mode == 'circle' and (self.p_locs_radius <= 0 or self.p_num <= 0):
            sys.exit("If the Pedestrian location is set to circle mode then the radius and number"
                     " should be a positive number")
        if self.p_locs_mode not in self.p_locs_modes:
            sys.exit("The Pedestrian Location modes can only be, one of: " + str(self.p_locs_modes))
        if self.mode not in self.modes:
            sys.exit("The Path finding modes can only be, one of: " + str(self.p_locs_modes))

    def initialize_grid(self):
        self.grid = np.zeros((self.grid_size, self.grid_size))

        self.dijkstra_cost = np.zeros((self.grid_size, self.grid_size))
        self.dijkstra_cost[self.dijkstra_cost == 0] = np.inf

        if self.p_locs_mode == "custom":
            for p_loc in self.p_locs:
                self.grid[p_loc[0], p_loc[1]] = self.p_code
        elif self.p_locs_mode == "circle":
            points = [(int((np.cos(2 * np.pi / self.p_num * x) * self.p_locs_radius) + self.t_locs[0, 0]),
                       int((np.sin(2 * np.pi / self.p_num * x) * self.p_locs_radius)) + self.t_locs[0, 1])
                      for x in range(self.p_num)]
            self.p_locs = np.array(points)
        for t_loc in self.t_locs:
            self.grid[t_loc[0], t_loc[1]] = self.t_code
            self.dijkstra_cost[t_loc[0], t_loc[1]] = 0
        for o_loc in self.o_locs:
            self.grid[o_loc[0], o_loc[1]] = self.o_code

    def visualize_grid(self):
        self.animation.append([plt.imshow(self.grid, interpolation='none')])

    def show_animation(self):
        anim = animation.ArtistAnimation(self.fig, self.animation, interval=200, blit=True, repeat=False)
        plt.show()


if __name__ == '__main__':
    fire.Fire(SimulationEnv)
