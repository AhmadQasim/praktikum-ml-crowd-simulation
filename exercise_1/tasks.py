import numpy as np
import fire
import sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import colors
import matplotlib

matplotlib.use("TkAgg")


class SimulationEnv:
    def __init__(self, grid_size_x, grid_size_y, p_locs, t_locs, o_locs, timesteps, disable_pedestrians=0,
                 avoid_obstacles=1, p_locs_mode="custom", p_locs_radius=0, p_num=0, mode="normal"):
        # arguments
        self.grid_size_x = np.array(grid_size_x)
        self.grid_size_y = np.array(grid_size_y)
        self.grid = None
        self.p_locs = np.array(p_locs)
        self.t_locs = np.array(t_locs)
        self.p_locs_mode = p_locs_mode

        if self.p_locs_mode != 'uniform':
            self.o_locs = np.array(o_locs)
        self.timesteps = np.array(timesteps)
        self.p_locs_radius = p_locs_radius
        self.p_num = p_num
        self.mode = mode
        self.disable_pedestrians = disable_pedestrians
        self.avoid_obstacles = avoid_obstacles

        # constants
        self.max_grid_size = 1000000
        self.colors = colors.ListedColormap(["White", "Blue", "Red", "Black", "lightgray", "Pink"])
        self.p_code = 1
        self.t_code = 2
        self.o_code = 3
        self.path_code = 4
        self.reached_code = 5
        self.eight_neighbors = np.array([[-1, -1], [0, -1], [1, -1],
                                         [-1, 0], [1, 0],
                                         [-1, 1], [0, 1], [1, 1]])
        self.four_neighbors = np.array([[0, -1],
                                        [-1, 0], [1, 0],
                                        [0, 1]])
        self.neighbors = self.four_neighbors
        self.fig = plt.figure(figsize=(16, 16))
        self.animation = []
        self.p_locs_modes = ['custom', 'circle', 'random', 'uniform']
        self.modes = ['normal', 'dijkstra']
        self.r_max = int(np.sqrt(self.grid_size_x))

        # variables
        self.dijkstra_cost = None

        # initialization
        self.validate_arguments()
        self.initialize_grid()
        self.visualize_grid()

    def simulate(self):
        total_timesteps = 0
        for timestep in range(self.timesteps - 1):
            if self.update_grid():
                self.visualize_grid()
                total_timesteps += 1
            else:
                self.visualize_grid()
                break
        self.show_animation(total_timesteps)

    def update_grid(self):
        target_reached = np.zeros(shape=(1, self.p_locs.shape[0]))
        for i, p_loc in enumerate(self.p_locs):
            neighbors_p = self.get_neighbors(p_loc)
            dist = np.zeros(shape=neighbors_p.shape[0])
            for j, neighbor_p in enumerate(neighbors_p):
                if self.mode == 'normal':
                    dist[j] = np.linalg.norm(neighbor_p - self.t_locs[0])
                    if self.avoid_obstacles:
                        dist[j] += self.calculate_cost_obstacles(neighbor_p)
                    dist[j] += self.calculate_cost_pedestrian(neighbor_p, i)
                elif self.mode == 'dijkstra':
                    dist[j] = self.dijkstra_cost[neighbor_p[0], neighbor_p[1]]
                    dist[j] += self.calculate_cost_pedestrian(neighbor_p, i)
            min_neighbor_p = np.argmin(dist)
            if np.linalg.norm(neighbors_p[min_neighbor_p, :] - self.t_locs[0]) != 0:
                # if np.sum(np.abs(neighbors_p[min_neighbor_p, :] - self.o_locs), axis=1).all():
                self.grid[p_loc[0], p_loc[1]] = self.path_code
                self.p_locs[i] = neighbors_p[min_neighbor_p, :]
                self.grid[p_loc[0], p_loc[1]] = self.p_code
            elif not self.disable_pedestrians:
                self.grid[p_loc[0], p_loc[1]] = self.reached_code
                target_reached[0, i] = 1
            else:
                target_reached[0, i] = 1

        if np.sum(target_reached) == self.p_locs.shape[0]:
            return False
        elif not self.disable_pedestrians:
            self.p_locs = self.p_locs[target_reached[0] == 0, :]
            return True
        else:
            return True

    def initialize_grid_dijkstra(self):
        n_visited = np.zeros(shape=(self.grid_size_x, self.grid_size_y))
        n_cost = np.ones(shape=(int(self.grid_size_x), int(self.grid_size_y))) * np.inf
        c_n = None

        # set the obstacle nodes as visited
        for o_loc in self.o_locs:
            n_visited[o_loc[0], o_loc[1]] = 1
            n_cost[o_loc[0], o_loc[1]] = np.inf
            self.dijkstra_cost[o_loc[0], o_loc[1]] = np.inf

        # set the target nodes' distance as zero
        for t_loc in self.t_locs:
            n_visited[t_loc[0], t_loc[1]] = 1
            n_cost[t_loc[0], t_loc[1]] = 0
            self.dijkstra_cost[t_loc[0], t_loc[1]] = 0
            c_n = t_loc

        # while all nodes are not visited
        while not np.all(n_visited):
            c_neighs = self.get_neighbors(c_n)
            for c_neigh in c_neighs:
                # skip the neighbors which are already set to visited
                if n_visited[c_neigh[0], c_neigh[1]] == 1:
                    continue
                if n_cost[c_neigh[0], c_neigh[1]] > n_cost[c_n[0], c_n[1]] + 1:
                    n_cost[c_neigh[0], c_neigh[1]] = n_cost[c_n[0], c_n[1]] + 1

            # mark current node as visited
            n_visited[c_n[0], c_n[1]] = 1
            self.dijkstra_cost[c_n[0], c_n[1]] = n_cost[c_n[0], c_n[1]]

            # get unvisited node with minimum cost
            c_n = np.unravel_index(np.argmin(np.where(n_visited == 0, n_cost, np.inf)), n_cost.shape)

    def get_neighbors(self, loc):
        neighbors_p = loc - self.neighbors
        neighbors_p[neighbors_p < 0] = 0
        neighbors_p[neighbors_p[:, 0] >= self.grid_size_x, 0] = self.grid_size_x - 1
        neighbors_p[neighbors_p[:, 1] >= self.grid_size_y, 1] = self.grid_size_y - 1

        return neighbors_p

    def calculate_cost_obstacles(self, neighbor_p):
        dist = 0
        for o_loc in self.o_locs:
            r = np.linalg.norm(neighbor_p - o_loc)
            dist += self.r_max / r

        return dist

    def calculate_cost_pedestrian(self, neighbor_p, i):
        dist = 0
        for j, p_loc in enumerate(self.p_locs):
            r = np.linalg.norm(neighbor_p - p_loc)
            if i == j:
                continue
            dist += 1.5 / (r + 1e-5)

        return dist

    def validate_arguments(self):
        if self.grid_size_x is None or self.grid_size_y is None or \
                self.grid_size_x > self.max_grid_size or self.grid_size_y > self.max_grid_size:
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
        self.grid = np.zeros((self.grid_size_x, self.grid_size_y))

        self.grid[0, 0] = self.reached_code
        self.grid[self.grid_size_x - 1, self.grid_size_y - 1] = self.o_code

        self.dijkstra_cost = np.zeros((self.grid_size_x, self.grid_size_y))
        self.dijkstra_cost[self.dijkstra_cost == 0] = np.inf

        if self.mode == 'dijkstra':
            self.initialize_grid_dijkstra()

        if self.p_locs_mode == "custom":
            for p_loc in self.p_locs:
                self.grid[p_loc[0], p_loc[1]] = self.p_code
        elif self.p_locs_mode == "circle":
            points = [(int((np.cos(2 * np.pi / self.p_num * x) * self.p_locs_radius) + self.t_locs[0, 0]),
                       int((np.sin(2 * np.pi / self.p_num * x) * self.p_locs_radius)) + self.t_locs[0, 1])
                      for x in range(self.p_num)]
            self.p_locs = np.array(points)
            for p in self.p_locs:
                self.grid[p[0]][p[1]] = self.p_code
        for t_loc in self.t_locs:
            self.grid[t_loc[0], t_loc[1]] = self.t_code
        for o_loc in self.o_locs:
            self.grid[o_loc[0], o_loc[1]] = self.o_code

    def visualize_grid(self):
        self.animation.append([plt.imshow(self.grid, interpolation='none', cmap=self.colors)])

    def show_animation(self, total_timesteps):
        anim = animation.ArtistAnimation(self.fig, self.animation, interval=333, blit=True, repeat=False)
        plt.title("Total Timesteps: %s" % str(total_timesteps))
        plt.show()


if __name__ == '__main__':
    fire.Fire(SimulationEnv)
