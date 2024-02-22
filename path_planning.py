from functools import cached_property
from typing import List, NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pyswarms as ps
from matplotlib.patches import Circle


class Obstacle(NamedTuple):
    x: float
    y: float
    radius: float


class Path:
    def __init__(self, position: np.ndarray, start: tuple, end: tuple):
        """
        position: optimizer.swarm.position (waypoint_n * 2, )
        start, end: np.ndarray (2, )
        """
        self.start, self.end = start, end
        self.position = position

    @cached_property
    def waypoints(self) -> np.ndarray:
        waypoint_n = int(self.position.shape[0] / 2)  # 要转化成 int，否则会报错
        pos_x = self.position[:waypoint_n]
        pos_y = self.position[waypoint_n:]
        waypoints = np.stack((pos_x, pos_y), axis=1)
        return np.vstack((self.start, waypoints, self.end))  # not sure

    @cached_property
    def length(self) -> float:
        waypoints = self.waypoints
        return np.sum(np.linalg.norm(waypoints[1:] - waypoints[:-1], axis=1))


def show(path: Path, obstacles: List[Obstacle] = []):
    _, ax = plt.subplots()

    # plot start and end
    s, e = path.start, path.end
    ax.scatter(s[0], s[1], marker="x", label="start")
    ax.scatter(e[0], e[1], marker="x", label="end")

    # set axis limits
    ax.set_aspect("equal")
    ax.set_xlim(min(s[0], e[0]), max(s[0], e[0]))
    ax.set_ylim(min(s[1], e[1]), max(s[1], e[1]))

    # plot obstacle
    for obstacle in obstacles:
        circle = Circle((obstacle.x, obstacle.y), obstacle.radius)
        ax.add_patch(circle)

    # plot path
    ax.plot(path.waypoints[:, 0], path.waypoints[:, 1])
    ax.scatter(path.waypoints[:, 0], path.waypoints[:, 1], marker="x")

    plt.show()


def evaluate(positions: np.ndarray) -> np.ndarray:
    """positions: optimizer.swarm.position
    return: cost
    """
    n_particles, waypoint_n = positions.shape[0], int(positions.shape[1] / 2)

    # reshape matrix
    pos_x = positions[:, :waypoint_n]
    pos_y = positions[:, waypoint_n:]
    positions = np.stack((pos_x, pos_y), axis=2)  # (n_particles, waypoint_n, 2)

    # add start and end point
    start_point = np.zeros((n_particles, 1, 2))
    end_point = np.ones((n_particles, 1, 2)) * 10
    positions = np.concatenate((start_point, positions, end_point), axis=1)

    path_diff = positions[:, 1:] - positions[:, :-1]
    path_lengths = np.sum(np.linalg.norm(path_diff, axis=2), axis=1)
    return path_lengths


# setup
obstacles = [Obstacle(3, 3, 2), Obstacle(7, 5, 1)]
waypoint_n = 5
x_limit = (np.array([-10] * waypoint_n * 2), np.array([10] * waypoint_n * 2))

# pso
options = {"c1": 0.5, "c2": 0.3, "w": 0.9}
optimizer = ps.single.GlobalBestPSO(
    n_particles=50, dimensions=2 * waypoint_n, bounds=x_limit, options=options
)
cost, pos = optimizer.optimize(evaluate, iters=100)

# show
path = Path(pos, (0, 0), (10, 10))
show(path, obstacles)
