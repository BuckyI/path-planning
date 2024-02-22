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

    @property
    def waypoints(self) -> np.ndarray:
        waypoint_n = int(self.position.shape[0] / 2)  # 要转化成 int，否则会报错
        pos_x = self.position[:waypoint_n]
        pos_y = self.position[waypoint_n:]
        waypoints = np.stack((pos_x, pos_y), axis=1)
        return np.vstack((self.start, waypoints, self.end))  # not sure

    @property
    def length(self) -> float:
        waypoints = self.waypoints
        return np.sum(np.linalg.norm(waypoints[1:] - waypoints[:-1], axis=1))


def show(obstacles: List[Obstacle]):
    fig, ax = plt.subplots()
    for obstacle in obstacles:
        circle = Circle((obstacle.x, obstacle.y), obstacle.radius)
        ax.add_patch(circle)

    # 设置坐标轴范围
    ax.set_aspect("equal")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    # 显示图形
    plt.show()


def evaluate(positions: np.ndarray) -> np.ndarray:
    """positions: optimizer.swarm.position
    return: cost
    """
    n_particles = positions.shape[0]

    # 重新整理矩阵形式
    pos_x = positions[:, :waypoint_n]
    pos_y = positions[:, waypoint_n:]
    positions = np.stack((pos_x, pos_y), axis=2)  # (n_particles, waypoint_n, 2)

    # 路径中添加起点和终点
    start_point = np.zeros((n_particles, 1, 2))
    end_point = np.ones((n_particles, 1, 2)) * 10
    positions = np.concatenate((start_point, positions, end_point), axis=1)

    path_diff = positions[:, 1:] - positions[:, :-1]
    path_lengths = np.sum(np.linalg.norm(path_diff, axis=2), axis=1)
    return path_lengths


obstacles = [Obstacle(3, 3, 2), Obstacle(7, 5, 1)]
waypoint_n = 5
bounds = (np.array([-10] * waypoint_n * 2), np.array([10] * waypoint_n * 2))
# show(obstacles)
options = {"c1": 0.5, "c2": 0.3, "w": 0.9}
optimizer = ps.single.GlobalBestPSO(
    n_particles=50, dimensions=2 * waypoint_n, bounds=bounds, options=options
)
cost, pos = optimizer.optimize(evaluate, iters=1000)

plt.figure()

p = Path(pos, (0, 0), (10, 10))
waypoints = p.waypoints
plt.plot(waypoints[:, 0], waypoints[:, 1])
plt.scatter(waypoints[:, 0], waypoints[:, 1])
plt.show()
