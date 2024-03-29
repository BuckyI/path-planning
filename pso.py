from functools import cached_property
from typing import List, NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pyswarms as ps
from matplotlib.patches import Circle
from pyswarms.utils.plotters import plot_contour, plot_cost_history, plot_surface
from scipy.interpolate import make_interp_spline


class Point(NamedTuple):
    x: float
    y: float


class Obstacle(NamedTuple):
    center: Point
    radius: float

    def inside(self, point: Point) -> bool:
        c = self.center
        return (point.x - c.x) ** 2 + (point.y - c.y) ** 2 < self.radius**2


class Path:
    def __init__(self, position: np.ndarray, start: Point = None, end: Point = None):
        """
        position: [x0, x1, x2, y0, y1, y2] like np.ndarray, optimizer.swarm.position
        start, end: Point, insert to position if given
        """
        assert len(position) % 2 == 0
        if start is not None and end is not None:  # insert start and end
            pl = int(position.shape[0] / 2)
            position = np.concatenate(
                ([start.x], position[:pl], [end.x], [start.y], position[pl:], [end.y])
            )
        self.position = position
        self.len = int(position.shape[0] / 2)  # 要转化成 int，否则切片会报错

    @property
    def pos_x(self) -> np.ndarray:
        return self.position[: self.len]

    @property
    def pos_y(self) -> np.ndarray:
        return self.position[self.len :]

    @cached_property
    def waypoints(self) -> np.ndarray:
        return np.stack((self.pos_x, self.pos_y), axis=1)

    @cached_property
    def smoothed_waypoints(self) -> np.ndarray:
        waypoints = self.waypoints

        # 沿行一阶差分，得到相邻路径点之间的欧氏距离
        # here prepend waypoints[0] before diff
        # to assure len(di_dis) == len(waypoints) and di_dis[0] == 0
        # to use waypoints[0:1] to obtain dimension
        di_vec = np.diff(waypoints, n=1, axis=0, prepend=waypoints[0:1])
        di_dis = np.linalg.norm(di_vec, axis=1)  # 相邻路径点之间的欧氏距离
        # 先累积距离，再除以总长度，变成0-1的索引
        di_index = np.cumsum(di_dis) / sum(di_dis)

        # make spline, and interpolate every unit distance
        spl = make_interp_spline(di_index, waypoints, bc_type="clamped")
        t = np.linspace(0, 1, int(sum(di_dis)))
        splined_routes = spl(t)
        return splined_routes

    @cached_property
    def length(self) -> float:
        waypoints = self.waypoints
        return np.sum(np.linalg.norm(waypoints[1:] - waypoints[:-1], axis=1))


def show(path: Path, obstacles: List[Obstacle] = []):
    _, ax = plt.subplots()

    # plot start and end
    s = Point(path.pos_x[0], path.pos_y[0])
    e = Point(path.pos_x[-1], path.pos_y[-1])
    ax.scatter(s[0], s[1], marker="x", label="start")
    ax.scatter(e[0], e[1], marker="x", label="end")

    # set axis limits
    ax.set_aspect("equal")
    ax.set_xlim(min(s[0], e[0]) - 5, max(s[0], e[0]) + 5)
    ax.set_ylim(min(s[1], e[1]) - 5, max(s[1], e[1]) + 5)

    # plot obstacle
    for obstacle in obstacles:
        circle = Circle(
            (obstacle.center.x, obstacle.center.y),
            obstacle.radius,
            color="black",
            alpha=0.5,
        )
        ax.add_patch(circle)

    # plot path
    ax.plot(path.smoothed_waypoints[:, 0], path.smoothed_waypoints[:, 1])
    ax.scatter(path.waypoints[:, 0], path.waypoints[:, 1], marker="x")

    plt.show()


def evaluate(positions: np.ndarray) -> np.ndarray:
    """positions: optimizer.swarm.position
    return: cost
    """

    def path_evaluate(xy) -> float:
        "xy: np.ndarray, shape = (n, 2)"
        # distance between points
        distance = np.linalg.norm(np.diff(xy, axis=0, prepend=xy[0:1]), axis=1)
        length = np.sum(distance)  # length of smoothed path

        # colision = np.zeros(xy.shape[0])
        # for obs in obstacles:  # Artificial Potential Field
        #     obs_distance = np.linalg.norm(xy - np.array(obs.center), axis=1)
        #     colision += 0.5 * np.clip(1 / obs_distance - 1 / obs.radius, 0, np.inf) ** 2
        # colision = np.sum(colision)

        # colision = 0
        # for obs in obstacles:  # Artificial Potential Field
        #     obs_distance = np.linalg.norm(xy - np.array(obs.center), axis=1)
        #     colision += np.sum(obs_distance <= obs.radius)

        colision = np.zeros(xy.shape[0])
        for obs in obstacles:  # distance in colision
            obs_distance = np.linalg.norm(xy - np.array(obs.center), axis=1)
            # path distance in colision turned to punishment
            colision[obs_distance <= obs.radius] += distance[obs_distance <= obs.radius]
        colision = colision.sum()

        # 计算距离的标准差
        return length + length * colision

    n_particles, wlen = positions.shape[0], int(positions.shape[1] / 2)

    # x, y position with start and end (n_particles, wlen+2)
    ones = np.ones((n_particles, 1))
    pos_x = np.concatenate((ones * start.x, positions[:, :wlen], ones * end.x), axis=1)
    pos_y = np.concatenate((ones * start.y, positions[:, wlen:], ones * end.y), axis=1)

    # calculate distance of current path (for interpolated path)
    x2 = np.diff(pos_x, n=1, axis=1, prepend=pos_x[:, 0:1])
    y2 = np.diff(pos_y, n=1, axis=1, prepend=pos_y[:, 0:1])
    distance_diff = np.sqrt(x2**2 + y2**2)  # (n_particles, wlen+2)
    distance = np.cumsum(distance_diff, axis=1)  # (n_particles, wlen+2)
    distance_norm = distance / distance[:, -1].reshape(-1, 1)  # normalize to 0-1

    # evaluate
    fitness = np.full(n_particles, np.inf)
    for i in range(n_particles):
        length = distance[i, -1]  # length of the path

        # here take normalized distance as index
        # moreover, there are risks of duplicated value, rising ValueError
        # so use np.unique to select
        idx, unique_idx = np.unique(distance_norm[i], return_index=True)
        # pos_x, pos_y are stacked to a matrix (wlen+2-?, 2)
        pos_xy = np.column_stack((pos_x[i][unique_idx], pos_y[i][unique_idx]))

        bspl = make_interp_spline(idx, pos_xy, bc_type="clamped")
        xy = bspl(np.linspace(0, 1, int(length)))  # get a point each unit length

        path_loss = path_evaluate(xy)
        waypoints_dev_loss = np.std(distance_diff[i])  # evenly distribute the points
        fitness[i] = path_loss + length * waypoints_dev_loss

    return fitness


# setup
if __name__ == "__main__":
    obstacles = [Obstacle(Point(3, 3), 2), Obstacle(Point(7, 5), 1)]
    waypoint_n = 10
    x_limit = (np.array([-20] * waypoint_n * 2), np.array([20] * waypoint_n * 2))
    start, end = Point(0, 0), Point(10, 10)

    # pso
    options = {"c1": 0.5, "c2": 0.3, "w": 0.9}
    optimizer = ps.single.GlobalBestPSO(
        n_particles=1000, dimensions=2 * waypoint_n, bounds=x_limit, options=options
    )
    cost, pos = optimizer.optimize(evaluate, iters=200)

    # show
    path = Path(pos, start, end)
    show(path, obstacles)

    plot_cost_history(cost_history=optimizer.cost_history)
    plt.show()
