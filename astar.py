from typing import List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from grid_astar import AStar, GridWithWeights, Location


class Graph:
    "2-layer graph for multiple middle waypoints"

    def __init__(
        self,
        grid: GridWithWeights,
        start: Location,
        goal: Location,
        charging_points: List[Location],
    ):
        """
        grid: base grid map
        start: start point
        goal: end point
        charging_points: list of charging points, is needed for fueling
        """
        self.grid = grid

        assert all(
            grid.in_bounds(c) and grid.passable(c)
            for c in charging_points + [start, goal]
        )
        self.start = start
        self.goal = goal
        self.charging_points = charging_points

    def heuristic(self, pos: Location) -> float:
        "heuristic of distance from current pos to goal"
        goal = self.goal
        return np.sqrt((pos.x - goal.x) ** 2 + (pos.y - goal.y) ** 2)


def visualize(graph: Graph, path: List[Location] = []):
    map = graph.grid
    plt.figure(figsize=(map.width / 8, map.height / 8))

    boarder = Rectangle(
        (0, 0), map.width, map.height, edgecolor="black", facecolor="none"
    )
    plt.gca().add_patch(boarder)

    walls_x = [w.x for w in map.walls]
    walls_y = [w.y for w in map.walls]
    plt.scatter(walls_x, walls_y, marker="s", color="black")

    path_x = [p.x for p in path]
    path_y = [p.y for p in path]
    plt.plot(path_x, path_y, marker=".")

    style = {"marker": "*", "zorder": 2}
    plt.scatter(graph.start.x, graph.start.y, color="green", **style)
    plt.scatter(graph.goal.x, graph.goal.y, color="red", **style)
    for charging_point in graph.charging_points:
        plt.scatter(charging_point.x, charging_point.y, color="blue", **style)

    plt.show()


if __name__ == "__main__":
    diagram = GridWithWeights(30, 15)
    DIAGRAM1_WALLS: List[Location] = []
    for i in range(3, 5):
        for j in range(3, 12):
            DIAGRAM1_WALLS.append(Location(i, j))

    for i in range(13, 15):
        for j in range(4, 15):
            DIAGRAM1_WALLS.append(Location(i, j))

    for i in range(21, 23):
        for j in range(0, 7):
            DIAGRAM1_WALLS.append(Location(i, j))

    for i in range(23, 26):
        for j in range(5, 7):
            DIAGRAM1_WALLS.append(Location(i, j))

    diagram.walls = DIAGRAM1_WALLS

    graph = Graph(
        diagram,
        start=Location(1, 1),
        goal=Location(24, 3),
        charging_points=[Location(2, 6)],
    )

    a_star = AStar(diagram, start=Location(1, 1))

    goal = Location(24, 3)
    path = a_star.search(goal)
    visualize(graph, path)
