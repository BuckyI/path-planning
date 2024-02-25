from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from grid_astar import AStar, GridWithWeights, Location, PriorityQueue, reconstruct_path


def euclidean_distance(from_pos: Location, to_pos: Location) -> float:
    return np.sqrt((from_pos.x - to_pos.x) ** 2 + (from_pos.y - to_pos.y) ** 2)


class LayerGraph:
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

        self.nodes = [start, goal] + charging_points
        assert all(grid.in_bounds(c) and grid.passable(c) for c in self.nodes)
        self.start = start
        self.goal = goal
        self.charging_points = charging_points

        # (from, to) -> path
        self.path: Dict[Tuple[Location, Location], List[Location]] = {}
        self.a_stars: Dict[Location, AStar] = {
            n: AStar(grid, n) for n in [start] + charging_points
        }

    def register_path(self, a: Location, b: Location, path: List[Location]):
        "find path between a and b, empty list if no path"
        # here we assume (a, b) and (b, a) have the same path
        # i.e. this is undirected graph
        self.path[(a, b)] = path
        self.path[(b, a)] = path

    def cost(self, from_pos: Location, to_pos: Location) -> float:
        """
        cost of path from from_pos to to_pos
        """
        # find path if not registered before
        if (from_pos, to_pos) not in self.path:
            path = self.a_stars[from_pos].search(to_pos)
            self.register_path(from_pos, to_pos, path)

        path = self.path[(from_pos, to_pos)]
        # path with no waypoints is considered unreachable
        if not path:
            return float("inf")

        # Euclidean distance
        return np.linalg.norm(np.diff(np.array(path), axis=0), axis=1).sum()

    def neighbors(self, pos: Location, fuel: float) -> List[Location]:
        "neighbors within fuel capability"
        # here distance should >0 to exclude pos self
        within_reach = lambda p1, p2: 0 < euclidean_distance(p1, p2) <= fuel
        # Note: goal will be the first element if in neighbors
        return [
            n
            for n in [self.goal, self.start] + self.charging_points
            if within_reach(pos, n)
        ]


def a_star_search_with_fuel(graph: LayerGraph, fuel: float, charge_fuel: float = 100):
    """
    perform a star search in graph from start to goal
    but with fuel as max path length limit
    fuel: init fuel level, decrease in path, but refuel in charging points
    charge_fuel: amount of fuel to refuel after reaching a charging point
    """

    def _reconstruct_path(current: Location) -> List[Location]:
        "get path from start to current location"
        path = reconstruct_path(came_from, graph.start, current)
        whole_path = []
        for p1, p2 in zip(path[:-1], path[1:]):
            whole_path += graph.path[(p1, p2)]
        return whole_path

    frontier = PriorityQueue()
    cost_so_far: Dict[Location, float] = {}
    came_from: Dict[Location, Optional[Location]] = defaultdict(None)

    start = graph.start
    frontier.put(start, 0)
    cost_so_far[start] = 0

    while not frontier.empty():
        current = frontier.get()

        # update fuel
        fuel_times = 0
        path = _reconstruct_path(current)
        for c in graph.charging_points:
            if c in path:
                fuel_times += 1
        current_fuel = fuel + fuel_times * charge_fuel
        if current == graph.goal:
            break

        for n in graph.neighbors(current, current_fuel):
            new_cost = cost_so_far[current] + graph.cost(current, n)
            if new_cost > current_fuel:  # not reachable, skip
                print(f"not reachable {current} -> {n}")
                continue
            if n not in cost_so_far or new_cost < cost_so_far[n]:
                cost_so_far[n] = new_cost
                came_from[n] = current
                # here we may return if n == graph.goal to save time?
                priority = new_cost + euclidean_distance(n, graph.goal)
                frontier.put(n, priority)

    if current != graph.goal:
        print("no valid path found")
        return []
    return _reconstruct_path(graph.goal)


def visualize(graph: LayerGraph, path: List[Location] = []):
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

    graph = LayerGraph(
        diagram,
        start=Location(1, 1),
        goal=Location(24, 3),
        charging_points=[
            Location(2, 6),
            Location(6, 5),
            Location(20, 5),
            Location(12, 2),
            Location(25, 10),
        ],
    )

    path = a_star_search_with_fuel(graph, 10, 10)
    if path:
        print("length", np.linalg.norm(np.diff(np.array(path), axis=0), axis=1).sum())
    visualize(graph, path)
