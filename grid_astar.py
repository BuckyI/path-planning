import heapq
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np


class Location(NamedTuple):
    x: float
    y: float


def heuristic(a: Location, b: Location) -> float:
    return np.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


class PriorityQueue:
    def __init__(self):
        self.elements: List[Tuple[float, Location]] = []

    def empty(self) -> bool:
        return len(self.elements) == 0

    def put(self, item: Location, priority: float) -> None:
        heapq.heappush(self.elements, (priority, item))

    def get(self) -> Location:
        return heapq.heappop(self.elements)[1]


class GridWithWeights:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.walls: List[Location] = []
        self.weights: Dict[Location, float] = {}  # weights between nodes, 1 by default

    def cost(self, from_pos: Location, to_pos: Location):
        return self.weights.get(to_pos, 1)

    def in_bounds(self, pos: Location):
        return 0 <= pos.x < self.width and 0 <= pos.y < self.height

    def passable(self, pos: Location):
        return pos not in self.walls

    def neighbors(self, pos: Location):
        (x, y) = pos
        results = [
            Location(x + 1, y),
            Location(x, y - 1),
            Location(x - 1, y),
            Location(x, y + 1),
        ]
        if (x + y) % 2 == 0:
            results.reverse()
        results = filter(self.in_bounds, results)
        results = filter(self.passable, results)
        return results


def a_star_search(grid: GridWithWeights, start: Location, goal: Location):
    # 做成一个类，可以重复搜索！保存 frontier 继续搜索，如果路径已经在 camefrom 中，说明已经找到路了
    frontier = PriorityQueue()
    frontier.put(start, 0)
    came_from: Dict[Location, Optional[Location]] = {}
    cost_so_far: Dict[Location, float] = {}
    came_from[start] = None
    cost_so_far[start] = 0

    while not frontier.empty():
        current = frontier.get()

        if current == goal:
            break

        for next in grid.neighbors(current):
            new_cost = cost_so_far[current] + grid.cost(current, next)
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost + heuristic(next, goal)
                frontier.put(next, priority)
                came_from[next] = current

    return came_from, cost_so_far


def reconstruct_path(
    came_from: Dict[Location, Optional[Location]], start: Location, goal: Location
) -> list[Location]:
    current = goal
    path = []

    while current != start:
        path.append(current)
        current = came_from.get(current, None)
        if current is None:
            raise ValueError("no path found")
    else:
        path.append(start)

    path.reverse()
    return path


def draw_grid(
    graph,
    *,
    start: Optional[Location] = None,
    goal: Optional[Location] = None,
    number: Dict[Location, float] = {},
    point_to: Dict[Location, Optional[Location]] = {},
    path: List[Location] = [],
):
    """
    visualize the grid
    Optional parameters to render
    start: start point
    goal: goal point
    number: cost of location
    point_to: came from location
    path: locations in path
    """

    def draw_tile(graph, pos: Location) -> str:
        # default
        r = " . " if pos not in graph.walls else "###"
        if pos in number:
            r = " %-2d" % number[pos]
        if (npos := point_to.get(pos, None)) is not None:
            if npos.x == pos.x + 1:
                r = " > "
            if npos.x == pos.x - 1:
                r = " < "
            if npos.y == pos.y + 1:
                r = " v "
            if npos.y == pos.y - 1:
                r = " ^ "
        if pos in path:
            r = " @ "
        if pos == start:
            r = " A "
        if pos == goal:
            r = " Z "
        return r

    print("___" * graph.width)
    for y in range(graph.height):
        for x in range(graph.width):
            print("%s" % draw_tile(graph, Location(x, y)), end="")
        print()
    print("~~~" * graph.width)


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

    start = Location(1, 1)
    goal = Location(28, 14)
    came_from, cost_so_far = a_star_search(diagram, start, goal)
    draw_grid(diagram, point_to=came_from, start=start, goal=goal)
    draw_grid(diagram, number=cost_so_far, start=start, goal=goal)

    path = reconstruct_path(came_from, start, goal)
    draw_grid(diagram, start=start, goal=goal, path=path)
