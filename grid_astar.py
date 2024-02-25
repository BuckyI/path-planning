import heapq
from collections import defaultdict, deque
from typing import Callable, Dict, List, NamedTuple, Optional, Protocol, Tuple

import numpy as np


class Location(NamedTuple):
    x: float
    y: float


class PriorityQueue:
    def __init__(self):
        self.elements: List[Tuple[float, Location]] = []

    def empty(self) -> bool:
        return len(self.elements) == 0

    def put(self, item: Location, priority: float) -> None:
        heapq.heappush(self.elements, (priority, item))

    def get(self) -> Location:
        return heapq.heappop(self.elements)[1]


def heuristic(a: Location, b: Location) -> float:
    return np.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


class Graph(Protocol):
    def cost(self, from_pos: Location, to_pos: Location) -> float: ...

    def neighbors(self, pos: Location) -> list[Location]: ...


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

    def neighbors(self, pos: Location) -> list[Location]:
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
        return list(results)


class AStar:
    "A star search, from start to multiple goals"

    def __init__(
        self,
        graph: Graph,
        start: Location,
        heuristic: Callable[[Location, Location], float] = heuristic,
    ):
        self.graph = graph
        self.heuristic = heuristic
        self.frontier = PriorityQueue()
        self.came_from: Dict[Location, Optional[Location]] = defaultdict(None)
        self.cost_so_far: Dict[Location, float] = {}

        # init start point
        self.start = start
        self.frontier.put(start, 0)
        self.cost_so_far[start] = 0

    def reconstruct_path(self, goal: Location) -> list[Location]:
        "get path from start to goal"
        path = deque()
        current = goal
        while current != self.start:
            path.appendleft(current)
            current = self.came_from.get(current, None)
            assert current is not None, "no valid path"

        path.appendleft(self.start)
        return list(path)

    def search(self, goal: Location) -> list[Location]:
        if goal in self.came_from:
            return self.reconstruct_path(goal)

        # resume search
        frontier = self.frontier
        graph = self.graph
        cost_so_far = self.cost_so_far
        came_from = self.came_from
        while not frontier.empty():
            current = frontier.get()

            if current == goal:
                # note item in frontier has been registered in came_from
                return self.reconstruct_path(goal)

            for next in graph.neighbors(current):
                new_cost = cost_so_far[current] + graph.cost(current, next)
                if next not in cost_so_far or new_cost < cost_so_far[next]:
                    cost_so_far[next] = new_cost
                    priority = new_cost + self.heuristic(next, goal)
                    came_from[next] = current
                    frontier.put(next, priority)

        print(f"no valid path from {self.start} to {goal}")
        return []  # no valid path


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

    a_star = AStar(diagram, start=Location(1, 1))

    goal = Location(24, 3)
    path = a_star.search(goal)
    draw_grid(diagram, point_to=a_star.came_from, start=a_star.start, goal=goal)
    draw_grid(diagram, number=a_star.cost_so_far, start=a_star.start, goal=goal)
    draw_grid(diagram, start=a_star.start, goal=goal, path=path)

    goal = Location(28, 14)
    path = a_star.search(goal)
    draw_grid(diagram, point_to=a_star.came_from, start=a_star.start, goal=goal)
    draw_grid(diagram, number=a_star.cost_so_far, start=a_star.start, goal=goal)
    draw_grid(diagram, start=a_star.start, goal=goal, path=path)

    goal = Location(35, 14)
    path = a_star.search(goal)
    draw_grid(diagram, number=a_star.cost_so_far, start=a_star.start, goal=goal)
