from typing import List

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from grid_astar import AStar, GridWithWeights, Location


def visualize(graph: GridWithWeights, path: List[Location] = []):
    plt.figure(figsize=(graph.width / 8, graph.height / 8))

    boarder = Rectangle(
        (0, 0), graph.width, graph.height, edgecolor="black", facecolor="none"
    )
    plt.gca().add_patch(boarder)

    walls_x = [w.x for w in graph.walls]
    walls_y = [w.y for w in graph.walls]
    plt.scatter(walls_x, walls_y, marker="s", color="black")

    path_x = [p.x for p in path]
    path_y = [p.y for p in path]
    plt.plot(path_x, path_y, marker=".")

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

    a_star = AStar(diagram, start=Location(1, 1))

    goal = Location(24, 3)
    path = a_star.search(goal)
    visualize(diagram, path)
