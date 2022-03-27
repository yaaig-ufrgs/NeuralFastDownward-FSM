#!/usr/bin/env python3

"""
Print the samples in grid format
(specific for grid 4x4)

Usage: ./print_visitall_samples.py sample_file
"""

from sys import argv

def print_tiles(robot, tiles):
    for i in range(4):
        for j in range(4):
            idx = i*4+j
            print("x" if tiles[idx] else "_", end="")
            print("'" if robot[idx] else " ", end=" ")
        print()
    input()

def check(robot, tiles):
    robot_tile_visited = tiles[robot.index(True)] if robot.count(True) > 0 else False

    def neighbours(idx):
        nb = []
        if idx > 3:    nb.append(idx-4) # up
        if idx < 12:   nb.append(idx+4) # down
        if idx%4 != 0: nb.append(idx-1) # left
        if idx%4 != 3: nb.append(idx+1) # right
        return nb

    graph = {}
    for t in range(16):
        graph[t] = []
    for t in range(16):
        for n in neighbours(t):
            graph[t].append(n)

    visited = []
    queue = []
    def bfs(visited, graph, tiles, node):
        visited.append(node)
        queue.append(node)
        while queue:
            m = queue.pop(0)
            for neighbour in graph[m]:
                if neighbour not in visited and tiles[neighbour]:
                    visited.append(neighbour)
                    queue.append(neighbour)

    bfs(visited, graph, tiles, tiles.index(True))
    assert len(visited) <= tiles.count(True)
    valid_path = len(visited) == tiles.count(True)

    return robot_tile_visited, valid_path

for file in argv[1:]:
    with open(file, "r") as f:
        samples = [x.strip().split(";") for x in f.readlines() if x and x[0] != "#"]

    robot_tile_not_visited_counter = 0
    invalid_path_counter = 0
    for h, s in samples:
        print(f"h = {h} | s = {s}")
        s = s.replace("*", "0")
        robot = [x == "1" for x in s[:16]]
        # assert robot.count(True) == 1

        tiles = [x == "1" for x in s[16:]]
        tiles.insert(5, True) # task specific: grid 4x4 and robot initially on tile x
        tiles.reverse()

        robot_tile_visited, valid_path = check(robot, tiles)
        print("robot tile visited:", robot_tile_visited)
        print("valid path:", valid_path)
        print_tiles(robot, tiles)
