# Graph Algorithms

This program implements various graph search algorithms, including Breadth-First Search (BFS), Depth-First Search (DFS), Uniform-Cost Search (UCS), Iterative Deepening Search (IDS), Greedy Best-First Search (GBFS), A* Search, and Hill Climbing (HC). The algorithms are designed to find paths in a graph defined by an adjacency matrix and heuristic values.

## Table of Contents

- [Graph Algorithms](#graph-algorithms)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Input Format](#input-format)
  - [Output Format](#output-format)
  - [Algorithms](#algorithms)
  - [Performance Metrics](#performance-metrics)
  - [Contributing](#contributing)
    - [Instructions for Use](#instructions-for-use)

## Features

- Implements multiple graph search algorithms.
- Loads graph data from text files.
- Measures execution time and memory usage for each algorithm.
- Outputs results to a specified file.


## Input Format

Each input file should follow this format:

1. The first line contains an integer representing the number of nodes in the graph.
2. The second line contains two integers representing the start and end node indices.
3. The following lines represent the adjacency matrix (one row per line).
4. The final line contains the heuristic values for each node.

**Example Input File (`input1.txt`):**

```
5
0 4
0 1 4 0 0
1 0 0 2 0
4 0 0 3 0
0 0 0 0 0
0 0 0 0 0
1 1 1 1 0
```

## Output Format

The output will be written to a specified file and includes:

- The name of the algorithm used.
- The path found (or -1 if no path exists).
- The time taken for the algorithm to execute.
- The memory used during execution.

**Example Output File (`output1.txt`):**

```
Breadth-first search
Path: 0 -> 1 -> 4
Time: 0.00234567 seconds
Memory: 15.20 KB
```

## Algorithms

The following algorithms are implemented:

1. **Breadth-First Search (BFS)**: Explores all neighbors at the present depth prior to moving on to nodes at the next depth level.
2. **Depth-First Search (DFS)**: Explores as far down a branch as possible before backtracking.
3. **Uniform-Cost Search (UCS)**: Expands the least costly node first.
4. **Iterative Deepening Search (IDS)**: Combines the benefits of BFS and DFS by iteratively deepening the search.
5. **Greedy Best-First Search (GBFS)**: Expands nodes based on their heuristic values.
6. **A* Search**: A combination of UCS and GBFS that considers both the path cost and heuristic.
7. **Hill Climbing (HC)**: A local search algorithm that continuously moves toward the goal state.

## Performance Metrics

The program measures:

- Execution time (in seconds).
- Memory usage (in KB).

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any improvements or features.

### Instructions for Use

1. Replace `<repository-url>` with the URL of your repository.
2. Ensure the example input matches the expected format for your specific implementation.
3. You can modify sections based on your specific needs or any additional features you might add.

Feel free to ask if you need further modifications or additional sections!