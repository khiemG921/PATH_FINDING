import time
import tracemalloc
from queue import PriorityQueue

NO_PATH = None

algo = ['BFS', 'DFS', 'UCS', 'IDS', 'GBFS', 'A*', 'HC']

algo_name = ['Breadth-first search', 'Tree-search Depth-first search', 'Uniform-cost search',
             'Iterative deepening search', 'Greedy best-first search', 'Graph-search A*', 'Hill-climbing']

class Node:
    def __init__(self, index):
        self.index = index
        self.heuristic = 0

class Graph:
    def __init__(self):
        self.nodes = {}
        self.adjacency_matrix = []
    
    def get_distance(self, from_index, to_index):
        if from_index not in self.nodes or to_index not in self.nodes:
            raise ValueError("Both nodes must exist in the graph.")
        
        return self.adjacency_matrix[from_index][to_index]
    
    def load_from_file(self, filename):
        with open(filename, 'r') as file:
            lines = file.readlines()
        
        num_nodes = int(lines[0].strip())
        start_end = list(map(int, lines[1].strip().split()))
        start_index = start_end[0]
        end_index = start_end[1]
        
        self.adjacency_matrix = []
        for i in range(2, 2 + num_nodes):
            row = list(map(int, lines[i].strip().split()))
            self.adjacency_matrix.append(row)
        
        heuristic_values = list(map(int, lines[2 + num_nodes].strip().split()))
        for index in range(num_nodes):
            node = Node(index)
            node.heuristic = heuristic_values[index]
            self.nodes[node.index] = node
        
        return start_index, end_index
    
    @staticmethod
    def write_result_to_file(algorithm_name, path, start_time, memory_start, memory_end, file_name='output.txt'):
        end_time = time.time()
        elapsed_time = end_time - start_time
        memory_used = (memory_end - memory_start) / 1024  # in KB

        with open(file_name, 'a') as file:
            file.write(f"{algo_name[algo.index(algorithm_name)]}\n")
            file.write(f"Path: {' -> '.join(map(str, path)) if path else '-1'}\n")
            file.write(f"Time: {elapsed_time:.8f} seconds\n")
            file.write(f"Memory: {memory_used:.2f} KB\n\n")
    
    def bfs(self, start_index, end_index):
        if start_index not in self.nodes or end_index not in self.nodes:
            raise ValueError("Both nodes must exist in the graph.")
        
        visited = [False] * len(self.nodes)
        queue = []
        path = []

        queue.append([start_index])
        visited[start_index] = True

        while queue:
            current_path = queue.pop(0)
            current_node = current_path[-1]

            if current_node == end_index:
                return current_path
            
            for i in range(len(self.adjacency_matrix[current_node])):
                if self.adjacency_matrix[current_node][i] > 0 and not visited[i]:
                    new_path = list(current_path)
                    new_path.append(i)
                    queue.append(new_path)
                    visited[i] = True
        
        return NO_PATH
    
    def dfs(self, start_index, end_index):
        if start_index not in self.nodes or end_index not in self.nodes:
            raise ValueError("Both nodes must exist in the graph.")
        
        visited = [False] * len(self.nodes)
        stack = []
        
        stack.append([start_index])
        
        while stack:
            current_path = stack.pop()
            current_node = current_path[-1]
            
            if not visited[current_node]:
                visited[current_node] = True

                if current_node == end_index:
                    return current_path
                
                for i in range(len(self.adjacency_matrix[current_node]) - 1, -1, -1):
                    if self.adjacency_matrix[current_node][i] > 0 and not visited[i]:
                        new_path = list(current_path)
                        new_path.append(i)
                        stack.append(new_path)
        
        return NO_PATH
    
    def ucs(self, start_index, end_index):
        def put_with_replace(pq, item):
            if pq.empty():
                pq.put(item)
                return

            temp_items = []
            replaced = False

            while not pq.empty():
                current_item = pq.get()
                if current_item[0] > item[0] and current_item[1][0] == item[1][0] and current_item[1][-1] == item[1][-1]:
                    temp_items.append(item)
                    replaced = True
                else:
                    temp_items.append(current_item)

            if not replaced:
                temp_items.append(item)
            
            for temp_item in temp_items:
                pq.put(temp_item)

        if start_index not in self.nodes or end_index not in self.nodes:
            raise ValueError("Both nodes must exist in the graph.")
        
        visited = [False] * len(self.nodes)
        pq = PriorityQueue()
        
        pq.put((0, [start_index]))
        
        while not pq.empty():
            current_cost, current_path = pq.get()
            current_node = current_path[-1]
            
            if visited[current_node]:
                continue
            
            visited[current_node] = True
            
            if current_node == end_index:
                return current_path
            
            for i in range(len(self.adjacency_matrix[current_node])):
                if self.adjacency_matrix[current_node][i] > 0 and not visited[i]:
                    new_path = list(current_path)
                    new_path.append(i)
                    new_cost = current_cost + self.adjacency_matrix[current_node][i]
                    put_with_replace(pq, (new_cost, new_path))
        
        return NO_PATH
    
    def depth_limited_search(self, start_index, end_index, limit):
        def dls(current_node, path, depth):
            if current_node == end_index:
                return path
            
            if depth >= limit:
                return NO_PATH
            
            for i in range(len(self.adjacency_matrix[current_node])):
                if self.adjacency_matrix[current_node][i] > 0 and i not in path:
                    new_path = list(path)
                    new_path.append(i)
                    result = dls(i, new_path, depth + 1)
                    if result is not NO_PATH:
                        return result
            return NO_PATH

        return dls(start_index, [start_index], 0)
    
    def ids(self, start_index, end_index):
        depth = 0
        while True:
            result = self.depth_limited_search(start_index, end_index, depth)
            if result is not NO_PATH:
                return result
            depth += 1
    
    def gbfs(self, start_index, end_index):
        if start_index not in self.nodes or end_index not in self.nodes:
            raise ValueError("Both nodes must exist in the graph.")
        
        visited = [False] * len(self.nodes)
        pq = PriorityQueue()
        
        pq.put((self.nodes[start_index].heuristic, [start_index]))
        
        while not pq.empty():
            current_priority, current_path = pq.get()
            current_node = current_path[-1]
            
            if visited[current_node]:
                continue
            
            visited[current_node] = True
            
            if current_node == end_index:
                return current_path
            
            for i in range(len(self.adjacency_matrix[current_node])):
                if self.adjacency_matrix[current_node][i] > 0 and not visited[i]:
                    new_path = list(current_path)
                    new_path.append(i)
                    pq.put((self.nodes[i].heuristic, new_path))
        
        return NO_PATH
    
    def gsas(self, start_index, end_index):
        if start_index not in self.nodes or end_index not in self.nodes:
            raise ValueError("Both nodes must exist in the graph.")
        
        visited = set()
        pq = PriorityQueue()
        pq.put((0, [start_index], 0))
        
        while not pq.empty():
            current_f_cost, current_path, current_g_cost = pq.get()
            current_node = current_path[-1]
            
            if current_node in visited:
                continue
            
            visited.add(current_node)
            
            if current_node == end_index:
                return current_path
            
            for i in range(len(self.adjacency_matrix[current_node])):
                if self.adjacency_matrix[current_node][i] > 0 and i not in visited:
                    new_path = list(current_path)
                    new_path.append(i)
                    new_g_cost = current_g_cost + self.adjacency_matrix[current_node][i]
                    f_cost = new_g_cost + self.nodes[i].heuristic
                    pq.put((f_cost, new_path, new_g_cost))
        
        return NO_PATH
    
    def hc(self, start_index, end_index):
        if start_index not in self.nodes or end_index not in self.nodes:
            raise ValueError("Both nodes must exist in the graph.")
        
        current_node = start_index
        path = [current_node]
        
        while current_node != end_index:
            neighbors = []
            for i in range(len(self.adjacency_matrix[current_node])):
                if self.adjacency_matrix[current_node][i] > 0:
                    neighbors.append((self.nodes[i].heuristic, i))
            
            if not neighbors:
                return NO_PATH
            
            next_node = min(neighbors, key=lambda x: x[0])[1]
            
            if self.nodes[next_node].heuristic >= self.nodes[current_node].heuristic:
                return NO_PATH 
            
            path.append(next_node)
            current_node = next_node
        
        return path
    
    def __str__(self):
        result = "Adjacency Matrix:\n"
        for row in self.adjacency_matrix:
            result += " ".join(map(str, row)) + "\n"
        return result

    def run_algorithm(self, num_input, algorithm_name, start_index, end_index):
        algorithm_map = {
            'BFS' : self.bfs,
            'DFS' : self.dfs,
            'UCS' : self.ucs,
            'IDS' : self.ids,
            'GBFS': self.gbfs,
            'A*'  : self.gsas,
            'HC'  : self.hc
        }
        
        if algorithm_name not in algorithm_map:
            raise ValueError(f"Algorithm {algorithm_name} is not supported.")
        
        algorithm = algorithm_map[algorithm_name]

        tracemalloc.start()
        start_time = time.time()
        memory_start = tracemalloc.get_traced_memory()[0]
        
        path = algorithm(start_index, end_index)
        
        memory_end = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        
        self.write_result_to_file(algorithm_name, path, start_time, memory_start, memory_end, f"./data/output/output{num_input}.txt")


if __name__ == '__main__':
    g = Graph()

    for i in range(1,8):
        num_input = i
        start_index, end_index = g.load_from_file(f'./data/input/input{num_input}.txt')
        print(g)

        for a in algo:
            g.run_algorithm(num_input, a, start_index, end_index)