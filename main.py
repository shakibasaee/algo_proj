import random
import copy
from string import ascii_uppercase
import networkx as nx
import matplotlib.pyplot as plt


class Place:
    def __init__(self, name: str, visit_time: int):
        self.name = name
        self.visit_time = visit_time

    def __str__(self):
        return f"Place: {self.name}, visit time: {self.visit_time} minutes"


class TourMap:
    def __init__(self, places: list[Place]):
        self.places = places
        self.n = len(places)
        self.matrix = self._generate_distance_matrix()

    def _generate_distance_matrix(self):
        matrix = [[0 for _ in range(self.n)] for _ in range(self.n)]
        for i in range(self.n):
            for j in range(i + 1, self.n):
                distance = random.randint(10, 50)
                matrix[i][j] = distance
                matrix[j][i] = distance
        return matrix

    def floyd_warshall(self):
        distance = copy.deepcopy(self.matrix)
        for k in range(self.n):
            for i in range(self.n):
                for j in range(self.n):
                    if distance[i][j] > distance[i][k] + distance[k][j]:
                        distance[i][j] = distance[i][k] + distance[k][j]
        return distance

    def total_distance(self, shortest_paths):
        total = 0
        for i in range(self.n - 1):
            total += shortest_paths[i][i + 1]
        return total

    def print_distance_matrix(self, matrix):
        print("    " + "   ".join(place.name for place in self.places))
        for i in range(self.n):
            row = [f"{matrix[i][j]:3}" for j in range(self.n)]
            print(f"{self.places[i].name}  " + " ".join(row))

    
    def draw_graph(self, matrix=None, best_path=None, title="Graph"):
        import networkx as nx
        import matplotlib.pyplot as plt

        graph = nx.Graph()

        if best_path:
            # Only add nodes and edges involved in the best path
            for i in range(len(best_path)):
                graph.add_node(self.places[best_path[i]].name)
            for i in range(len(best_path) - 1):
                u = best_path[i]
                v = best_path[i + 1]
                name_u = self.places[u].name
                name_v = self.places[v].name
                weight = self.matrix[u][v] if matrix is None else matrix[u][v]
                graph.add_edge(name_u, name_v, weight=weight)
        else:
            # Draw full graph (all nodes and edges)
            used_matrix = matrix if matrix is not None else self.matrix
            for place in self.places:
                graph.add_node(place.name)
            for i in range(self.n):
                for j in range(i + 1, self.n):
                    weight = used_matrix[i][j]
                    if weight > 0:
                        name_x = self.places[i].name
                        name_y = self.places[j].name
                        graph.add_edge(name_x, name_y, weight=weight)

        # Draw the graph
        pos = nx.spring_layout(graph, seed=42)
        edge_labels = nx.get_edge_attributes(graph, 'weight')

        plt.figure(figsize=(8, 6))
        nx.draw(
            graph, pos, with_labels=True,
            node_color="lightblue" if best_path else "lightgreen",
            node_size=1000, edge_color="black", linewidths=2, font_size=12
        )
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)

        plt.title(title)
        plt.show()
        

class DFSPathFinder:
    def __init__(self, shortest_paths, visit_times, total_allowed_time):
        self.shortest_paths = shortest_paths
        self.visit_times = visit_times
        self.total_allowed_time = total_allowed_time

        self.best_path = []
        self.max_places = 0
        self.min_total_time = float("inf")

    def search(self, start_node=0):
        visited = set([start_node])
        path = [start_node]
        time_spent = self.visit_times[start_node]

        self._dfs(
            current_node=start_node, visited=visited, time_spent=time_spent, path=path
        )

        return self.best_path, self.min_total_time

    def _dfs(self, current_node, visited, time_spent, path):
        if len(path) > self.max_places or (
            len(path) == self.max_places and time_spent < self.min_total_time
        ):
            self.best_path = path[:]
            self.max_places = len(path)
            self.min_total_time = time_spent

        for next_node in range(len(self.shortest_paths)):
            if next_node not in visited:
                travel_time = self.shortest_paths[current_node][next_node]
                visit_time = self.visit_times[next_node]
                total_time = time_spent + travel_time + visit_time

                if total_time <= self.total_allowed_time:
                    visited.add(next_node)
                    path.append(next_node)

                    self._dfs(next_node, visited, total_time, path)

                    visited.remove(next_node)
                    path.pop()


class TourPlanner:
    def __init__(self, number_of_places: int, total_time: int):
        self.number_of_places = number_of_places
        self.total_time = total_time
        self.places = self._generate_places()
        self.tour_map = TourMap(self.places)

    def _generate_places(self):
        places = []
        for i in range(self.number_of_places):
            name = ascii_uppercase[i]
            visit_time = random.randint(30, 100)
            places.append(Place(name, visit_time))
        return places

    def total_visit_time(self):
        return sum(p.visit_time for p in self.places)

    def plan(self):
        print("Places:")
        for place in self.places:
            print(place)

        print("\nDistance Matrix:")
        self.tour_map.print_distance_matrix(self.tour_map.matrix)

        shortest_paths = self.tour_map.floyd_warshall()
        print("\nShortest Path Matrix (Floyd-Warshall):")
        self.tour_map.print_distance_matrix(shortest_paths)

        time_used = (
            self.tour_map.total_distance(shortest_paths) + self.total_visit_time()
        )
        print("\nTotal time needed:", time_used)

        if self.total_time < time_used:
            print("Unfortunately, you cannot visit all the places in time :(")
        else:
            print("Congratulations! You can visit all the places in time ^^")

        return shortest_paths




if __name__ == "__main__":
    number_of_places = int(input("Enter number of places you want to visit: "))
    total_time = int(input("Enter the total time you have for the tour: "))

    planner = TourPlanner(number_of_places, total_time)
    shortest_paths = planner.plan()

    visit_times = [place.visit_time for place in planner.places]
    start_node = visit_times.index(min(visit_times))
    
    dfs_finder = DFSPathFinder(shortest_paths, visit_times, total_time)
    best_path, min_time = dfs_finder.search(start_node=start_node)


#draw th graph
    planner.tour_map.draw_graph(planner.tour_map.matrix, title="init tour") 
    path_matrix = [[0 for _ in range(number_of_places)] for _ in range (number_of_places)]
    
    #best grah to see
    planner.tour_map.draw_graph(best_path=best_path, title="best dfs path")
    
    print("\nBest path:", [planner.places[i].name for i in best_path])
    print("Time used:", min_time)