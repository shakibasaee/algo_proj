import random
from string import ascii_uppercase
import copy

number_of_places = int(input("enter number of places that you want:"))
total_time = int(input("enter the time that you want to end the tour:"))


class Place:
    def __init__(self, name: str, visit_time: int):
        self.name = name
        self.visit_time = visit_time


places = []


for i in range(number_of_places):
    name = ascii_uppercase[i]
    visit_time = random.randint(30, 100)
    place = Place(name, visit_time)
    places.append(place)


def total_visit_time(places):
    return sum(map(lambda p: p.visit_time, places))


for place in places:
    print(f"Place: {place.name}, visit time:{place.visit_time} minuts")


class Map:
    def __init__(self, places):
        self.places = places
        self.n = len(places)
        self.matrix = [[0 for _ in range(self.n)] for _ in range(self.n)]

        for i in range(self.n):
            for j in range(i, self.n):
                distance = random.randint(10, 50)

                if i == j:
                    distance = 0
                else:
                    self.matrix[i][j] = distance
                    self.matrix[j][i] = distance

    def floyd_warshall(self):
        distance = copy.deepcopy(self.matrix)
        n = self.n
        for k in range(n):  # K: گره واسط
            for i in range(n):
                for j in range(n):
                    if distance[i][j] > distance[i][k] + distance[k][j]:
                        distance[i][j] = distance[i][k] + distance[k][j]
        return distance

    def total_distance(self, shortest_paths):
        total = 0
        for i in range(self.n - 1):
            total += shortest_paths[i][i + 1]
        return total


# test 2
def print_distance_matrix(matrix, places):
    print("    " + "   ".join(place.name for place in places))
    for i in range(len(places)):
        row = [f"{matrix[i][j]:3}" for j in range(len(places))]
        print(f"{places[i].name}  " + " ".join(row))


my_map = Map(places)
print("\nDistance Matrix:")
print_distance_matrix(my_map.matrix, places)

# test 3
shortest_paths = my_map.floyd_warshall()
print("\nShortest Path Matrix (Floyd-Warshall):")
print_distance_matrix(shortest_paths, places)


def total_visit_time(places):
    return sum(map(lambda p: p.visit_time, places))


shortest_paths = my_map.floyd_warshall()
time_used = my_map.total_distance(shortest_paths) + total_visit_time(places)
if total_time <= time_used:
    print("unfortunately you can not visit all the places in the city:(")
else:
    print("congragulation you can visit all the places in time ^^")
print("\ntotal time:", time_used)


# test 4
# print(vars(map))
