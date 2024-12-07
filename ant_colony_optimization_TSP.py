# travelling salesman problem using ANT COLONY OPTIMIZATION

import numpy as np
import random
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim

# Function to calculate the Euclidean distance between two points
def Distance(P1, P2):
    if P1 == P2:
        return 0.0
    d = np.sqrt((P1[0] - P2[0])**2 + (P1[1] - P2[1])**2)
    return d

# Function to calculate the total distance of a sequence of cities
def TotalDistance(P, seq):
    dist = 0.0
    N = len(seq)
    for i in range(N - 1):
        dist += Distance(P[seq[i]], P[seq[i + 1]])
    dist += Distance(P[seq[N - 1]], P[seq[0]])  # Return to the starting city
    return dist

# Function to read city coordinates using Geopy (from "India_cities.txt")
def readCities(PNames):
    P = []
    geolocator = Nominatim(user_agent="TSP_AntColony")
    j = 0
    with open('India_cities.txt') as file:
        for line in file:
            city = line.strip()
            if city == "":
                break
            theLocation = city + ", India"
            pt = geolocator.geocode(theLocation, timeout=10)
            y = round(pt.latitude, 2)
            x = round(pt.longitude, 2)
            print(f"City[{j}] = {city} ({x:.2f}, {y:.2f})")
            P.append([x, y])
            PNames.append(city)
            j += 1
    return P

# Function to plot the best path and display the cities
def Plot(seq, P, dist, PNames):
    Pt = [P[seq[i]] for i in range(len(seq))]
    Pt.append(P[seq[0]])  # Return to the starting city
    Pt = np.array(Pt)
    plt.title('Total distance = ' + str(dist))
    plt.plot(Pt[:, 0], Pt[:, 1], '-o')

    for i in range(len(P)):
        plt.annotate(PNames[i], (P[i][0], P[i][1]))
    plt.show()

# Ant Colony Optimization Implementation
class AntColonyOptimization:
    def __init__(self, P, n_ants=20, n_iterations=100, alpha=1.0, beta=2.0, evaporation_rate=0.5, q=100):
        self.P = P
        self.N = len(P)
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha  # Influence of pheromone
        self.beta = beta    # Influence of heuristic (inverse distance)
        self.evaporation_rate = evaporation_rate
        self.q = q          # Constant for pheromone update
        self.pheromone = np.ones((self.N, self.N))
        self.distances = self._calculate_distances()

    def _calculate_distances(self):
        dist_matrix = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(self.N):
                if i != j:
                    dist_matrix[i][j] = Distance(self.P[i], self.P[j])
        return dist_matrix

    def _initialize_ants(self):
        return [random.randint(0, self.N - 1) for _ in range(self.n_ants)]

    def _update_pheromones(self, paths, distances):
        self.pheromone *= (1 - self.evaporation_rate)
        for path, dist in zip(paths, distances):
            for i in range(len(path) - 1):
                self.pheromone[path[i]][path[i + 1]] += self.q / dist
            self.pheromone[path[-1]][path[0]] += self.q / dist

    def _choose_next_city(self, current_city, visited):
        probabilities = []
        for i in range(self.N):
            if i not in visited:
                tau = self.pheromone[current_city][i] ** self.alpha
                eta = (1.0 / self.distances[current_city][i]) ** self.beta
                probabilities.append(tau * eta)
            else:
                probabilities.append(0)
        
        probabilities = np.array(probabilities)
        probabilities /= probabilities.sum()
        return np.random.choice(range(self.N), p=probabilities)

    def _construct_solution(self):
        paths = []
        distances = []
        for _ in range(self.n_ants):
            visited = []
            current_city = random.randint(0, self.N - 1)
            visited.append(current_city)
            
            while len(visited) < self.N:
                next_city = self._choose_next_city(current_city, visited)
                visited.append(next_city)
                current_city = next_city
            
            paths.append(visited)
            distances.append(TotalDistance(self.P, visited))
        
        return paths, distances

    def run(self):
        best_distance = float('inf')
        best_path = None
        
        for _ in range(self.n_iterations):
            paths, distances = self._construct_solution()
            self._update_pheromones(paths, distances)
            
            min_dist = min(distances)
            if min_dist < best_distance:
                best_distance = min_dist
                best_path = paths[distances.index(min_dist)]
        
        return best_path, best_distance

# Usage example
PNames = []
P = readCities(PNames)
aco = AntColonyOptimization(P, n_ants=20, n_iterations=100)
best_path, best_distance = aco.run()

print("Best Path:", best_path)
print("Best Distance:", best_distance)
Plot(best_path, P, best_distance, PNames)
