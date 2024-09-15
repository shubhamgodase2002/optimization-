# travelling salesman problem using Genetic Algorithm

#from scipy import sqrt
from numpy import array
import random
from geopy.geocoders import Nominatim
import matplotlib.pyplot as plt
import numpy as np


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

# Function to read city coordinates using Geopy (from "cities_india.txt")
def readCities(PNames):
    P = []
    geolocator = Nominatim(user_agent="FomsApp")
    j = 0
    with open('India_cities.txt') as file:
        for line in file:
            city = line.rstrip('\n')
            if city == "":
                break
            theLocation = city + ", India"
            pt = geolocator.geocode(theLocation, timeout=10000)
            y = round(pt.latitude, 2)
            x = round(pt.longitude, 2)
            print("City[%2d] = %s (%5.2f, %5.2f)" % (j, city, x, y))
            P.insert(j, [x, y])
            PNames.insert(j, city)
            j += 1
    return P

# Function to plot the best path and display the cities
def Plot(seq, P, dist, PNames):
    Pt = [P[seq[i]] for i in range(len(seq))]
    Pt += [P[seq[0]]]  # Return to the starting city
    Pt = array(Pt)
    plt.title('Total distance = ' + str(dist))
    plt.plot(Pt[:, 0], Pt[:, 1], '-o')

    for i in range(len(P)):
        plt.annotate(PNames[i], (P[i][0], P[i][1]))
    plt.show()

# Genetic algorithm functions
def generate_random_paths(total_destinations):
    random_paths = []
    for _ in range(1000):  # Large initial population
        random_path = list(range(1, total_destinations))
        random.shuffle(random_path)
        random_path = [0] + random_path  # Ensure path starts from city 0
        random_paths.append(random_path)
    return random_paths

def choose_survivors(points, old_generation):
    survivors = []
    random.shuffle(old_generation)
    midway = len(old_generation) // 2
    for i in range(midway):
        if TotalDistance(points, old_generation[i]) < TotalDistance(points, old_generation[i + midway]):
            survivors.append(old_generation[i])
        else:
            survivors.append(old_generation[i + midway])
    return survivors

def create_offspring(parent_a, parent_b):
    offspring = []
    start = random.randint(0, len(parent_a) - 1)
    finish = random.randint(start, len(parent_a))
    sub_path_from_a = parent_a[start:finish]
    remaining_path_from_b = [item for item in parent_b if item not in sub_path_from_a]
    for i in range(len(parent_a)):
        if start <= i < finish:
            offspring.append(sub_path_from_a.pop(0))
        else:
            offspring.append(remaining_path_from_b.pop(0))
    return offspring


def apply_crossovers(survivors):
    offsprings = []
    midway = len(survivors) // 2
    for i in range(midway):
        parent_a, parent_b = survivors[i], survivors[i + midway]
        for _ in range(2):
            offsprings.append(create_offspring(parent_a, parent_b))
            offsprings.append(create_offspring(parent_b, parent_a))
    return offsprings

def apply_mutations(generation):
    gen_wt_mutations = []
    for path in generation:
        if random.randint(0, 1000) <40 :  # 0.04% chance of mutation
            index1, index2 = random.randint(1, len(path) - 1), random.randint(1, len(path) - 1)
            path[index1], path[index2] = path[index2], path[index1]
        gen_wt_mutations.append(path)
    return gen_wt_mutations

def generate_new_population(points, old_generation):
    survivors = choose_survivors(points, old_generation)
    crossovers = apply_crossovers(survivors)
    new_population = apply_mutations(crossovers)
    return new_population

def choose_best(points, paths, count):
    return sorted(paths, key=lambda path: TotalDistance(points, path))[:count]

# Main genetic algorithm function
def genetic_algorithm(P, PNames, generations=100):
    population = generate_random_paths(len(P))  # Initial population

    for generation in range(generations):
        population = generate_new_population(P, population)  # Evolve population
        best = choose_best(P, population, 1)[0]  # Get the best path in the generation
        best_distance = TotalDistance(P, best)
        print(f"Generation {generation + 1}: Best distance = {best_distance}")

    # Return the best path and its distance
    best_path = choose_best(P, population, 1)[0]
    best_distance = TotalDistance(P, best_path)
    Plot(best_path, P, best_distance, PNames)  # Plot the best path
    return best_path, best_distance

# Example usage
PNames = []  # List to hold city names
P = readCities(PNames)  # Read city data

# Run the genetic algorithm
best_path, best_distance = genetic_algorithm(P, PNames)
print(f"Best path: {best_path}")
print(f"Best distance: {best_distance}")
