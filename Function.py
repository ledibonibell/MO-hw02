import random
import math
import time
import matplotlib.pyplot as plt

class Organism:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.fitness = 0.0


class Methods:
    def initialize_population(population_size, min_x, max_x, min_y, max_y):
        population = []
        seed = int(time.time() * 1000)
        random.seed(seed)
        for _ in range(population_size):
            x = random.uniform(min_x, max_x)
            y = random.uniform(min_y, max_y)
            population.append(Organism(x, y))
        return population

    def calculate_fitness(x, y):
        return (math.sin(x) * math.cos(y)) / (1 + x * x + y * y)

    def assess_population(population):
        for individual in population:
            fitness = (math.sin(individual.x) * math.cos(individual.y)) / (1 + individual.x * individual.x + individual.y * individual.y)
            individual.fitness = fitness
        return population

    def choose_parents(population):
        parents = []
        tournament_size = 3
        for _ in range(len(population)):
            best_parent = None
            best_fitness = -float('inf')
            for _ in range(tournament_size):
                random_index = random.randint(0, len(population) - 1)
                candidate = population[random_index]
                distance = math.sqrt((candidate.x - 0.5) ** 2 + (candidate.y + 0.000000005) ** 2)
                distance_weight = math.exp(-0.1 * distance)
                weighted_fitness = candidate.fitness * distance_weight
                if weighted_fitness > best_fitness:
                    best_parent = candidate
                    best_fitness = weighted_fitness
            parents.append(best_parent)
        return parents

    def apply_mutation(individual, mutation_rate, min_x, max_x, min_y, max_y):
        seed = int(time.time() * 1000)
        random.seed(seed)
        mutation_prob = random.uniform(0.0, 1.0)
        mutation_value = random.uniform(-mutation_rate, mutation_rate)
        x = individual.x
        y = individual.y
        if mutation_prob < mutation_rate:
            x += mutation_value
            x = max(min_x, min(x, max_x))
        if mutation_prob < mutation_rate:
            y += mutation_value
            y = max(min_y, min(y, max_y))
        mutated_individual = Organism(x, y)
        mutated_individual.fitness = (math.sin(x) * math.cos(y)) / (1 + x * x + y * y)
        return mutated_individual

    def perform_crossover(parent1, parent2):
        seed = int(time.time() * 1000)
        random.seed(seed)
        crossover_prob = random.uniform(0.0, 1.0)
        x = parent1.x if crossover_prob < 0.5 else parent2.x
        y = parent1.y if crossover_prob < 0.5 else parent2.y
        child = Organism(x, y)
        child.fitness = (math.sin(x) * math.cos(y)) / (1 + x * x + y * y)
        return child

    def check_population_convergence(population):
        fitness_sum = sum(p.fitness for p in population)
        average_fitness = fitness_sum / len(population)
        num_converged = sum(1 for p in population if abs(p.fitness - average_fitness) < 0.0001)
        convergence_ratio = num_converged / len(population)
        return convergence_ratio >= 0.7

    def display_results(population, generation):
        sum_fitness = 0
        iterations_to_display = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        if generation in iterations_to_display:
            print(f"Iteration {generation}")
            print("X:")
            for ind in population:
                print(f"{round(ind.x, 4)}")
            print("Y:")
            for ind in population:
                print(f"{round(ind.y, 4)}")
            print("F:")
            for ind in population:
                sum_fitness += ind.fitness
                print(f"{round(ind.fitness, 4)}")
            best_individual = max(population, key=lambda x: x.fitness)
            max_fitness = best_individual.fitness
            print(f"\nmax: {round(max_fitness, 4)}")
            print(f"Среднее: {round(sum_fitness / len(population), 4)}\n")

    def draw_fitness_graph(generation_list, average_fitness_list):
        plt.scatter(generation_list, average_fitness_list, marker='o')
        plt.title('Среднее FIT от x')
        plt.xlabel('Поколение')
        plt.ylabel('Среднее FIT')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig('Fitness.png')

    def draw_scatter_plots(population):
        plt.figure(figsize=(15, 10))
        plt.subplots_adjust(wspace=0.1, hspace=0.1)

        for i in range(10):
            x_values = [ind.x for ind in population]
            y_values = [ind.y for ind in population]
            plt.scatter(x_values, y_values, marker='o')
            plt.title(f'График рассеивания для поколения {i + 1}')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.savefig(f'Scatter - {i + 1}')
