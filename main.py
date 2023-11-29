import numpy as np
import matplotlib.pyplot as plt

L = 10


def fitness_function(x, y):
    return np.sin(x) * np.cos(y) / (1 + x ** 2 + y ** 2)


def initialize_population(population_size):
    population = []
    for _ in range(population_size):
        x_binary = ''.join(np.random.choice(['0', '1']) for _ in range(L))
        y_binary = ''.join(np.random.choice(['0', '1']) for _ in range(L))

        x = int(x_binary, 2) / (2 ** L - 1)
        y = -2 + int(y_binary, 2) / (2 ** L - 1) * 4

        population.append((x, y))
    return population


def roulette_selection(population, fitness_values):
    total_fitness = sum(fitness_values)
    probabilities = [max(0, fitness / total_fitness) for fitness in fitness_values]
    probabilities /= np.sum(probabilities)
    selected_index = np.random.choice(len(population), p=probabilities)
    return population[selected_index]


def two_point_crossover(parent1, parent2):
    point1 = np.random.randint(1, L)
    point2 = np.random.randint(point1, L)

    x_binary_parent1 = bin(int((parent1[0] * (2 ** L - 1)) + 0.5))[2:].zfill(L)
    x_binary_parent2 = bin(int((parent2[0] * (2 ** L - 1)) + 0.5))[2:].zfill(L)
    y_binary_parent1 = bin(int(((parent1[1] + 2) / 4) * (2 ** L - 1) + 0.5))[2:].zfill(L)
    y_binary_parent2 = bin(int(((parent2[1] + 2) / 4) * (2 ** L - 1) + 0.5))[2:].zfill(L)

    x_child = int(x_binary_parent1[:point1] + x_binary_parent2[point1:point2] + x_binary_parent1[point2:], 2) / (2 ** L - 1)
    y_child = -2 + int(y_binary_parent1[:point1] + y_binary_parent2[point1:point2] + y_binary_parent1[point2:], 2) / (2 ** L - 1) * 4

    return x_child, y_child


def mutation(child, mutation_rate):
    x_binary = bin(int((child[0] * (2 ** L - 1)) + 0.5))[2:].zfill(L)
    y_binary = bin(int(((child[1] + 2) / 4) * (2 ** L - 1) + 0.5))[2:].zfill(L)

    if np.random.rand() < mutation_rate:
        mutated_bit_x = np.random.randint(L)
        x_binary = x_binary[:mutated_bit_x] + ('0' if x_binary[mutated_bit_x] == '1' else '1') + x_binary[
                                                                                                 mutated_bit_x + 1:]

    if np.random.rand() < mutation_rate:
        mutated_bit_y = np.random.randint(L)
        y_binary = y_binary[:mutated_bit_y] + ('0' if y_binary[mutated_bit_y] == '1' else '1') + y_binary[
                                                                                                 mutated_bit_y + 1:]

    child = (int(x_binary, 2) / (2 ** L - 1), -2 + int(y_binary, 2) / (2 ** L - 1) * 4)
    return child


def plot_3d_surface():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x_vals = np.linspace(0, 2, 100)
    y_vals = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = fitness_function(X, Y)

    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Fitness')

    plt.savefig('3D.png')


def plot_average_fitness(generation_values, average_fitness_values):
    plt.figure()
    plt.plot(generation_values[:len(average_fitness_values)], average_fitness_values, marker='o',
             label='Average Fitness')
    plt.title('Average Fitness for Generations')
    plt.xlabel('Generation')
    plt.ylabel('Average Fitness')
    plt.grid(True)
    plt.legend()
    plt.savefig('average - fitness.png')


def plot_population_values(generation, x_values, y_values):
    plt.figure()
    plt.scatter(x_values, y_values, marker='o', label='Population Values')

    additional_points_x = [0, 2]
    additional_points_y = [-2, 2]
    plt.scatter(additional_points_x, additional_points_y, color='red', marker='o', label='Critical Points')

    plt.title(f'Population Values for Generation {generation}')
    plt.xlabel('X values')
    plt.ylabel('Y values')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'population - values - {generation}.png')


def genetic_algorithm(population_size, generations, populations_to_display):
    population = initialize_population(population_size)
    best_overall_fitness = -np.inf
    best_overall_individual = None
    average_fitness_values = []

    for generation in range(generations):
        fitness_values = [fitness_function(x, y) for x, y in population]

        if generation in populations_to_display:
            print(f"\nX values for Population {generation}:")
            for i in range(min(10, population_size)):
                print(round(population[i][0], 4))

            print(f"\nY values for Population {generation}:")
            for i in range(min(10, population_size)):
                print(round(population[i][1], 4))

            print(f"\nFitness values for Population {generation}:")
            for i in range(min(10, population_size)):
                print(round(fitness_values[i], 4))

            average_fitness = np.mean(fitness_values)
            print(f"\nAverage Fitness for Population {generation}: {round(average_fitness, 4)}")
            print(f"Max Fitness for Population {generation}: {round(max(fitness_values), 4)}")

            average_fitness_values.append(average_fitness)

            x_values = [individual[0] for individual in population[:10]]
            y_values = [individual[1] for individual in population[:10]]
            plot_population_values(generation, x_values, y_values)

        if max(fitness_values) > best_overall_fitness:
            best_overall_fitness = max(fitness_values)
            best_overall_individual = population[np.argmax(fitness_values)]

        parents = [roulette_selection(population, fitness_values) for _ in range(population_size)]

        children = []
        for i in range(0, population_size, 2):
            parent1 = parents[i]
            parent2 = parents[i + 1]
            child1 = two_point_crossover(parent1, parent2)
            child2 = two_point_crossover(parent2, parent1)
            children.extend([mutation(child1, 0.25), mutation(child2, 0.25)])

        population = children

    print(
        f"\nOverall Best Solution: ({round(best_overall_individual[0], 4)}, {round(best_overall_individual[1], 4)}), Overall Best Fitness: {round(best_overall_fitness, 4)}")

    plot_average_fitness(range(generations), average_fitness_values)
    plot_3d_surface()


genetic_algorithm(population_size=4, generations=100, populations_to_display=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
