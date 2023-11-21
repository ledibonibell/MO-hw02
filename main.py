import matplotlib.pyplot as plt

from Function import Methods

population_size = 4
num_generations = 100
mutation_rate = 0.25
min_x = 0.0
max_x = 2.0
min_y = -2.0
max_y = 2.0

population = Methods.initialize_population(population_size, min_x, max_x, min_y, max_y)
initial_mutation_rate = mutation_rate

generation_list = []
average_fitness_list = []

for generation in range(num_generations):
    population = Methods.assess_population(population)
    parents = Methods.choose_parents(population)

    Methods.display_results(population, generation)
    current_convergence = generation / num_generations
    mutation_rate = initial_mutation_rate * (1.0 - current_convergence)

    for j in range(0, population_size - 1, 2):
        child1 = Methods.perform_crossover(parents[j], parents[j + 1])
        child1 = Methods.apply_mutation(child1, mutation_rate, min_x, max_x, min_y, max_y)

        population[j] = child1
        population[j].x = child1.x
        population[j].y = child1.y
        population[j].fitness = Methods.calculate_fitness(population[j].x, population[j].y)

    if Methods.check_population_convergence(population):
        print("Fitness Convergence", generation)
        break

    generation_list.append(generation)
    average_fitness = sum(p.fitness for p in population) / len(population)
    average_fitness_list.append(average_fitness)

Methods.draw_fitness_graph(generation_list[:10], average_fitness_list[:10])
Methods.draw_scatter_plots(population)

best_individual = max(population, key=lambda x: x.fitness)
print("Оптимальное решение: \nx =", round(best_individual.x, 4), ", y =", round(best_individual.y, 4), " \nf(x, y) =", round(best_individual.fitness, 4))

