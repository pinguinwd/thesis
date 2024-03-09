import numpy as np
from multiprocessing import Pool, cpu_count
from GA_functions import *
from concurrent.futures import ProcessPoolExecutor

CA_size = 25
generations = 100
population_size = 100
mutation_rate = 0.01
crossover_rate = 0.7
mutation_params_gaussian_std_dev = 5
Tournament_Selection_size = 10


def genetic_algorithm(ga_params):
    # define input_output_pairs
    input_output_pairs_training, input_output_pairs_control = generate_input_output_pairs(sort=ga_params['case_study'],
                                                     CA_size=ga_params['CA_size'],
                                                     percentage=ga_params['input_percentage'])
    # Initialize population randomly
    population = initialize_population(CA_size=ga_params['CA_size'],
                                       population_size=ga_params['population_size'],
                                       percentage=ga_params['input_percentage'],
                                       input_locations=ga_params['input_locations'],
                                       input_order=ga_params['input_order'],
                                       sort_rules=ga_params['sort_rules'])

    # Trackers for mean and best fitness
    mean_fitness_training = []
    mean_fitness_control = []
    best_fitness_training = []
    best_fitness_control = []
    best_chromosomes = []  # List to track the best chromosome in each generation
    last_control_fitness = None

    for generation in range(generations):
        # Prepare arguments for parallel fitness evaluation for both training and control
        fitness_evaluation_args_training = []
        fitness_evaluation_args_control = []
        for individual in population:
            input_cells = individual['input_locations']
            output_cells = individual['output_locations']

            fitness_evaluation_args_training.append(
                (individual['chromosome'], input_output_pairs_training, input_cells, output_cells, ga_params['steps'],
                 ga_params['distance'])
            )

            fitness_evaluation_args_control.append(
                (individual['chromosome'], input_output_pairs_control, input_cells, output_cells, ga_params['steps'],
                 ga_params['distance'])
            )

        # Parallel fitness evaluation for training and control
        with Pool(processes=cpu_count()) as pool:
            fitnesses_training = pool.starmap(evaluate_fitness, fitness_evaluation_args_training)

        if generation % 20 == 0 or generation == 0:
            with Pool(processes=cpu_count()) as pool:
                fitnesses_control = pool.starmap(evaluate_fitness, fitness_evaluation_args_control)
            last_control_fitness = fitness_control
        else:
            # Use the last calculated control fitness
            fitness_control = last_control_fitness

        # Update trackers
        mean_fitness_training.append(np.mean(fitnesses_training))
        mean_fitness_control.append(np.mean(fitnesses_control))
        best_fitness_training.append(np.max(fitnesses_training))
        best_fitness_control.append(np.max(fitnesses_control))

        # Find and store the best chromosome of the current generation
        best_chromosome_index = np.argmax(fitnesses_training)
        best_chromosomes.append(population[best_chromosome_index])

        # Selection (based on training fitnesses)
        parents = select_parents(population, fitnesses_training, ga_params['selection'])

        # Crossover and mutation
        offspring = []
        for _ in range(len(population) // 2):
            parent1, parent2 = np.random.choice(parents, 2, replace=False)
            if np.random.rand() < ga_params['crossover']['rate']:
                child1, child2 = crossover(parent1, parent2, ga_params['crossover'])
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            if np.random.rand() < ga_params['mutation']['rate']:
                child1 = mutate(child1, ga_params['mutation'])
            else:
                child1 = parent1.copy()

            if np.random.rand() < ga_params['mutation']['rate']:
                child2 = mutate(child2, ga_params['mutation'])
            else:
                child2 = parent2.copy()

            offspring.append(child1).append(child2)

        # Create the new population
        population = offspring

        print(mean_fitness_training)

    save_info(mean_fitness_training, mean_fitness_control, best_fitness_training, best_fitness_control, ga_params, 'D:/PythonProjects/Thesis/configurations/output')
    save_chromosomes(best_chromosomes, 'D:/PythonProjects/Thesis/configurations/chromosomes')
    # Return the best solution found
    # For example:
    best_overall_index = np.argmax(best_fitness_training)  # Index of best overall in best_fitness_training list
    best_overall_chromosome = best_chromosomes[best_overall_index]  # The best ov
    return population[best_overall_index]


def generate_random_configuration():
    # Define possible configurations for each parameter, including single-point crossover
    crossover_options = {
        'uniform': {'type': 'uniform', 'rate': np.random.rand()},  # Random rate between 0 and 1
        'single_point': {'type': 'single_point', 'rate': np.random.rand()},  # Random rate between 0 and 1
    }

    mutation_options = {
        'bit_flip': {'type': 'bit_flip', 'rate': np.random.rand()},  # Random rate between 0 and 1
        'gaussian': {'type': 'gaussian', 'std_dev': np.random.uniform(1, 10)},  # Random std dev between 1 and 10
    }
    selection_options = {
        'roulette': {'type': 'roulette'},
        'tournament': {'type': 'tournament', 'size': np.random.randint(2, 10)},
        # Random tournament size between 2 and 10
    }
    distance_options = ['hamming', 'euclidean', 'manhattan', 'perfect_match']
    case_study_options = ['number_divided_by_two', 'number_plus_one']
    input_location_options = ['random', 'evenly_distributed']
    input_order_options = ['random', 'ordered']
    sort_rules_options = ['random', 'big_average_low_std', 'big_average_big_std', 'low_average_low_std', 'equal_behaviour', 'different_behaviour']

    # Randomly select a configuration for each key
    config = {
        'crossover': crossover_options[np.random.choice(list(crossover_options.keys()))],
        'mutation': mutation_options[np.random.choice(list(mutation_options.keys()))],
        'selection': selection_options[np.random.choice(list(selection_options.keys()))],
        'distance': np.random.choice(distance_options),
        'case_study': np.random.choice(case_study_options),
        'CA_size': np.random.choice(range(10, 51, 5)),
        'population_size': np.random.choice(range(30, 201, 10)),
        'input_locations': np.random.choice(input_location_options),
        'input_order': np.random.choice(input_order_options),
        'input_percentage': np.random.rand(),
        'steps': np.random.randint(10, 30),
        'sort_rules': np.random.choice(sort_rules_options)
    }

    return config

def run_ga(ga_params):
    # Wrapper function to run the genetic algorithm with given parameters
    # This function will be executed in parallel
    best_solution = genetic_algorithm(ga_params)
    return best_solution

def main():
    # Generate a list of GA configurations based on the number of available CPU cores
    for i in range(100):
        num_cpus = cpu_count()
        ga_configurations = [generate_random_configuration() for _ in range(num_cpus)]
        with ProcessPoolExecutor(max_workers=num_cpus) as executor:
            futures = [executor.submit(run_ga, ga_params) for ga_params in ga_configurations]
            results = [future.result() for future in futures]




if __name__ == "__main__":
    main()

