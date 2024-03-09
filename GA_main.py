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
    input_output_pairs = generate_input_output_pairs(sort=ga_params['case_study'],
                                                     CA_size=ga_params['CA_size'],
                                                     percentage=ga_params['input_percentage'])
    # Initialize population randomly
    population = initialize_population(CA_size=ga_params['CA_size'],
                                       population_size=ga_params['population_size'],
                                       percentage=ga_params['percentage'],
                                       input_locations=ga_params['input_locations'],
                                       input_order=ga_params['input_order'])

    for generation in range(generations):
        # Prepare arguments for parallel fitness evaluation
        fitness_evaluation_args = []
        for individual in population:
            # Extract input and output locations for each individual
            input_cells = individual['input_location']
            output_cells = individual['output_location']

            # Append the arguments needed for evaluate_fitness function
            fitness_evaluation_args.append(
                (individual['chromosome'], input_output_pairs, input_cells, output_cells, ga_params[''], ga_params['distance'])
            )

        # Parallel fitness evaluation
        with Pool(processes=cpu_count()) as pool:
            fitnesses = pool.starmap(evaluate_fitness, fitness_evaluation_args)
        # Selection
        parents = select_parents(population, fitnesses, ga_params['selection'])

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

        # Optionally, implement a termination condition or evaluation here

    # Return the best solution found
    best_index = np.argmax(fitnesses)
    return population[best_index]


def generate_random_configuration():
    # Define possible configurations for each parameter
    crossover_options = {
        'uniform': {'type': 'uniform', 'rate': np.random.rand()},  # Random rate between 0 and 1
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

    # Randomly select a configuration for each key
    config = {
        'crossover': crossover_options[np.random.choice(list(crossover_options.keys()))],
        'mutation': mutation_options[np.random.choice(list(mutation_options.keys()))],
        'selection': selection_options[np.random.choice(list(selection_options.keys()))],
        'distance': np.random.choice(distance_options),
        'case_study': np.random.choice(case_study_options),
        'CA_size': np.random.choice(range(10, 51, 5)),
        'population_size': np.random.choice(range(30, 201, 10)),
        'input_location': np.random.choice(input_location_options),
        'input_order': np.random.choice(input_order_options),
        'input_percentage': np.random.rand(),
        'steps': np.random.randint(10, 30)
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

