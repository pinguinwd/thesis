import uuid
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
    last_control_fitness = [0]
    fitnesses_training = []
    fitnesses_control = []

    identifier = uuid.uuid4()
    base_path = 'D:/PythonProjects/Thesis/simulations/' + str(identifier) + '/'

    for generation in range(generations):
        print('generation: ' + str(generation) + '/' + str(generations))
        fitnesses_training = []
        for individual in population:
            input_cells = individual['input_locations']
            output_cells = individual['output_locations']


            fitness = evaluate_fitness(individual['chromosome'], input_output_pairs_training, input_cells, output_cells, ga_params['steps'],
                    ga_params['distance'])
            fitnesses_training.append(fitness)

        if generation % 20 == 0 or generation == 0:

            fitnesses_control = []
            for individual in population:
                input_cells = individual['input_locations']
                output_cells = individual['output_locations']

                fitnesses = evaluate_fitness(individual['chromosome'], input_output_pairs_control, input_cells, output_cells, ga_params['steps'],
                    ga_params['distance'])
                fitnesses_control.append(fitnesses)
            last_control_fitness = fitnesses_control
        else:
            # Use the last calculated control fitness
            fitnesses_control = last_control_fitness

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
        # Crossover and mutation
        offspring = []
        for _ in range(len(population) // 2):
            parent1, parent2 = np.random.choice(parents, 2, replace=False)

            # Initialize children as copies of parents
            child1, child2 = parent1.copy(), parent2.copy()

            # Apply crossover
            if ga_params['chromosome_evol'].lower() == 'yes' and np.random.rand() < ga_params['crossover']['rate']:
                child1['chromosome'], child2['chromosome'] = crossover(parent1['chromosome'], parent2['chromosome'],
                                                                       ga_params['crossover'])

            # Apply mutation
            if ga_params['chromosome_evol'].lower() == 'yes' and np.random.rand() < ga_params['mutation']['rate']:
                child1['chromosome'] = mutate(child1['chromosome'], ga_params['mutation'], 'rules', ga_params['CA_size'])
                child2['chromosome'] = mutate(child2['chromosome'], ga_params['mutation'], 'rules', ga_params['CA_size'])

            # Check for input locations evolution
            if ga_params['input_loc_evol'].lower() == 'yes':
                child1['input_locations'], child2['input_locations'] = crossover(parent1['input_locations'], parent2['input_locations'],
                                                                       ga_params['crossover'])

            # Apply mutation
            if ga_params['input_loc_evol'].lower() == 'yes' and np.random.rand() < ga_params['mutation']['rate']:
                child1['input_locations'] = mutate(child1['input_locations'], ga_params['mutation'], 'locations', ga_params['CA_size'])
                child2['input_locations'] = mutate(child2['input_locations'], ga_params['mutation'],  'locations', ga_params['CA_size'])

            # Check for output locations evolution
            if ga_params['output_loc_evol'].lower() == 'yes':
                child1['output_locations'], child2['output_locations'] = crossover(parent1['output_locations'],
                                                                                 parent2['output_locations'],
                                                                                 ga_params['crossover'])

            # Apply mutation
            if ga_params['output_loc_evol'].lower() == 'yes' and np.random.rand() < ga_params['mutation']['rate']:
                child1['output_locations'] = mutate(child1['output_locations'], ga_params['mutation'], 'locations', ga_params['CA_size'])
                child2['output_locations'] = mutate(child2['output_locations'], ga_params['mutation'], 'locations', ga_params['CA_size'])

            offspring.extend([child1, child2])


        save_info(mean_fitness_training[-1], mean_fitness_control[-1], best_fitness_training[-1],
                  best_fitness_control[-1], ga_params, generation,
                  base_path)
        save_chromosomes(population[best_chromosome_index], base_path, generation)
        # Create the new population
        population = offspring




def generate_random_configuration():
    # Define possible configurations for each parameter, including single-point crossover
    crossover_options = {
        'uniform': {'type': 'uniform', 'rate': np.random.rand()},  # Random rate between 0 and 1
        'single_point': {'type': 'single_point', 'rate': np.random.rand()},
        'none': {'type': 'none', 'rate': np.random.rand()}# Random rate between 0 and 1
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
    distance_options = ['euclidean', 'manhattan', 'perfect_match']
    case_study_options = ['number_divided_by_two', 'number_plus_one']
    input_location_options = ['random', 'evenly_distributed']
    input_order_options = ['random', 'ordered']
    sort_rules_options = ['random', 'big_average_low_std', 'big_average_big_std', 'low_average_low_std', 'equal_behaviour', 'different_behaviour']
    chromosome_evol_options = ['yes', 'no']
    input_loc_evol = ['yes', 'no']
    output_loc_evol = ['yes', 'no']


    # Randomly select a configuration for each key
    config = {
        'crossover': crossover_options[np.random.choice(list(crossover_options.keys()))],
        'mutation': mutation_options[np.random.choice(list(mutation_options.keys()))],
        'selection': selection_options[np.random.choice(list(selection_options.keys()))],
        'distance': np.random.choice(distance_options),
        'case_study': np.random.choice(case_study_options),
        'CA_size': np.random.choice(range(10, 30, 5)),
        'population_size': np.random.choice(range(8,20)),
        'input_locations': np.random.choice(input_location_options),
        'input_order': np.random.choice(input_order_options),
        'input_percentage': np.random.rand(),
        'steps': np.random.randint(10, 30),
        'sort_rules': np.random.choice(sort_rules_options),
        'chromosome_evol': np.random.choice(chromosome_evol_options),
        'input_loc_evol': np.random.choice(input_loc_evol),
        'output_loc_evol': np.random.choice(output_loc_evol)
    }

    return config

def run_ga(ga_params):
    # Wrapper function to run the genetic algorithm with given parameters
    # This function will be executed in parallel
    genetic_algorithm(ga_params)
    return None

def main():
    # Generate a list of GA configurations based on the number of available CPU cores
    for i in range(100):
        num_cpus = cpu_count()
        ga_configurations = [generate_random_configuration() for _ in range(int(num_cpus))] #divided by two to fix memory issue
        with ProcessPoolExecutor(max_workers=num_cpus) as executor:
            futures = [executor.submit(run_ga, ga_params) for ga_params in ga_configurations]
            results = [future.result() for future in futures]




if __name__ == "__main__":
    main()

