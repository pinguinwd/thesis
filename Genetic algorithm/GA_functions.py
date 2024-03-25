import numpy as np
import cupy as cp
import os
import random
import json

def rule_to_binary_gpu(rule_number):
    binary_string = cp.binary_repr(rule_number, width=8)
    # Convert binary string directly to a CuPy array of integers
    return cp.array([int(bit) for bit in binary_string], dtype=cp.int32)



def apply_rule_gpu(left, center, right, rule_binary):
    """Apply the rule based on the current and neighbor states."""
    index = 7 - cp.left_shift(left, 2) + cp.left_shift(center, 1) + right
    return rule_binary[index]

def evolve_cellular_automata_gpu(initial_sequence, rule_numbers, steps, amount_of_cells):
    # Convert rule numbers to binary representations and prepare rules as a 2D array


    # Convert each rule to binary and stack them into a single CuPy array
    rules_binary = cp.stack([rule_to_binary_gpu(int(rule)) for rule in rule_numbers])

    current_sequence = cp.array(initial_sequence)
    for _ in range(steps):
        new_sequence = cp.empty_like(current_sequence)
        for i in range(len(current_sequence)):
            # Use periodic boundary conditions for neighbors
            left = current_sequence[(i - 1) % len(current_sequence)]
            center = current_sequence[i]
            right = current_sequence[(i + 1) % len(current_sequence)]
            # Apply the rule for the current cell
            new_state = apply_rule_gpu(left, center, right, rules_binary[i])
            new_sequence[i] = new_state
        current_sequence = new_sequence
    return current_sequence.get()  # Return to CPU memory if necessary

#GA steps

def bit_flip_mutation(individual, type, CA_SIZE):
    """Performs bit flip mutation on a randomly selected number from the individual list."""
    # Choose a random index in the individual list to mutate
    idx_to_mutate = np.random.randint(len(individual))

    # Convert the selected number to binary, ensuring correct length for 'rules' or 'locations'
    number_to_mutate = individual[idx_to_mutate]
    binary_length = 8 if type == 'rules' else int(CA_SIZE).bit_length()
    binary_rep = format(number_to_mutate, '0{}b'.format(binary_length))

    success = False
    while not success:
        # Choose a random bit position to flip in the binary representation
        bit_position = np.random.randint(len(binary_rep))
        flipped_binary = list(binary_rep)
        flipped_binary[bit_position] = '1' if binary_rep[bit_position] == '0' else '0'
        flipped_binary = ''.join(flipped_binary)

        # Convert the flipped binary back to a number
        new_number = int(flipped_binary, 2)

        # For 'locations', ensure the new number does not exceed CA_SIZE. If it does, try again.
        if type == 'rules' or (type == 'locations' and new_number <= CA_SIZE):
            success = True
            individual[idx_to_mutate] = new_number  # Apply the mutation

    return individual  # Return the mutated list of numbers


def gaussian_mutation(individual, type, CA_SIZE, std_dev=1.0):
    """Performs Gaussian mutation on an individual with type-specific bounds."""
    # Choose a random position to mutate
    position = np.random.randint(len(individual))
    # Add a Gaussian distributed random value
    mutation_value = np.random.normal(0, std_dev)

    # Apply mutation with bounds checking, specific to 'rules' or 'locations'
    if type == 'rules':
        max_bound = 255  # For 'rules', the max bound is 255
    elif type == 'locations':
        max_bound = CA_SIZE  # For 'locations', the max bound is CA_SIZE
    else:
        raise ValueError("Unknown type specified. Type should be 'rules' or 'locations'.")

    # Apply mutation within bounds
    individual[position] = np.clip(individual[position] + mutation_value, 0, max_bound)
    return individual

def roulette_wheel_selection(population, fitnesses):
    """Performs roulette wheel selection based on fitness."""
    total_fitness = sum(fitnesses)
    selection_probs = [f / total_fitness for f in fitnesses]
    selected_indices = np.random.choice(len(population), size=len(population), p=selection_probs)
    return [population[i] for i in selected_indices]

def tournament_selection(population, fitnesses, tournament_size):
    """Performs tournament selection."""
    selected_parents = []
    for _ in range(len(population)):
        # Select a random subset for the tournament
        indices = np.random.randint(0, len(population), size=tournament_size)
        tournament = [(population[i], fitnesses[i]) for i in indices]
        # Choose the winner with the highest fitness
        winner = max(tournament, key=lambda item: item[1])[0]
        selected_parents.append(winner)
    return selected_parents

def single_point_crossover(parent1, parent2, rate):
    """Performs single-point crossover between two parents."""
    crossover_point = np.random.randint(1, len(parent1))  # Choose crossover point
    child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
    child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
    return child1, child2

def uniform_crossover(parent1, parent2, rate):
    """Performs uniform crossover between two parents."""
    child1, child2 = parent1.copy(), parent2.copy()
    for i in range(len(parent1)):
        child1[i], child2[i] = child2[i], child1[i]
    return child1, child2


def euclidean_distance(s1, s2):
    """Calculate the Euclidean distance between two binary strings."""
    return np.sqrt(sum((int(el1) - int(el2)) ** 2 for el1, el2 in zip(s1, s2)))

def manhattan_distance(s1, s2):
    """Calculate the Manhattan distance between two binary strings."""
    return sum(abs(int(el1) - int(el2)) for el1, el2 in zip(s1, s2))

def perfect_match_distance(str1, str2):
    """Returns 1 if the strings are perfectly equal, 0 otherwise."""
    return 1 if str1 == str2 else 0

#possible input output pairs
def generate_input_output_pairs(sort, CA_size, percentage):
    # Calculate the number of bits and total inputs
    num_bits = int(CA_size * percentage)

    if num_bits < 5:
        num_bits = 5

    total_inputs = 2 ** num_bits



    # Choose the output function based on sort
    if sort == 'number_divided_by_two':
        output_func = number_divided_by_two
    elif sort == 'number_plus_one':
        output_func = number_plus_one
    else:
        raise ValueError(f"Unknown sort: {sort}")

    # Make sure total_inputs is greater than the number of samples you want to draw
    sample_size = min(100, total_inputs)

    # Sample without replacement
    sampled_indices = random.sample(range(total_inputs), sample_size)

    # Create pairs
    input_output_pairs = {i: output_func(i) for i in sampled_indices}

    # Randomly divide into training and control sets
    items = list(input_output_pairs.items())
    np.random.shuffle(items)
    split_index = int(len(items) * 0.5)  # Split 50-50 for training and control
    input_output_pairs_training = dict(items[:split_index])
    input_output_pairs_control = dict(items[split_index:])

    return input_output_pairs_training, input_output_pairs_control

# Define output functions
def number_divided_by_two(x):
    return x // 2

def number_plus_one(x):
    return x + 1


# Placeholder functions for selection, crossover, and mutation
def select_parents(population, fitnesses, selection_params):
    """Main selection function that chooses the selection strategy."""
    if selection_params['type'] == 'roulette':
        return roulette_wheel_selection(population, fitnesses)
    elif selection_params['type'] == 'tournament':
        return tournament_selection(population, fitnesses, selection_params['size'])
    else:
        raise ValueError(f"Unknown selection type: {selection_params['type']}")




def crossover(parent1, parent2, crossover_params):
    """Main crossover function that redirects to specific crossover strategies."""
    if crossover_params['type'] == 'single_point':
        return single_point_crossover(parent1, parent2, crossover_params['rate'])
    elif crossover_params['type'] == 'uniform':
        return uniform_crossover(parent1, parent2, crossover_params['rate'])
    elif crossover_params['type'] == 'none':
        return parent1, parent2
    else:
        raise ValueError("Unknown crossover type specified")


def mutate(individual, mutation_params, sort, CA_SIZE):
    """Main mutation function that redirects to specific mutation methods."""
    mutation_type = mutation_params.get('type', 'bit_flip')
    if mutation_type == 'bit_flip':
        return bit_flip_mutation(individual, sort, CA_SIZE)
    elif mutation_type == 'gaussian':
        std_dev = mutation_params.get('std_dev', 1.0)  # Default standard deviation
        return gaussian_mutation(individual, sort, CA_SIZE, std_dev=1.0)
    else:
        raise ValueError("Unsupported mutation type specified.")


# Master function to calculate distance based on the given option
def calculate_distance(output_binary, expected_output_binary, option):
    # Map of options to distance functions
    distance_functions = {
        'euclidean': euclidean_distance,
        'manhattan': manhattan_distance,
        'perfect_match': perfect_match_distance,
    }

    # Check if the option is valid
    if option not in distance_functions:
        raise ValueError(f"Unknown distance option: {option}")

    # Call the appropriate distance function
    distance_func = distance_functions[option]
    return distance_func(output_binary, expected_output_binary)


def evaluate_fitness(individual, input_output_pairs, input_cells, output_cells, steps, distance_method):
    # Use CuPy arrays for GPU computation
    individual = cp.array(individual)
    fitness_scores = []

    for input_value, expected_output in input_output_pairs.items():
        initial_sequence = cp.zeros(len(individual), dtype=cp.int32)
        input_binary = format(input_value, '0' + str(len(input_cells)) + 'b')

        for idx, bit in zip(input_cells, input_binary):
            initial_sequence[idx] = int(bit)

        final_sequence = evolve_cellular_automata_gpu(initial_sequence, individual, steps, len(input_cells))

        output_binary = ''.join(str(final_sequence[idx]) for idx in output_cells)
        expected_output_binary = format(expected_output, '0' + str(len(output_cells)) + 'b')

        # Calculate distance on GPU if possible or transfer data back to CPU for complex operations
        distance = calculate_distance(output_binary, expected_output_binary, distance_method)  # Adapt this for GPU

        fitness_score = 1 / (1 + distance)  # Inverse of distance as fitness
        fitness_scores.append(fitness_score)

    return np.mean(fitness_scores)

def initialize_population(CA_size, population_size, percentage, input_locations, input_order, sort_rules):
    population = []

    num_locations = int(CA_size * percentage)


    if num_locations < 5:
        num_locations = 5

    if sort_rules == 'random':
        possible_rules = [i for i in range(255)]
    if sort_rules == 'big_average_low_std':
        possible_rules = [105, 165, 122, 129]
    if sort_rules == 'big_average_big_std':
        possible_rules = [242, 243, 245, 139, 58, 112]
    if sort_rules == 'low_average_low_std':
        possible_rules = [71, 233, 164, 217, 178]
    if sort_rules == 'equal_behaviour':
        possible_rules = [137, 193, 110, 124, 147, 54]
    if sort_rules == 'different_behaviour':
        possible_rules = [162, 232, 2, 193, 150, 212]
    # Generating chromosomes from possible_rules using CuPy
    indices = cp.random.randint(0, len(possible_rules), size=(population_size, CA_size))
    chromosomes = cp.array(possible_rules)[indices]

    for i in range(population_size):
        individual = {'chromosome': chromosomes[i].get()}  # Use .get() to transfer array from GPU to CPU

        if input_locations == 'random':
            # Generate locations on the GPU and transfer them to the CPU
            locations = cp.random.choice(cp.arange(CA_size), size=num_locations, replace=False).get()
        elif input_locations == 'evenly_distributed':
            step = CA_size // num_locations
            # Generate locations directly on the CPU since they're deterministic
            locations = np.arange(0, step * num_locations, step)

        if input_order == 'random':
            # Shuffle on the CPU as CuPy doesn't have a direct shuffle method
            np.random.shuffle(locations)
        elif input_order == 'ordered':
            # Sort on the CPU, though locations generated by 'evenly_distributed' are already sorted
            locations = np.sort(locations)

        # Assuming input and output locations are the same for simplification
        individual['input_locations'] = individual['output_locations'] = locations.tolist()

        population.append(individual)

    return population

def numpy_json_serializer(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')

def save_info(mean_fitness_training, mean_fitness_control, best_fitness_training, best_fitness_control, ga_params, generation, base_path):
    os.makedirs(os.path.dirname(base_path), exist_ok=True)
    file_path = f"{base_path}_summary.txt"

    if not os.path.exists(file_path):
        with open(file_path, 'w') as file:
            file.write("GA Parameters:\n")
            serialized_str = json.dumps(ga_params, default=numpy_json_serializer, indent=4)
            file.write(serialized_str)
            file.write("\n\n")
            file.write("Mean Fitness Training:\n")
            file.write(f"{mean_fitness_training}\n\n")
            file.write("Mean Fitness Control:\n")
            file.write(f"{mean_fitness_control}\n\n")
            file.write("Best Fitness Training:\n")
            file.write(f"{best_fitness_training}\n\n")
            file.write("Best Fitness Control:\n")
            file.write(f"{best_fitness_control}\n")
    else:
        updated_content = ""
        section = None
        with open(file_path, 'r') as file:
            for line in file:
                if "Mean Fitness Training:" in line:
                    section = "Mean Fitness Training"
                elif "Mean Fitness Control:" in line:
                    section = "Mean Fitness Control"
                elif "Best Fitness Training:" in line:
                    section = "Best Fitness Training"
                elif "Best Fitness Control:" in line:
                    section = "Best Fitness Control"
                elif section:
                    line = line.strip()
                    if section == "Mean Fitness Training":
                        line += f" {mean_fitness_training}"
                        section = None
                    elif section == "Mean Fitness Control":
                        line += f" {mean_fitness_control}"
                        section = None
                    elif section == "Best Fitness Training":
                        line += f" {best_fitness_training}"
                        section = None
                    elif section == "Best Fitness Control":
                        line += f" {best_fitness_control}"
                        section = None
                    line += "\n"
                updated_content += line

        with open(file_path, 'w') as file:
            file.write(updated_content)

def save_chromosomes(best_chromosomes, base_path, generation):
    os.makedirs(os.path.dirname(base_path), exist_ok=True)

    # Construct the file path
    file_path = f"{base_path}_chromosomes.txt"

    # Check if the file exists
    if not os.path.exists(file_path):
        # File does not exist, create it and write ga_params and initial fitness values
        with open(file_path, 'w') as file:
            # Write ga_params in a readable format
            file.write("Best chromosomes:\n")
            file.write(f"{best_chromosomes}")

    else:
        # File exists, append new fitness values
        with open(file_path, 'a') as file:
            # Append new fitness values
            file.write(f"{best_chromosomes}\n")