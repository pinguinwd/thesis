import numpy as np
import cupy as cp

def rule_to_binary_gpu(rule_number):
    """Convert a rule number to its 8-bit binary representation."""
    return cp.binary_repr(rule_number, width=8)

def apply_rule_gpu(left, center, right, rule_binary):
    """Apply the rule based on the current and neighbor states."""
    index = 7 - cp.left_shift(left, 2) + cp.left_shift(center, 1) + right
    return rule_binary[index]

def evolve_cellular_automata_gpu(initial_sequence, rule_numbers, steps):
    # Convert rule numbers to binary representations and prepare rules as a 2D array
    rules_binary = cp.array([cp.array(list(map(int, rule_to_binary_gpu(rule)))) for rule in rule_numbers])

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
def bit_flip_mutation(individual):
    """Performs bit flip mutation on an individual."""
    # Choose a random bit position to flip
    bit_position = np.random.randint(len(individual))
    # Flip the bit at the chosen position
    individual[bit_position] = 1 - individual[bit_position]
    return individual

def gaussian_mutation(individual, std_dev=1.0):
    """Performs Gaussian mutation on an individual."""
    # Choose a random position to mutate
    position = np.random.randint(len(individual))
    # Add a Gaussian distributed random value
    mutation_value = np.random.normal(0, std_dev)
    # Apply mutation with bounds checking
    individual[position] = np.clip(individual[position] + mutation_value, 0, 255)
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
    if np.random.rand() < rate:
        crossover_point = np.random.randint(1, len(parent1))  # Choose crossover point
        child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
        return child1, child2
    else:
        # Return parents as is if crossover does not happen
        return parent1, parent2

def uniform_crossover(parent1, parent2, rate):
    """Performs uniform crossover between two parents."""
    child1, child2 = parent1.copy(), parent2.copy()
    for i in range(len(parent1)):
        if np.random.rand() < rate:  # With a certain probability, swap the genes
            child1[i], child2[i] = child2[i], child1[i]
    return child1, child2

def hamming_distance(s1, s2):
    """Calculate the Hamming distance between two binary strings."""
    return sum(el1 != el2 for el1, el2 in zip(s1, s2))

def euclidean_distance(s1, s2):
    """Calculate the Euclidean distance between two binary strings."""
    return np.sqrt(sum((el1 - el2) ** 2 for el1, el2 in zip(s1, s2)))

def manhattan_distance(s1, s2):
    """Calculate the Manhattan distance between two binary strings."""
    return sum(abs(el1 - el2) for el1, el2 in zip(s1, s2))

def perfect_match_distance(str1, str2):
    """Returns 1 if the strings are perfectly equal, 0 otherwise."""
    return 1 if str1 == str2 else 0

#possible input output pairs
def generate_input_output_pairs(sort, CA_size, percentage):
    # Calculate the number of bits and total inputs
    num_bits = int(CA_size * percentage)
    total_inputs = 2 ** num_bits

    # Choose the output function based on sort
    if sort == 'number_divided_by_two':
        output_func = number_divided_by_two
    elif sort == 'number_plus_one':
        output_func = number_plus_one
    else:
        raise ValueError(f"Unknown sort: {sort}")

    # Generate all possible input-output pairs
    input_output_pairs = {i: output_func(i) for i in range(total_inputs)}

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
    else:
        raise ValueError("Unknown crossover type specified")


def mutate(individual, mutation_params):
    """Main mutation function that redirects to specific mutation methods."""
    mutation_type = mutation_params.get('type', 'bit_flip')
    if mutation_type == 'bit_flip':
        return bit_flip_mutation(individual)
    elif mutation_type == 'gaussian':
        std_dev = mutation_params.get('std_dev', 1.0)  # Default standard deviation
        return gaussian_mutation(individual, std_dev)
    else:
        raise ValueError("Unsupported mutation type specified.")


# Master function to calculate distance based on the given option
def calculate_distance(output_binary, expected_output_binary, option):
    # Map of options to distance functions
    distance_functions = {
        'hamming': hamming_distance,
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

        final_sequence = evolve_cellular_automata_gpu(initial_sequence, individual, steps)

        output_binary = ''.join(str(final_sequence[idx].get()) for idx in output_cells)
        expected_output_binary = format(expected_output, '0' + str(len(output_cells)) + 'b')

        # Calculate distance on GPU if possible or transfer data back to CPU for complex operations
        distance = calculate_distance(output_binary, expected_output_binary, distance_method)  # Adapt this for GPU

        fitness_score = 1 / (1 + distance)  # Inverse of distance as fitness
        fitness_scores.append(fitness_score.get())

    return cp.mean(fitness_scores).get()

def initialize_population(CA_size, population_size, percentage, input_locations, input_order):
    population = []

    num_locations = int(CA_size * percentage)

    # Use CuPy to generate the chromosomes directly on the GPU
    chromosomes = cp.random.randint(0, 256, size=(population_size, CA_size), dtype=cp.int32)

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
        individual['input_location'] = individual['output_location'] = locations.tolist()

        population.append(individual)

    return population