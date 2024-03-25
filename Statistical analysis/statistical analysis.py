# ---------------------------------------------------------
# File to calculate the effect of going from a uniform CA to a non-uniform CA where only the middle rule is flipped
# --------------------------------------------------------

import copy
import random
import numpy as np
import matplotlib.pyplot as plt
from bitflip import *

# Sanity check: deffect cones
#mss niet totaal space time maar dichtheid defect cone
#lyaponov coef.


def run_CA(ca, NUM_STEPS):
    ca_state = ca.initial_ca_state
    all_states = copy.deepcopy(ca.initial_ca_state)

    for step in range(NUM_STEPS-1):
        next_ca_state = [0] * CA_SIZE

        for i in range(CA_SIZE):
            binary_rule = format(ca.rule[i], '08b')  # Convert rule number to 8-bit binary
            rule_index = 7 - (ca_state[(i - 1) % CA_SIZE] * 4 + ca_state[i] * 2 + ca_state[(i + 1) % CA_SIZE])
            state = int(binary_rule[rule_index])
            next_ca_state[i] = state
            all_states.append(state)

        ca_state = next_ca_state

    ca.current_ca = ca_state
    return ca, all_states


class c_automata:
    def __init__(self, rulelist, initial_ca_state):
        self.rule = rulelist
        self.initial_ca_state = initial_ca_state

    def __init__copy(self, copy_me):
        self.rule = copy_me.rule[:]
        self.initial_ca_state = copy_me.initial_ca_state[:]


def plot_numbers(numbersx, numbersy, reason):
    plt.clf()
    plt.plot(numbersx, numbersy)
    plt.xlabel('which secondrule')
    plt.ylabel('Difference with uniform')
    plt.title('Line Plot of Numbers')
    plt.ylim(0, 0.3)
    name = 'ruleflip_' + reason + '__firstrule= ' + str(firstrule) + '_sim=' + str(SIMULATIONS)  # use of random integere to make sure it doesn't overwrite
    save_image_to_onedrive(name, plt)




# The first option is to change the rule in the center
def vary_uni_vs_non_uni(SIMULATIONS):
    initial_ca_state = [random.choice([0, 1]) for _ in range(CA_SIZE)]
    rulelist = [firstrule] * CA_SIZE
    ca = c_automata(rulelist, initial_ca_state)
    ca, all_states_oc = run_CA(ca, NUM_STEPS)

    effect = []
    secondrulelist = []

    for secondrule in range(0, 255):
        print(secondrule)
        secondrulelist.append(secondrule)

        # flip the central rule
        rulelist[round(CA_SIZE / 2)] = secondrule

        ca = c_automata(rulelist, initial_ca_state)
        ca, all_states_new = run_CA(ca, NUM_STEPS)

        dif = [abs(all_states_oc[i] - all_states_new[i]) for i in range(CA_SIZE * NUM_STEPS)]

        plot_difference_pattern(all_states_oc, all_states_new, dif, 'central', SIMULATIONS, secondrule, 'ruleflip')

        effect.append(sum(dif) / (NUM_STEPS * CA_SIZE))

    plot_numbers(secondrulelist, effect, 'central')



# The second is to vary the location of the rule that is flipped and compare with uniform
def vary_ruleloc(SIMULATIONS):
    initial_ca_state = [random.choice([0, 1]) for _ in range(CA_SIZE)]
    rulelist = [firstrule] * CA_SIZE

    ca = c_automata(rulelist, initial_ca_state)
    ca, all_states_oc = run_CA(ca, NUM_STEPS)

    effect = []
    secondrulelist = []

    for secondrule in range(0, 255):
        secondrulelist.append(secondrule)

        # restore the rulelist to remove previous flip
        rulelist = [firstrule] * CA_SIZE

        # do a new flip
        rulelist[round(CA_SIZE / 2)] = secondrule

        for sim in range(SIMULATIONS):
            # scatter the rulelist (as this has no effect on the OC_ca, we can recycle the original one)
            random.shuffle(rulelist)

            ca = c_automata(rulelist, initial_ca_state)
            ca, all_states_new = run_CA(ca, NUM_STEPS)
            dif = [abs(all_states_oc[i] - all_states_new[i]) for i in range(CA_SIZE * (NUM_STEPS))]

            if sim == 0 and secondrule in plotrules:
                # Create subplots
                plot_difference_pattern(all_states_oc, all_states_new, dif, 'loc', SIMULATIONS, secondrule, 'ruleflip')

            differences[sim] = sum(dif)/ (NUM_STEPS * (CA_SIZE))

        effect.append(np.mean(differences))

    plot_numbers(secondrulelist, effect, 'loc')


# Another option is to always flip the middle cell but to use different starting conditions over the simulation
def vary_inicond(SIMULATIONS, firstrule):
    rulelist = [firstrule] * CA_SIZE
    effect = []
    secondrulelist = []

    for secondrule in range(0, 255):
        secondrulelist.append(secondrule)

        for sim in range(SIMULATIONS):
            initial_ca_state = [random.choice([0, 1]) for _ in range(CA_SIZE)]

            #reset rulelist
            rulelist[round(CA_SIZE / 2)] = firstrule


            ca = c_automata(rulelist, initial_ca_state)
            ca, all_states_oc = run_CA(ca, NUM_STEPS)

            # flip the central rule
            rulelist[round(CA_SIZE / 2)] = secondrule

            ca = c_automata(rulelist, initial_ca_state)
            ca, all_states_new = run_CA(ca, NUM_STEPS)

            dif = [abs(all_states_oc[i] - all_states_new[i]) for i in range(CA_SIZE * NUM_STEPS)]
            differences[sim] = sum(dif) / (NUM_STEPS * CA_SIZE)

            if sim == 0 and secondrule in plotrules:
                # Create subplots
                plot_difference_pattern(all_states_oc, all_states_new, dif, 'seed', SIMULATIONS, secondrule, 'ruleflip')
            save_text_to_onedrive('simulation', firstrule, secondrule, all_states_oc)

        effect.append(np.mean(differences))
        save_text_to_onedrive('summary_dif', firstrule, differences)

    save_text_to_onedrive('summary_effect', firstrule, effect)
    plot_numbers(secondrulelist, effect, 'seed')

#ofwel seed ofwel locatie
#defect cone symmetrie
#defect cone breedte
#information boundaries
#regel 54
#images exporteren als pdf ipv png
#netwerk anaylyse als pdf opslaan
#documenteren hoe je data exporteert voor keyword analysis

SIMULATIONS = 50
save_text_to_onedrive('READ_ME', CA_SIZE, NUM_STEPS, NUM_BINS, SIMULATIONS)

for i in range(54,55):
    firstrule = i

    differences = [0] * SIMULATIONS

    #ruleflip
    #vary_ruleloc(SIMULATIONS)
    vary_inicond(SIMULATIONS, firstrule)
    #vary_uni_vs_non_uni(SIMULATIONS)

    # bitflip
    #vary_loc(SIMULATIONS)
    #vary_rules(SIMULATIONS)
    #vary_seed(SIMULATIONS)


def plot_first_line(file_path, NUM_STEPS, CA_SIZE, output_path):
    # Read the first line from the file and convert it into a list
    with open(file_path, 'r') as file:
        first_line = file.readline().strip()
        data_list = eval(first_line)  # Safely evaluates the string as a python list

    # Ensure the data list size matches NUM_STEPS * CA_SIZE
    if len(data_list) != NUM_STEPS * CA_SIZE:
        raise ValueError("The size of data_list does not match NUM_STEPS * CA_SIZE")

    # Reshape the data to fit the specified dimensions
    data = np.reshape(data_list, (NUM_STEPS, CA_SIZE))

    # Plot the data as an image
    plt.imshow(data, cmap='gray', interpolation='nearest')
    plt.colorbar()  # Optional: Adds a colorbar to indicate scale
    plt.axis('off')  # Hide the axes for an image-like presentation

    # Adjust layout for better spacing
    plt.tight_layout()

    # Save the figure to the specified output path
    plt.savefig(output_path, bbox_inches='tight',
                dpi=300)  # Saves with minimal padding; adjust dpi for higher resolution

    # Optionally, display the plot (comment out if not needed)
    # plt.show()
