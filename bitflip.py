#File to calculate some statistics about non-uniform CA
import copy
import random
import numpy as np
import matplotlib.pyplot as plt
#plt.rcParams['backend'] = 'QT5Agg'
import os
import shutil
from io import BytesIO
import time

#parameters
CA_SIZE = int(101)
NUM_STEPS = int(100)
effect = [0] * 255
NUM_BINS = 20
#firstrules = [0, 11, 15, 18, 30, 32, 35, 40, 41, 54, 60, 62, 73, 90, 106, 110, 150, 160, 204]
firstrules = [30, 32, 35, 40, 41, 54, 60, 62, 73, 90, 106, 110, 150, 160, 204]
plotrules = firstrules

def save_image_to_onedrive(name, image):
    #adjust margins
    image.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

    # Show the figure
    [specific_folder_path, file_name] = name.split('__')
    file_name = file_name + '.pdf'
    file_name2 = file_name + '.png'

    general_folder_path = 'D:/user/Documents/unief/2e master/Thesis/'
    folder_path = general_folder_path + specific_folder_path

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Combine folder path and file name to get the full pathD:\user\Documents\unief\2e master
    full_path = os.path.join(folder_path, file_name)
    full_path2 = os.path.join(folder_path, file_name2)

    # Save the image data to the specified path
    image.savefig(full_path, bbox_inches = 'tight')
    image.savefig(full_path2, bbox_inches = 'tight')
    image.clf()


def firstrulefun(i):
    global firstrule
    firstrule = int(firstrules[i])
    return(firstrule)

def run_CA(ca, NUM_STEPS):
    ca_state = ca.initial_ca_state
    all_states = copy.deepcopy(ca.initial_ca_state)

    for step in range(NUM_STEPS-1):
        next_ca_state = [0] * CA_SIZE

        for i in range(CA_SIZE):
            binary_rule = format(int(ca.rule[i]), '08b')  # Convert rule number to 8-bit binary
            rule_index = 7 - (ca_state[(i-1) % CA_SIZE] * 4 + ca_state[i] * 2 + ca_state[(i+1) % CA_SIZE])
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

def plot_numbers(numbersy, reason, SIMULATIONS):
    plt.clf()
    plt.plot(numbersy)
    plt.xlabel('which secondrule')
    plt.ylabel('Difference with uniform')
    plt.title('Difference over ruleset')
    plt.ylim(0, 0.3)


    name = 'bitflip_' + reason + '__firstrule= ' + str(firstrule) + '_sim=' + str(SIMULATIONS)#use of random integere to make sure it doesn't overwrite
    save_image_to_onedrive(name, plt)

def plot_difference_pattern(all_states_oc, all_states_new, dif, reason, SIMULATIONS, secondrule, change):
    pass

def plot_difference_pattern_False(all_states_oc, all_states_new, dif, reason, SIMULATIONS, secondrule, change):
    # Create subplots
    fig, axs = plt.subplots(3, 1, figsize=(8, 6))

    # Plot each cellular automaton in a subplot
    axs[0].imshow(np.reshape(all_states_oc, [NUM_STEPS, CA_SIZE]), cmap='Greys', interpolation='nearest')
    axs[0].set_title('CA 1')

    axs[1].imshow(np.reshape(all_states_new, [NUM_STEPS, CA_SIZE]), cmap='Greys', interpolation='nearest')
    axs[1].set_title('CA 2')

    axs[2].imshow(np.reshape(dif, [NUM_STEPS, CA_SIZE]), cmap='Greys', interpolation='nearest')
    axs[2].set_title('Dif')

    # Adjust layout for better spacing
    plt.tight_layout()

    # Show the figure
    name = change + '_' + reason + '__firstrule= ' + str(firstrule) + '_sec_rule= ' + str(secondrule) + '_sim=' + str(SIMULATIONS)
    save_image_to_onedrive(name, plt)

#equal rules, equal seed, iterate bitflip location
def vary_loc(SIMULATIONS):

    effect = []
    secondrulelist = []
    differences = [0] * SIMULATIONS

    for secondrule in range(0, 255):
        secondrulelist.append(secondrule)
        initial_ca_state = [random.choice([0, 1]) for _ in range(CA_SIZE)]
        rulelist = [random.choice([firstrule, secondrule]) for _ in range(CA_SIZE)]

        ca = c_automata(rulelist, initial_ca_state)
        ca, all_states_oc = run_CA(ca, NUM_STEPS)

        for sim in range(SIMULATIONS):

            #do one random bitflip
            loc = random.randint(0, CA_SIZE-1)
            ca.initial_ca_state[loc] = (ca.initial_ca_state[loc] + 1) % 2

            #find new states
            ca = c_automata(rulelist, initial_ca_state)
            ca, all_states_new = run_CA(ca, NUM_STEPS)

            #reset bitflip
            ca.initial_ca_state[loc] = (ca.initial_ca_state[loc] + 1) % 2

            #calculate the difference for this simulation
            dif = [abs(all_states_oc[i] - all_states_new[i]) for i in range(CA_SIZE*NUM_STEPS)]
            differences[sim] = sum(dif)/(NUM_STEPS*CA_SIZE)

            if sim == 0 and secondrule in plotrules:
                # Create subplots
                plot_difference_pattern(all_states_oc, all_states_new, dif, 'loc', SIMULATIONS, secondrule, 'bitflip')

        effect.append(np.mean(differences))

    plot_numbers(effect, 'loc', SIMULATIONS)

#iterate rules, equal seed, equal bitflip location
def vary_rules(SIMULATIONS):

    effect = []
    secondrulelist = []
    differences = [0] * SIMULATIONS

    for secondrule in range(0, 255):
        initial_ca_state = [random.choice([0, 1]) for _ in range(CA_SIZE)]
        secondrulelist.append(secondrule)

        for sim in range(SIMULATIONS):
            #first define the rulelist
            rulelist = [random.choice([firstrule, secondrule]) for _ in range(CA_SIZE)]

            #find original states
            ca = c_automata(rulelist, initial_ca_state)
            ca, all_states_oc = run_CA(ca, NUM_STEPS)

            #do one central bitflip
            loc = int(round(CA_SIZE/2))
            ca.initial_ca_state[loc] = (ca.initial_ca_state[loc] + 1) % 2

            #find new states
            ca = c_automata(rulelist, initial_ca_state)
            ca, all_states_new = run_CA(ca, NUM_STEPS)

            # reset bitflip
            ca.initial_ca_state[loc] = (ca.initial_ca_state[loc] + 1) % 2

            #calculate the difference for this simulation
            dif = [abs(all_states_oc[i] - all_states_new[i]) for i in range(CA_SIZE*NUM_STEPS)]
            differences[sim] = sum(dif)/(NUM_STEPS*CA_SIZE)

            if sim == 0 and secondrule in plotrules:
                # Create subplots
                plot_difference_pattern(all_states_oc, all_states_new, dif, 'rules', SIMULATIONS, secondrule, 'bitflip')

        effect.append(np.mean(differences))
    plot_numbers(effect, 'rules', SIMULATIONS)

#equal rules, iterate seed, equal bitflip location
def vary_seed(SIMULATIONS):

    effect = []
    secondrulelist = []
    differences = [0] * SIMULATIONS

    for secondrule in range(0, 255):
        rulelist = [random.choice([firstrule, secondrule]) for _ in range(CA_SIZE)]
        secondrulelist.append(secondrule)

        for sim in range(SIMULATIONS):
            #first define the seed
            initial_ca_state = [random.choice([0, 1]) for _ in range(CA_SIZE)]

            #find original states
            ca = c_automata(rulelist, initial_ca_state)
            ca, all_states_oc = run_CA(ca, NUM_STEPS)

            #do one central bitflip
            loc = int(round(CA_SIZE/2))
            ca.initial_ca_state[loc] = (ca.initial_ca_state[loc] + 1) % 2

            #find new states
            ca = c_automata(rulelist, initial_ca_state)
            ca, all_states_new = run_CA(ca, NUM_STEPS)

            # reset bitflip
            ca.initial_ca_state[loc] = (ca.initial_ca_state[loc] + 1) % 2

            #calculate the difference for this simulation
            dif = [abs(all_states_oc[i] - all_states_new[i]) for i in range(CA_SIZE*NUM_STEPS)]
            differences[sim] = sum(dif)/(NUM_STEPS*CA_SIZE)

            if sim == 0 and secondrule in plotrules:
                # Create subplots
                plot_difference_pattern(all_states_oc, all_states_new, dif, 'seed', SIMULATIONS, secondrule, 'bitflip')

        effect.append(np.mean(differences))
    plot_numbers(effect, 'seed', SIMULATIONS)


def save_text_to_onedrive(action, var2, var3, var4=None, var5=None):

    base_path = "D:/user/Documents/unief/2e master/Thesis/"

    if action == 'READ_ME':
        file_path = os.path.join(base_path, "README.txt")
        with open(file_path, 'w') as file:
            file.write(f"CA_SIZE = {var2}\nNUM_STEPS = {var3}\nNUM_BINS = {var4}\nSIMULATIONS = {var5}")

    elif action == 'simulation':
        data_path = os.path.join(base_path, f"Data/Rule_{var2}/")
        os.makedirs(data_path, exist_ok=True)
        file_path = os.path.join(data_path, f"Sim_firstrule_{var2}_secondrule_{var3}.txt")
        dif_str = '\n[' + ', '.join(map(str, var4)) + ']'
        if not os.path.exists(file_path):
            with open(file_path, 'w') as file:
                file.write(dif_str)
        else:
            with open(file_path, 'a') as file:
                file.write(f"\n{dif_str}")

    elif action == 'summary_dif':
        data_path = os.path.join(base_path, "Data/")
        os.makedirs(data_path, exist_ok=True)
        file_path = os.path.join(data_path, f"summary_{var2}.txt")

        # Convert each list of numbers to a string that preserves list format
        dif_str = '\n[' + ', '.join(map(str, var3)) + ']'

        with open(file_path, 'a') as file:
            file.write(f"\n{dif_str}")

    elif action == 'summary_effect':
        data_path = os.path.join(base_path, "Data/")
        os.makedirs(data_path, exist_ok=True)
        file_path = os.path.join(data_path, f"summary_{var2}.txt")
        effect_str = '\n[' + ', '.join(map(str, var3)) + ']'
        with open(file_path, 'a') as file:
            file.write(f"\n\neffect:\n{effect_str}")

