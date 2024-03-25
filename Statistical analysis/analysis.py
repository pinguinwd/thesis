import numpy as np
import os
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import pandas as pd
from scipy.stats import linregress
from scipy.spatial.distance import euclidean

#First analysis is the comparison of big and small CA
def read_data(folder_path):
    databefore = {}
    dataafter = {}
    for filename in os.listdir(folder_path):
        if filename.startswith("summary_"):
            with open(os.path.join(folder_path, filename), 'r') as file:
                content = file.read()
                before_content, after_content = content.split('effect:', 1)  # Split at 'effect:'

                # Process content before 'effect:'
                before_lists = before_content.strip().split('\n')
                before_lists = [ast.literal_eval(l) for l in before_lists if l.strip()]
                databefore[filename] = before_lists

                # Process content after 'effect:'
                after_lists = after_content.strip().split('\n')
                after_lists = [ast.literal_eval(l) for l in after_lists if l.strip()]
                dataafter[filename] = after_lists

    return databefore, dataafter


def compute_p_values(folder1, folder2):
    data1, _ = read_data(folder1)
    data2, _ = read_data(folder2)
    p_values_matrix = np.zeros((30, 30))

    for i in range(0, 30):  # Assuming files are indexed from 1 to 30
        file_key = f'summary_{i}.txt'
        if file_key in data1 and file_key in data2:
            for j in range(0, 30):
                list1 = data1[file_key][j]
                list2 = data2[file_key][j]
                _, p_value = ttest_ind(list1, list2, equal_var=True)
                p_values_matrix[i, j] = p_value
                print(p_value)
    return p_values_matrix


def plot_p_values_matrix(p_values_matrix):
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(p_values_matrix, cmap='viridis')
    plt.title('P-Values Matrix')
    plt.xlabel('Second Rule')
    plt.ylabel('First Rule')

    # Find the maximum value in the matrix and display it in the corner
    max_value = np.nanmax(p_values_matrix)
    plt.text(0, 0, f'Max: {max_value:.2f}', color='red', ha='left', va='bottom')

    plt.show()
#End of first analysis

#second analysis is the creation of the dataframe which will be used in RStudio
def analyze_list(rule, num_list):
    mean = np.mean(num_list)
    std_dev = np.std(num_list)
    peaks = [x for x in num_list if abs(x - mean) > 2 * std_dev]
    upward_peaks = [x for x in peaks if x > mean]
    downward_peaks = [x for x in peaks if x < mean]
    zeros = num_list.count(0)
    if len(peaks) > 0:
        avg_size_peaks = np.mean([abs(x - mean) for x in peaks])
    else:
        avg_size_peaks = np.nan
    slope, _, _, _, _ = linregress(range(len(num_list)), num_list)

    return {
        'Rule': rule,
        'Average': mean,
        'Amount of peaks': len(peaks),
        'Upward peaks': len(upward_peaks),
        'Downward peaks': len(downward_peaks),
        'Rate of drifting': slope,
        'Amount of zero': zeros,
        'Average size of peaks': avg_size_peaks,
        'Standard deviation': std_dev
    }


def create_dataframe(folder_path):
    data = []
    _, num_dict = read_data(folder_path)
    for key in num_dict.keys():
        index = key.split('summary_')[-1].split('.txt')[0]
        value = num_dict[key]
        num_list = value[0]
        file_data = analyze_list(index, num_list)
        data.append(file_data)

    df = pd.DataFrame(data)
    return df

# Helper function to compute Euclidean distance matrix
def compute_distance_matrix(data, indices):
    size = len(indices)
    matrix = np.zeros((size, size))
    for i, index_i in enumerate(indices):
        for j, index_j in enumerate(indices):
            if i < j:  # Compute once, fill both i,j and j,i
                dist = euclidean(data[f'summary_{index_i}.txt'], data[f'summary_{index_j}.txt'])
                matrix[i, j] = matrix[j, i] = dist
    return matrix

def plot_matrix(matrix, path, title):
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(matrix, cmap='viridis')
    plt.title(title)
    plt.xlabel('Second Rule')
    plt.ylabel('First Rule')

    plt.gca().invert_yaxis()

    plt.savefig(path + title + '.png')

#comparison analysis
def comparison_analysis(folder_path):
    _, data = read_data(folder_path)
    for key in data:
        data[key] = data[key][0]

    # Step 1: Extract and sort indices
    indices = sorted(int(key.split('_')[-1].split('.txt')[0]) for key in data.keys())
    indices = sorted(indices, key=lambda x: (count_ones_in_binary(x), x), reverse=False)

    # Step 2: Compute Euclidean Distance Matrix
    distance_matrix = compute_distance_matrix(data, indices)

    # Normalize lists around 1
    normalized_data = {key: np.array(val) / np.mean(val) for key, val in data.items()}

    # Step 3: Compute normalized distance matrix
    normalized_distance_matrix = compute_distance_matrix(normalized_data, indices)

    # Replace list values with their standard deviation
    std_dev_data = {key: [(x - np.mean(val)) / np.std(val) for x in val] for key, val in data.items()}

    # Step 4: Compute distance matrix for standard deviation replaced lists
    sd_distance_matrix = compute_distance_matrix(std_dev_data, indices)

    # Convert indices to array for indexing
    indices_array = np.array(indices)

    # Printing matrices (or return them as needed)
    print("Euclidean Distance Matrix:")
    plot_matrix(distance_matrix, folder2, "Euclidean Distance Matrix.png")
    print("\nNormalized Distance Matrix:")
    plot_matrix(normalized_distance_matrix, folder2, "Normalized Distance Matrix.png")
    print("\nStandard Deviation Distance Matrix:")
    plot_matrix(sd_distance_matrix, folder2, "Standard Deviation Distance Matrix.png")

def count_ones_in_binary(number):
    return bin(int(number)).count('1')

def ordered_df(df):
    # Adding a helper column for the count of ones in binary representation
    df['ones_count'] = df['Rule'].apply(count_ones_in_binary)

    # Sorting the DataFrame based on ones_count and rule
    df_sorted = df.sort_values(by=['ones_count', 'Rule'], ascending=[True, False])

    # Dropping the helper column as it's no longer needed after sorting
    df_sorted = df_sorted.drop('ones_count', axis=1)
    return df_sorted

#Running of analysis
# Paths to the folders
folder1 = r'D:\user\Documents\unief\2e master\Thesis\bigCA\Data'
folder2 = r'D:\user\Documents\unief\2e master\Thesis\smallCA\Data'
plotfolder = r'D:\user\Documents\unief\2e master\Thesis\Statistical testing\plots'
csvpath = r'D:\user\Documents\unief\2e master\Thesis\Statistical testing\information.csv'

#First analysis is comparison of big and small CA

if False:
    # Compute p-values
    p_values_matrix = compute_p_values(folder1, folder2)
    # Plot the matrix
    plot_p_values_matrix(p_values_matrix)

#creation of dataframe
if True:
    df = create_dataframe(folder2)
    df = ordered_df(df)
    if os.path.isfile(csvpath):
        os.remove(csvpath)
    df.to_csv(csvpath, index=False)

if False:
    comparison_analysis(folder2)