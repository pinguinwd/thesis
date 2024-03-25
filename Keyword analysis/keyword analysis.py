#packages
import matplotlib.pyplot as plt
import re
import pandas as pd
import numpy
import networkx as nx
from pyvis.network import Network
from collections import Counter
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import codecs

#network visualisation

def keywordanalysisbar(file, name):
    #preallocation
    abundance = {}

    #Step 1: read in abstract
    with open(file) as f:
        lines = f.read().replace('\n', '').lower()


    #step 2: define the keywords
    keywords = ['deep learning', 'genetic algorithms', 'fuzzy logic', 'machine learning', ' ai ', 'artificial intelligence']

    #find the keywords in the abstracts
    for keyword in keywords:
        abundance[keyword] = lines.count(keyword.lower())


    keys = list(abundance.keys())
    values = list(abundance.values())

    # Create a bar plot
    plt.bar(keys, values)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')

    # Add labels and title
    plt.xlabel('Keywords')
    plt.ylabel('Abundance')
    plt.title('Keyword Abundance Bar Plot')

    #adjust subplot parameter
    plt.subplots_adjust(bottom=0.33)

    # Show the plot
    plt.savefig('C:/Users/user/PycharmProjects/Thesis-Wannes/Statistical analysis/keywordfigs/' + name)

def remove_empty_keywords(keyword_string):
    keywords = keyword_string.split(';')
    non_empty_keywords = [keyword.strip() for keyword in keywords if keyword.strip() != '']
    return ';'.join(non_empty_keywords)

def keywordanalysisnetwork(file, sort):
    # Step 1: read in abstract
    with codecs.open(file, encoding='utf-8') as f:
        lines = f.read()

    #regular expression
    if sort == 'authors':
        pattern = re.compile(r'\sAU ([\s\S]*?)AF')

    if sort == 'keywords':
        pattern = re.compile(r'\nDE ([\s\S]*?)[A-Z]{2}')

    matches = pattern.findall(lines)
    print(matches)
    print('len = ', len(matches))


    if sort == 'authors':
        matches2 = []
        for match in matches:
            k = match.replace('\n', ';').replace(';af ',';')
            matches2.append(k)
        matches = matches2

    if sort == 'keywords':
        matches2 = []
        for match in matches:
            k = match.lower()
            matches2.append(k)
        matches = matches2
    print('matches: ', matches)

    string_list = [remove_empty_keywords(keywords) for keywords in matches]
    print('stringlist: ', string_list)
    forbidden_keywords = ['biosecurity', 'cellular automata', 'antibiotic resistance', 'artificial intelligence'] #no capital letters

    #Only use the 100 most occuring keywords
    # Step 1: Extract all keywords
    # Step 1: Count occurrences of each keyword
    all_keywords = [keyword.strip() for keywords in string_list for keyword in keywords.split(';')]
    print('all keywords: ', all_keywords)
    #all_keywords = list({keyword.strip(): None for keyword in all_keywords}) #make sure keywords that only differ in a space are not used twice #this code removes doubles
    print('all keywords: ', all_keywords)
    all_keywords = [keyword for keyword in all_keywords if keyword not in forbidden_keywords]
    print('all keywords: ', all_keywords)

    keyword_counts = Counter(all_keywords)

    if sort == 'authors':
        number = 100
    if sort == 'keywords':
        number = 60

    # Step 2: Identify the top 100 most used keywords
    top_keywords = [keyword for keyword, count in keyword_counts.most_common(number)]

    print('top keywords: ', top_keywords)

    # Step 3: Filter each string to include only the top 100 keywords
    filtered_string_list = [
        ';'.join([keyword for keyword in keywords.split(';') if keyword.strip() in top_keywords])
        for keywords in string_list
    ]
    print('filtered stringlist: ', filtered_string_list)
    # Step 4: Remove elements with zero or one keyword
    filtered_string_list = [keywords for keywords in filtered_string_list if len(keywords.split(';')) >= 1]
    print('filtered stringlist: ', filtered_string_list)

    if sort == 'keywords':
        #add interesting keywords to the list
        interesting_keywords = ['deep learning', 'genetic algorithm']
    if sort == 'authors':
        #add interesting keywords to the list
        interesting_keywords = []

    filtered_string_list += interesting_keywords
    print('filtered stringlist: ', filtered_string_list)

    # Create an empty DataFrame
    df = pd.DataFrame(columns=['Source', 'Target', 'Weight', 'Type'])

    # Iterate through each string in the list
    for string in filtered_string_list:
        keywords = string.split(';')
        keywords = [keyword.strip() for keyword in keywords]  # Remove leading/trailing spaces

        # Generate pairs of connected keywords
        keyword_pairs = [(keywords[i], keywords[j]) for i in range(len(keywords))
                         for j in range(i + 1, len(keywords))]

        # Update the DataFrame with the connections
        for pair in keyword_pairs:
            source, target = pair
            existing_connection = df[(df['Source'] == source) & (df['Target'] == target)].index.values

            if numpy.any(existing_connection):
                # If connection already exists, increment the weight
                df.at[existing_connection[0], 'Weight'] += 1
            else:
                # If connection doesn't exist, add a new row
                df.loc[len(df)] = [source, target, 1, 'undirected']

    # Display the resulting DataFrame
    G = nx.from_pandas_edgelist(df,
                                    source = 'Source',
                                    target = 'Target',
                                    edge_attr= 'Weight')

    # Create a network graph
    net = Network(notebook=True, cdn_resources='in_line')
    net.from_nx(G)

    # Calculate node size and edge color based on the specified attributes
    node_size = df[['Source', 'Target']].apply(lambda x: x.str.strip()).stack().value_counts().to_dict()
    print(node_size)
    # Normalize 'Weight' values for edge colors
    edge_weights = df['Weight']
    min_weight, max_weight = min(edge_weights), max(edge_weights)
    norm = mcolors.Normalize(vmin=min_weight, vmax=max_weight)
    edge_colors = [cm.viridis(norm(weight)) for weight in edge_weights]


    for node in net.nodes:
        node['size'] = node_size[node['id']]

    # Add edges with edge width attributes
    min_weight, max_weight = min(edge_weights), max(edge_weights)
    norm = mcolors.Normalize(vmin=min_weight, vmax=max_weight)
    edge_widths = [norm(weight) for weight in edge_weights]

    for i in range(len(net.edges)-1):
        net.edges[i]['width'] = edge_widths[i]


    # Generate HTML and save to file
    filename = 'graph_'+sort+'.html'

    html = net.generate_html()
    with open(filename, mode='w', encoding='utf-8') as fp:
        fp.write(html)

    # Display the HTML file in the default web browser
    import webbrowser
    webbrowser.open(filename, new=2)

def netwerkanalysisfuse(filenames, sort):
    with open('outputfile', 'w', encoding='utf-8') as outfile:
        for file in filenames:
            print('file = ', file)
            with open(file, 'r', encoding='utf-8') as f:
                lines = f.read()
                print(lines)
                outfile.write(lines)
    outfile.close()
    print(outfile)
    keywordanalysisnetwork(outfile.name, sort)

#keywordanalysisbar('Keyword_analysis_data/67highlycited.txt', '67highlycited')
#keywordanalysisbar('Keyword_analysis_data/1000relevance.txt', '1000relevance')

#keywordanalysisnetwork('Keyword_analysis_data/1000_relevant_keywords.txt', 'keywords') #authors or keywords
#keywordanalysisnetwork('Keyword_analysis_data/nipt+cell-free.txt', 'authors') #authors or keywords
#keywordanalysisnetwork('Keyword_analysis_data/nipt+cell-free.txt', 'keywords')

netwerkanalysisfuse(['Keyword_analysis_data/all.txt','Keyword_analysis_data/ai.txt'], 'keywords')
#netwerkanalysisfuse(['Keyword_analysis_data/NIPT.txt','Keyword_analysis_data/methylation.txt'], 'authors')

#CA + agent based models


def not_used():
    import requests

    api_key = 'your_api_key'
    base_url = 'https://api.clarivate.com/apis/wos-starter/v1'

    headers = {
        'X-ApiKey': api_key,
        'Content-Type': 'application/json'
    }

    def search_articles(query):
        endpoint = 'search/query'
        params = {
            'databaseId': 'WOS',
            'query': f'TI=({query})',
            'count': 10
        }

        response = requests.get(f'{base_url}{endpoint}', headers=headers, params=params)
        data = response.json()

        return data

    def export_articles(article_ids):
        endpoint = 'data/records'
        params = {
            'databaseId': 'WOS',
            'count': len(article_ids),
            'firstRecord': 1,
            'fields': ['title', 'abstract'],
            'uniqueIds': article_ids
        }

        response = requests.get(f'{base_url}{endpoint}', headers=headers, params=params)
        data = response.json()

        return data

    # Example usage
    query = 'your_search_query'
    search_results = search_articles(query)

    article_ids = [result['uid'] for result in search_results.get('Data', {}).get('Records', [])]
    if article_ids:
        export_data = export_articles(article_ids)
        print(export_data)
    else:
        print("No articles found.")
