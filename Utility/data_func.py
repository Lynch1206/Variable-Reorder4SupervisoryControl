import pandas as pd
import numpy as np
import os
from pathlib import Path
import re
from Utility.Hygraph_json import Hyper_Read
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

def edge_missing(edges, nodes):
    """
    Return a new edges dataframe with the missing nodes added to the 'hyperedge' column
    """
    e = edges.copy()
    n = nodes.copy()
    unique_ids = set([id for sublist in e['hyperedge'] for id in sublist])
    missing_ids = n[~nodes['id'].isin(unique_ids)]
    new_row = pd.DataFrame({'hyperedge':  missing_ids['id'] })
    last_id = e['id'].iloc[-1]
    new_row['id'] = [last_id + i + 1 for i in range(len(missing_ids))]
    df = pd.concat([edges, new_row], ignore_index=True)
    # Check and convert every row in the 'hyperedge' column to list type
    for index, row in df.iterrows():
        if not isinstance(row['hyperedge'], list):
            df.at[index, 'hyperedge'] = [row['hyperedge']]
    return df


# new graph read
def Graph_conv(edge, nodes):
    Graph = {}
    for i in range(len(edge)):
        Graph[edge['id'][i]] = []
        for j in range(len(edge['hyperedge'][i])):
            Graph[edge['id'][i]].append(nodes['label'][edge['hyperedge'][i][j]])

    return Graph


def files_list_dir(Path=None, search_str = None):
    """
    for .json file
    """
    if Path is None:
        print(f"No input directory.")
        return None
    files = os.listdir(Path)
    data = [file for file in files if file.endswith('.json')]
    if search_str is not None:
        filtered_list = [item for item in data if search_str in item]
        data = pd.DataFrame(data, columns=["Data"])
        return data, filtered_list
    else:
        return pd.DataFrame(data, columns=["Data"])

    

def parse_row(row):
    """
    Parse edges_source column to event, guard and assignments columns
    """
    # \(event: ([^)]*)\)
    pattern = r'\(event: (.*)\) \(guard: (.*?)\) \(assignments: (.*?)\)'
    match = re.search(pattern, row)
    if match:
        return match.groups()
    else:
        return None, None, None

def parse_edges_source(df):
    """
    Parse edges_source column to event, guard and assignments columns
    match eclipse ESCET string format
    """
    # check whether input has edges_source column
    if 'edges_source' not in df.columns:
        raise ValueError('Input dataframe does not have edges_source column')
    df_new = pd.DataFrame()
    df_new[['event', 'guard', 'assignments']] = df['edges_source'].apply(parse_row).apply(pd.Series)
    # drop event and guards
    df_new = df_new.drop(columns=['event'])
    df_new = df_new.drop(columns = ['guard'])
    # merge the new dataframe with the original dataframe
    return df.join(df_new)

def extract_edges_data(edges_df):
    # Initialize an empty list to store the results
    results = []

    # Iterate over each row in the DataFrame
    for i in edges_df['edges_source']:
        pattern = r'(event|guard|assignments): (.*?)(?=\s+\(|$)'
        # Find all matches for the pattern
        matches = re.findall(pattern, i)

        # Initialize a dictionary to hold the current row's results
        row_result = {'event': None, 'guard': None, 'assignments': None}

        # Iterate over each match and update the row_result dictionary accordingly
        for match in matches:
            if match[0] == 'event':
                row_result['event'] = match[1]
                #  remove the last item when it's ')'
                if row_result['event'][-1] == ')':
                    row_result['event'] = row_result['event'][:-1]
            elif match[0] == 'guard':
                row_result['guard'] = match[1]
                #  remove the last item when it's ')'
                if row_result['guard'][-1] == ')':
                    row_result['guard'] = row_result['guard'][:-1]
            elif match[0] == 'assignments':
                row_result['assignments'] = match[1]
                #  remove the last item when it's ')'
                if row_result['assignments'][-1] == ')':
                    row_result['assignments'] = row_result['assignments'][:-1]

        # Append the row_result dictionary to the results list
        results.append(row_result)

    # Convert the results list to a DataFrame
    new_df = pd.DataFrame(results)
    
    return new_df

def parse_edges_sources_merege(df):
    if 'edges_source' not in df.columns:
        raise ValueError('Input dataframe does not have edges_source column')
    df_edge = df.copy()
    df_new = extract_edges_data(edges_df= df_edge)
    df.drop(columns=['edges_event'], inplace=True)
    df.drop(columns=['edges_guard'], inplace=True)
    return df.join(df_new)

def key_word_load(path, keyword):
    """
    A much simpler version 
    """
    folder_list = os.listdir(path)
    # check whether keywords match the list of files names
    if not any(keyword in file for file in folder_list):
        raise ValueError(f"No matched {keyword} files in the folder")
    for file in folder_list:
        if keyword in file:
            data = (path / file)
            # print(f"File {file} is loaded")
        else:
            # read first  file in the folder, if no file, raise error 
            if len(folder_list) == 0:
                raise ValueError("No files in the folder")        
    # read json file 
    if data.suffix == '.json':
        data = Hyper_Read(data).read_js()
        for i in list(data.keys()):
            data = pd.DataFrame(data[i])
        # if data is empty data frame, simply print out message
        if data.empty:
            print("Data is empty")
            return None
        else:
            return data
        return data
    else:
        raise ValueError("File is not in json format")

def check_alphabet(df_alphabet, df_edges_source):
    if not df_alphabet['name'].equals(df_edges_source['event']):
        print("The alphabet is not equal to the edges source")
        # replace alphabet name with df_edges_source['event']
        df_alphabet['name'] = df_edges_source['event']
    else:
        print("The alphabet is equal to the edges source")
    return df_alphabet

def read_data(Route):
    # check if directoyr is empty
    if len(os.listdir(Route)) == 0:
        print(f"Empty directory {Route}")
        return None
    df = {}
    # read files only ends with .json
    for i in os.listdir(Route):
        if i.endswith('.json'):
            if "edges" in i:
                df[i] = parse_edges_sources_merege(key_word_load(path=Route, keyword=i))
            else:
                df[i] = key_word_load(path=Route, keyword=i)
    # check empty
    for i in df.keys():
        if df[i] is None:
            print(f"Empty data in {i}")
    key1 = "alphabet"
    key2 = "edges"
    clean_data = [0, 0] # initial clean_data 2 zeros
    for i in os.listdir(Route):
        if key1 in i:
            clean_data[0] = i
        if key2 in i:
            clean_data[1] = i

    return df

def Var_plant_rq(df_variable_raw, df_plant, df_rq):
    """
    df_variable_raw: varaible.json
    df_plant: plants.json
    df_rq: requirements_aut.json
    Extended the variables dataframe by labeling the plant, requirement id.

    Identify the source of variables.
    """
    df_variable = df_variable_raw.copy()
    # initial with Nan
    df_variable['plant_id'] = pd.NA
    df_variable['requirement_id'] = pd.NA

    for index,row in df_variable.iterrows():
        # check if plant name in the variables
        if df_plant is None:
            df_variable.at[index, 'plant_id'] = pd.NA
        else:
            for i in df_plant['name']:
                if i in row['name']:
                    df_variable.at[index, 'plant_id'] = i
                    break
        if df_rq is None:
            df_variable.at[index, 'requirement_id'] = pd.NA
        else:
            for j in df_rq['name']:
                if j in row['name']:
                    df_variable.at[index, 'requirement_id'] = j
                    break
        
    return df_variable

def Alphabet_plant_rq(df_alphabet, df_plant, df_rq):
    """
    df_alphabet: alphabet.json
    df_plant: plants.json
    df_rq: requirements_aut.json
    Extended the alphabet dataframe by labeling the plant, requirement id.

    Identify the source of alphabet.
    """
    df_alphabet_new = df_alphabet.copy()
    # initial with Nan
    df_alphabet_new['plant_id'] = pd.NA
    df_alphabet_new['requirement_id'] = pd.NA

    for index,row in df_alphabet_new.iterrows():
        # check if plant name in the alphabet
        if df_plant is None:
            df_alphabet_new.at[index, 'plant_id'] = pd.NA
        else:
            for i in df_plant['name']:
                if i in row['name']:
                    df_alphabet_new.at[index, 'plant_id'] = i
                    break
        if df_rq is None:
            df_alphabet_new.at[index, 'requirement_id'] = pd.NA
        else:
            for j in df_rq['name']:
                if j in row['name']:
                    df_alphabet_new.at[index, 'requirement_id'] = j
                    break
        
    return df_alphabet_new


def label_alphabet(df):
    """
    label controllable and rquiremen_event for the input alphabet dataframe
    df : df['alphabet.json']
    """
    if not df.empty:
        le = LabelEncoder()
        df['controllable'] = le.fit_transform(df['controllable'])
        df['requirement_event'] = le.fit_transform(df['requirement_event'])
        return df
    else:
        print("The dataframe is empty")
        return df
    

def fnc_var_2list(a,b,variables):
    """
    a : guard_variable : pd.series
    b : assignments_variable : pd.series
    variables : pd.series, df['variables.json']['name']
    """
    guard_list = set(v for l in a for v in l)
    # unique of list
    a = list(set(guard_list))
    # assingments list
    assignments_list = set(v for l in b for v in l)
    b = list(set(assignments_list))
    # merge 2 list
    missing = []
    result = list(set(a + b))
    for i in variables:
        if i not in result:
            missing.append(i)
    
    if len(missing) == 0:
        print("All the variables are in the guard and assignments")
    return result, missing


def Match_V(df, df2, match_item, search_item, create_item):
    """
    df : dataframe that needs to expand
    df2 : dataframe that contains the match item, variables ... 
    match_item : the column name of the dataframe that needs to match, 'name', 'variables'...
    search_item : the column name of the dataframe that needs to search, 'guard', 'assignments'...
    create_item : the column name of the dataframe that needs to create, 'guard_variable', 'assignments_variable'...
    """
    df = df.copy()
    df_match = df2.copy()
    for item in df_match[match_item]:
        for index, row in df.iterrows():
            try:
                # exact match
                # tokens = re.findall(r'\b\w+\b', row[search_item])
                # if any(token == item for token in tokens):
                #     if item not in df.at[index, create_item]:
                #         df.at[index, create_item].append(item)
                if item in row[search_item]:
                    if item not in df.at[index, create_item]:
                        df.at[index, create_item].append(item)
            except:
                pass
                # print(f"Matched {item} in {search_item} and added to {create_item}")
    return df

def edge_mapping(df, event, vairable, plant, requirement_aut):
    df_edge = df.copy()
    df_event = Alphabet_plant_rq(event, plant, requirement_aut)
    df_variable = Var_plant_rq(vairable, plant, requirement_aut)
    # rename the column name of variable : name -> variable
    df_variable = df_variable.rename(columns={'name': 'variable'})
    # rename the column name of event : name -> alphabet
    df_event = label_alphabet(df_event)  # for encoding the plant, requirement id
    df_event = df_event.rename(columns={'name': 'alphabet'})
    # concatenate the df_edge with df_event
    df_edge = pd.concat([df_edge, df_event], axis=1)
    # add new columns for guard_variable, assignments_variable, guard_source, assignments_source
    new_columns = ['guard_variable', 'assignments_variable', 'guard_source', 'assignments_source', 'guard_rq_source', 'assignments_rq_source']
    for i in new_columns:
        df_edge[i] = np.empty((len(df_edge), 0)).tolist()

    # fill in the variables in the new created columns
    df_edge = Match_V(df_edge, df_variable, 'variable', 'guard', new_columns[0])
    df_edge = Match_V(df_edge, df_variable, 'variable', 'assignments', new_columns[1])
    # plants
    df_edge = Match_V(df_edge, plant, 'name', 'guard', new_columns[2])
    df_edge = Match_V(df_edge, plant, 'name', 'assignments', new_columns[3])
    # requirement_aut
    if requirement_aut is not None:
        df_edge = Match_V(df_edge, requirement_aut, 'name', 'guard', new_columns[4])
        df_edge = Match_V(df_edge, requirement_aut, 'name', 'assignments', new_columns[5])
    else:
        # fill in with nan
        df_edge[new_columns[4]] = df_edge[new_columns[5]].apply(lambda x: np.nan if len(x) == 0 else x)
        df_edge[new_columns[4]] = df_edge[new_columns[5]].apply(lambda x: np.nan if len(x) == 0 else x)

    return df_edge

# make a 2-d  / 3-d plot function for component analysis result
def CA_plot(data, label_data ,text_v, dimension=2, interact=False, note_text = True,figsize=(8, 6), Title="", label_column_name ="label", legend_ticker = True):
    """
    data : component results
    """
    if interact:
        plt.switch_backend('Qt5Agg')
    else:
        pass
    # data label
    categories = np.unique(label_data[label_column_name])
    colors = plt.cm.rainbow(np.linspace(0, 1, len(categories)))  # Define colors outside of the conditional blocks

    if dimension == 2:
        # assert the data is 2-d
        # assert data.shape[1] == 2
        # plot the data
        # fig size
        plt.figure(figsize=figsize)
        for color, category in zip(colors, categories):
            # Select indices belonging to the current category
            idx = np.where(label_data[label_column_name] == category)
            plt.scatter(data[idx, 0], data[idx, 1], edgecolor='w', color=color, label=category)
            if note_text:
            # annotate the points with variable_all['name']
                for i in idx[0]:
                    plt.text(data[i, 0], data[i, 1], text_v['name'][i], color='black', fontsize=9)
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        # add legend
        if legend_ticker == True:
            plt.legend(title='Label', labels=categories)
        plt.title(Title)
        plt.show()
    # 3-d plot
    else:
        # assert the data is 3-d
        assert data.shape[1] == 3
        # 3d scatter plot, with size of plot
        fig = plt.figure(figsize=figsize)
        ax = plt.axes(projection='3d')
        for color, category in zip(colors, categories):
            # Select indices belonging to the current category
            idx = np.where(label_data[label_column_name] == category)
            ax.scatter(data[idx, 0], data[idx, 1], data[idx, 2], edgecolor='w', color=color, label=category)
            if note_text:
                # annotate the points with variable_all['name']
                for i in idx[0]:
                    ax.text(data[i, 0], data[i, 1], data[i, 2], text_v['name'][i], color='black')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        # plt.set_zlabel('Component 3')
        # add legend
        if legend_ticker == True:
            plt.legend(title='Label', labels=categories)
        plt.title(Title)
        plt.show()

def predicate_source(column_name, map_df, uniq_dataframe, replace = False):
    df = map_df.copy()
    print(f"replace in predicate: {replace}")
    # create a new pd.Series
    df_source = pd.Series()
    k = 0
    for i in df[column_name]:
        # if i is nan, or [] then fill in with -1
        
        if (isinstance(i, float) and np.isnan(i)) or i == []:
            i = [-1]
        new_list = []
        if i != [-1]:
            for j in i:
                # if its not uniq_dataframe['id'] then uniq_dataframe[' id']
                if 'id' in uniq_dataframe.columns:
                    new_list.append(uniq_dataframe['id'][uniq_dataframe['name'] == j].values[0])
                else:
                    new_list.append(uniq_dataframe[' id'][uniq_dataframe['name'] == j].values[0])
            df_source[k] = new_list
        else:
            df_source[k] = [-1]
        k += 1
    if replace:
        map_df[column_name] = df_source
    return df_source

def find_uniq(df, column_name):
    uniq = set(label for sublist in df[column_name] for label in sublist if label != -1)
    # convert to 1d array
    uniq = np.array(list(uniq))
    # create a dataframe with uniq
    df_uniq = pd.DataFrame(uniq, columns=['name'])
    # create a new column with index
    df_uniq['id'] = df_uniq.index
    return df_uniq

def label_encod_input(map_df, column_name):
    df = map_df.copy()
    # check if element in df[column_name] is not Nan or empty
    if df[column_name].isnull().any() or df[column_name].empty:
        return False
    else:
        return find_uniq(df, column_name)
    
def label_ecode_give(df_input, column_name, map_frame, replace=False):
    df = df_input.copy()
    for i in column_name:
        if df[i].isnull().any() or df[i].empty:
            continue
        else:
            print(f"replace : {replace}")
            if replace:
                predicate_source(i, df_input, map_frame, replace=replace)
            else:
                print(predicate_source(i, df, map_frame, replace=replace))

def clean_2_int(df, column, type='int'):
    for i in column:
        df[i] = df[i][df[i].notna()].astype(int)
    return df


def make_variable_all(input_variable_list):
    # concatenate dataframe list
    df = pd.DataFrame()
    for i in input_variable_list:
        # concatenate dataframe list
        df = pd.concat([df, i], axis =0, ignore_index=True)
        df.reset_index(drop=True, inplace=True)
    # drop id column
        # drop ' id' or 'id' column
    if ' id' in df.columns:
        df.drop(columns=' id', inplace=True)
    if 'id' in df.columns:
        df.drop(columns='id', inplace=True)
    df['id'] = df.index
    return df
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.cluster.hierarchy import dendrogram

def encode_and_concat(df, column_name):
    mlb = MultiLabelBinarizer()
    # Apply MultiLabelBinarizer to the column
    encoded_data = mlb.fit_transform(df[column_name])
    # Create a DataFrame from the encoded data
    encoded_df = pd.DataFrame(encoded_data, columns=[f"{column_name}_{cls}" for cls in mlb.classes_])
    # Concatenate the new DataFrame with the original one
    return pd.concat([df, encoded_df], axis=1)

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

def filter_file_names(key_words, file_names):
    
    # Filtered list based on key words
    filtered_files = [file for file in file_names if any(key_word in file for key_word in key_words)]
    
    return filtered_files

def name_filter(name, list_name):
    """
    a help function to filter the name of a file to modulize format (a given list)
    ex: 'multi_agent_folder' -> 'multi_agent' or 'variables_phase2.json' -> 'phase2'
    """
    # if any list_name in name
    if any(i in name for i in list_name):
        for i in list_name:
            if i in name:
                return i
    else:
        raise ValueError(f"no {list_name} in {name}")
# create edge mapping 

def edge_map_multi(input_file):
    df = pd.DataFrame()
    # file name in sequence : edge, alphabet, variables, plant, requirements_aut
    # file_list = ['edge', 'alphabet', 'variables', 'plant', 'requirements_aut', 'count']
    edge = filter_file_names(['edge'], list(input_file.keys()))
    alphabet = filter_file_names(['alphabet'], list(input_file.keys()))
    variables = filter_file_names(['variable'], list(input_file.keys()))
    plant = filter_file_names(['plant'], list(input_file.keys()))
    requirements_aut = filter_file_names(['requirements_aut'], list(input_file.keys()))
    requirement_count = filter_file_names(['count'], list(input_file.keys()))
    # edge mapping
    df = edge_mapping(input_file[edge[0]], input_file[alphabet[0]], input_file[variables[0]], input_file[plant[0]], input_file[requirements_aut[0]])
    for i in input_file[requirement_count[0]].columns:
        df[i] = input_file[requirement_count[0]][i].iloc[0]
    # assignment variables 'guard_assignments_variable'
    df['guard_assignments_variable'] = df['guard_variable'] + df['assignments_variable']
    
    return df

def generate_file_paths(data_dict, key_list, inplace_column = 'variables.json'):
    """
    data_dict: dictionary
    key_list: list of files to be loaded
    inplace_column: load json file, default is variables.json, can modify to 'plants.json'
    """
    # Initialize an empty list to store the paths
    paths = []
    df = pd.DataFrame()
    
    # Loop through each key in the provided list
    for key in key_list:
        # Construct the path and append it to the list
        # Check if the key exists in the dictionary to avoid KeyError
        if key in data_dict:
            input_c = filter_file_names([inplace_column], list(data_dict[key].keys()))[0]
            path = data_dict[key][input_c]
            #  concatenate the data frame
            df = pd.concat([df, path], axis=0, ignore_index=True, sort=False)
            paths.append(path)
        else:
            print(f"Warning: Key '{key}' not found in the data dictionary.")
    # reseet 'id' with index
            # drop id column
    if ' id' in df.columns:
        df = df.drop(columns=[' id'])
    df['id'] = df.index
    return df



def Files_to_dict(Title, name_list,Route_dir):
    # next layer
    After_det = 'After_determin'
    name_list.append(After_det)
    if 'AE' in name_list[0]:
        name_list[0], name_list[1] = name_list[1], name_list[0]
    
    data_dict = {}
    for folder in name_list:
        print(f"read folder: {folder}")
        if 'After_determin' not in folder:
            read_dirc = Route_dir / folder
            data_dict[folder] = read_data(Route=read_dirc)
        else:
            if 'AE' in name_list[1]:
                read_dirc = Route_dir / name_list[1] / folder
                data_dict[folder] = read_data(Route=read_dirc)
            else:
                if 'AE' in name_list[0]:
                    read_dirc = Route_dir / name_list[0] / folder
                    data_dict[folder] = read_data(Route=read_dirc)
                else:
                    read_dirc = Route_dir / name_list[2] / folder
                    data_dict[folder] = read_data(Route=read_dirc)
    # data_dict.keys()
    edge_nam1 = filter_file_names(['edges'], list(data_dict[name_list[0]].keys()))[0]
    edge_nam2 = filter_file_names(['edges'], list(data_dict[name_list[1]].keys()))[0]
    edge_nam3 = filter_file_names(['edges'], list(data_dict[name_list[2]].keys()))[0]
    print(f"edges length")
    print(np.shape(data_dict[name_list[0]][edge_nam1])[0], 
        np.shape(data_dict[name_list[1]][edge_nam2])[0], 
        np.shape(data_dict[name_list[2]][edge_nam3])[0])
    
    print(f"read folder: {data_dict.keys()}")
    return data_dict, name_list

def edge_map_result(data_dict, name_list):
    initial = edge_map_multi(data_dict[name_list[0]])
    # add a column name phase to the initial dataframe
    initial['phase'] = 'Initial'
    AE_stage = edge_map_multi(data_dict[name_list[1]])
    AE_stage['phase'] = 'After_enforce_requirement'
    Adapat_guard = edge_map_multi(data_dict[name_list[2]])
    Adapat_guard['phase'] = 'After_adaptive_guards'
    result = pd.concat([initial, AE_stage, Adapat_guard], axis=0, ignore_index=True, sort=False)
    result['phase'].astype('category')
    return result

# Reference the 12_2_PCA_CaseExperiment.ipynb file
def load_layer_folder(route, folder_names, model_name=None):
    data_dict = {}
    model_dict = {}
    model_dict[model_name] = {}
    for i in folder_names:
        model_dict[model_name][i] = {}
        # print(f"read folder {i}")
        # if have content under directory
        if os.listdir(route / i):
            # if there has files ends with .json files
            if any(j.endswith('.json') for j in os.listdir(route / i)):
                data_dict[i] = {}
                for j in os.listdir(route / i):
                    if os.path.isfile(route / i / j):
                        if j.endswith('.json'):
                            data_dict[i][j] = key_word_load((route / i), keyword=j)
            # if edges folder exists
            if os.path.isdir(route / i / 'edges') and model_name is not None:
                for j in os.listdir(route / i / 'edges'):
                    if os.path.isfile(route / i / 'edges' / j):
                        if j.endswith('.json'):
                            model_dict[model_name][i][j] = key_word_load((route / i / 'edges'), keyword=j)
    # return the one that is not empty
    if model_name is not None:
        return model_dict
    else:
        return data_dict
    
def load_var_folder(route, folder_names, models=None):
    var_data = {}
    # folder_names : model1, model2,...
    # models : case name folder
    var_data[models]={}
    for i in folder_names:
        var_data[models][i]={}
        for j in os.listdir(route/ i):
            if 'variables' in j:
                for k in os.listdir(route / i / j):
                    if os.path.isfile(route / i / j / k):
                        if k.endswith('.json'):
                            var_data[models][i][k] = key_word_load((route / i / j), keyword=k)
    return var_data

def var_model2df(case, phase_list, case_list, model):
    variable_model = pd.DataFrame()
    u = 0
    for i in case:
        df_case_var = pd.DataFrame()
        for j in phase_list:
            df_case_var = pd.concat([df_case_var, i[case_list[u]][model][j]['name']], axis=1)
            df_case_var = df_case_var.rename(columns={'name': j})
        df_case_var['case'] = case_list[u]
        variable_model = pd.concat([variable_model, df_case_var], axis=0)
        u += 1
    variable_model.reset_index(drop=True, inplace=True)
    variable_model['model'] = model
    return variable_model


def load_edges(df,case,models):
    """
    df['multi_agent_folder']['model1']['multi_agent_phase1_edges.json']
    df[case][model][phase]
    Iterate the model to concatenate all phases
    """
    # df = df[case].copy()
    edges_all = pd.DataFrame()
    for i in models:
        for j in df[case][i].keys():
            new_df = pd.DataFrame(df[case][i][j])
            new_df['phase'] = j
            new_df['model'] = i
            edges_all = pd.concat([edges_all, new_df], axis=0)
    return edges_all.reset_index(drop=True)

def edge_case_map(df_case, var, check_col, new_col, case, replace=False):
    df = df_case.copy()
    a_case = df_case[df_case['case'] == case]
    a_var = var[var['case'] == case]
    a_case[new_col] = np.empty((len(a_case), 0)).tolist()
    df = Match_V(a_case, a_var, 'name', check_col, new_col)
    if replace:
        return df
    else:
        df = df[new_col]
        return df


def edge_mapping_new(df, event, plant, requirement_aut):
    df_edge = df.copy()
    
    new_columns = ['guard_source', 'assignments_source', 'guard_rq_source', 'assignments_rq_source']
    for i in new_columns:
        df_edge[i] = np.empty((len(df_edge), 0)).tolist()
    # plants source
    df_edge = Match_V(df_edge, plant, 'name', 'guard', new_columns[0])
    df_edge = Match_V(df_edge, plant, 'name', 'assignments', new_columns[1])
    # requirement_aut source
    if requirement_aut is not None:
        df_edge = Match_V(df_edge, requirement_aut, 'name', 'guard', new_columns[2])
        df_edge = Match_V(df_edge, requirement_aut, 'name', 'assignments', new_columns[3])
    else:
        # fill in with nan
        df_edge[new_columns[4]] = df_edge[new_columns[5]].apply(lambda x: np.nan if len(x) == 0 else x)
        df_edge[new_columns[4]] = df_edge[new_columns[5]].apply(lambda x: np.nan if len(x) == 0 else x)

    return df_edge

def check_controllability(event_code, B):
    # Find the event code in B and return the corresponding 'controllable' value
    for name, controllable in zip(B['name'], B['controllable']):
        if name in event_code:
            return controllable
        
def check_controllability_rq(event_code, B):
    # Find the event code in B and return the corresponding 'controllable' value
    for name, controllable in zip(B['name'], B['requirement_event']):
        if name in event_code:
            return controllable
        
def predicate_source_var(column_name, map_df, uniq_dataframe,column_atr , replace = False):
    df = map_df.copy()
    # create a new pd.Series
    df_source = pd.Series()
    k = 0
    for i in df[column_name]:
        # if i is nan, or [] then fill in with -1
        if (isinstance(i, float) and np.isnan(i)) or i == []:
            i = [-1]
        new_list = []
        if i != [-1]:
            for j in i:
                # if its not uniq_dataframe['id'] then uniq_dataframe[' id']
                if 'id' in uniq_dataframe.columns:
                    new_list.append(uniq_dataframe['id'][uniq_dataframe[column_atr] == j].values[0])
                else:
                    new_list.append(uniq_dataframe[' id'][uniq_dataframe[column_atr] == j].values[0])
            if len(new_list) > 0:
                df_source[k] = np.unique(new_list)
            if len(new_list) == 1:
                df_source[k] = new_list
        else:
            df_source[k] = [-1]
        k += 1
    if replace:
        for i in range(len(map_df[column_name])):
            if len(df_source[i]) > 0:
                if len(np.unique(df_source[i])) == 1:
                    map_df[column_name].iloc[i] = [np.unique(df_source[i])]
                else:
                    map_df[column_name].iloc[i] = np.unique(df_source[i])
            if len(df_source[i]) == 1:
                # type(df_source[i]) is np.array, then convert to list
                if type(df_source[i]) is not list:
                    map_df[column_name].iloc[i] = list(df_source[i])
                else:
                    map_df[column_name].iloc[i] = df_source[i]
            # map_df[column_name].iloc[i] = np.unique(df_source[i])
    return df_source

def select_var_id(var_df, phase_item, model_item, case_item):
    """
    var_df: variable dataframe
    phase_item: phase name: 'phase1'...
    model_item: model name: 'model2'...
    case_item: case name: 'Wolf'...
    @ return : dataframe of columns:['model2_phase1', 'id', 'case']
    """
    # select case_item
    c = var_df[var_df['case'].str.contains(case_item)]
    mask_phase = c.columns.str.contains(phase_item)
    select_1 = c.columns[mask_phase].tolist()
    select_df = c[select_1]
    mask_model = c[select_1].columns.str.contains(model_item)
    select_2 = select_df.columns[mask_model].tolist() + ['id', 'case']
    
    return c[select_2]

def name_filter(name, list_name):
    """
    a help function to filter the name of a file to modulize format (a given list)
    key words: 'name'
    list_name: ['name', 'variables', 'plant', 'requirements_aut']
    @ return : name
    ex: 'multi_agent_folder' -> 'multi_agent' or 'variables_phase2.json' -> 'phase2'
    """
    # if any list_name in name
    if any(i in name for i in list_name):
        for i in list_name:
            if i in name:
                return i
    else:
        raise ValueError(f"no {list_name} in {name}")
    
def get_edges_at_phase(phase_item, case_item, model_item, df):
    data = df.copy()
    # coordinate the case and phase
    export_edge = data[(data['case'].str.contains(case_item)) & (data['phase'].str.contains(phase_item))]['phase'].unique()
    # specific the phase
    export_edge_phase = data[data['phase'] == export_edge[0]] 
    # specific the case
    export_edge_phase = export_edge_phase[export_edge_phase['case'].str.contains(case_item)]
    # specific the model
    return export_edge_phase[export_edge_phase['model'].str.contains(model_item)]



def convert2row(df, model_length=9, array=True):
    """
    Flatten each model in the dataframe into a single row.
    Parameters:
    - df: Input DataFrame with dimensions (n_models * model_length, n_features)
    - model_length: Number of rows representing a single model
    
    Returns:
    if array is true:
        - A NumPy array with each model flattened into a single row
    else:
        - A DataFrame with each model flattened into a single row, keep the input columns name
    """
    # Calculate the number of models in the dataframe
    n_models = df.shape[0] // model_length
    # Initialize an empty list to store the flattened models
    flattened_models = []
    # Iterate through each model and flatten it
    for i in range(n_models):
        start_row = i * model_length
        end_row = start_row + model_length
        model_flat = df.iloc[start_row:end_row].values.flatten()
        flattened_models.append(model_flat)
    if array:
    # Convert the list of flattened models to a NumPy array
        flattened_array = np.array(flattened_models)
        return flattened_array
    else:

        return flattened_models
    
def encod_predicate(df, column_name):
    """
    df: input dataframe
    column_name: column name
    --> import the df[column_name] as pd.Series
    index groups
    ```
    0     ['a','b','c']
    1     ['c']
    2     ['b','c','e']
    3     ['a','c']
    4     ['b','e']
    ```
    into 
    ```
    index  a   b   c   d   e
    0      1   1   1   0   0
    1      0   0   1   0   0
    2      0   1   1   0   1
    3      1   0   1   0   0
    4      0   1   0   0   0
    ```
    """
    return pd.get_dummies(df[column_name].explode()).groupby(level=0).sum()

def rows_columns_select(df, case, case_column_name, model):
    """
    @df: input dataframe
    @case: the case name, key words
    @case_column_name: the column name (the key words in column line)
    @model: the model name, key words, model1, ... (represent the configuration)
    return : dataframe of features with selected rows and columns
    ------
    ex: rows_columns_select(df, 'festo', 'case' ,'model1')
    """
    mask_model = df.columns.str.contains(model)
    columns_select = df.columns[mask_model | (df.columns == case_column_name)]
    rows_select = df[case_column_name].str.contains(case)
    return df.loc[rows_select, columns_select]

import hypernetx as hnx

def model_degree_frame(G1, G2, G3):
    """
    Export each phases' all nodes degree
    """
    degree_nt= pd.DataFrame()
    # G1 nodes and degree
    G1_deg = []
    G2_deg = []
    G3_deg = []
    Node_G1 = list(G1.nodes)
    Node_G2 = list(G2.nodes)
    Node_G3 = list(G3.nodes)
    for i in range(len(list(G1.nodes))):
        G1_deg.append(G1.degree(Node_G1[i], s=1))
    degree_nt['Phase1_degree'] = pd.Series(G1_deg)
    for i in range(len(list(G2.nodes))):
        G2_deg.append(G2.degree(Node_G2[i], s=1))
    degree_nt['Phase2_degree'] = pd.Series(G2_deg)
    for i in range(len(list(G3.nodes))):
        G3_deg.append(G3.degree(Node_G3[i], s=1))
    degree_nt['Phase3_degree'] = pd.Series(G3_deg)
    return degree_nt

def case_degeree(df, case_item, config_item):
    """
    `df`: Edges information
    `case_item`: Case's key words
    `config_item`: which configuration, `model1`, `model2`. ...
    ---
    1. Generate 3 phases' network
    2. Compute the graphs' connecitivity of each nodes using `model_degree_frame`
    3. Returning with a dataframe of degree of all nodes
    """

    df_phase1 = df[df['case'].str.contains(case_item) & df['model'].str.contains(config_item) & df['phase'].str.contains('phase1')][['guard_assignment_variable']]
    df_phase2 = df[df['case'].str.contains(case_item) & df['model'].str.contains(config_item) & df['phase'].str.contains('phase2')][['guard_assignment_variable']]
    df_phase3 = df[df['case'].str.contains(case_item) & df['model'].str.contains(config_item) & df['phase'].str.contains('phase3')][['guard_assignment_variable']]

    G1 = hnx.Hypergraph(df_phase1['guard_assignment_variable'].to_dict())
    G2 = hnx.Hypergraph(df_phase2['guard_assignment_variable'].to_dict())
    G3 = hnx.Hypergraph(df_phase3['guard_assignment_variable'].to_dict())

    return model_degree_frame(G1, G2, G3)