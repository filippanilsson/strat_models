import networkx as nx
import pandas as pd
from itertools import combinations
from sklearn.model_selection import train_test_split
import json
import numpy as np

from strat_models.fit import *
from strat_models.models import *
from strat_models.losses import *
from strat_models.regularizers import *
from strat_models.utils import *


""" Creates G and prepares the edges. For now Z is kontorsid where an edge exists if two offices are in the same kommun.
"""
def create_g(stratify_feature=False):
    G = nx.Graph()
    if not stratify_feature: # Stratify over kontorid for now
        kontor = pd.read_csv('../../initial_data/kontor_kommuner.csv')

        # Group by 'Kommunnamn' and create pairs of 'KontorId' within each group
        pairs = []
        for _, group in kontor.groupby('Kommunnamn'):
            kontors_ids = group['KontorId'].dropna().tolist()
            pairs.extend(combinations(kontors_ids, 2)) # Create all pairs within the group

        pairs_list = list(pairs)
        return G, pairs_list


""" Splits data into training and testing sets
"""
def split_data(file):
    with open(file, 'r') as f:
        data = json.load(f)

    X = data['X']
    Y = np.array(data['Y']).reshape(-1, 1) # dependent variable slutpris
    Y = np.log1p(Y) # logtransform slutpris for smaller loss
    Z = data['Z'] # stratifying feature

    X_df = pd.DataFrame(X)
    # Make sure everything is numeric
    X_df = X_df.apply(pd.to_numeric, errors='coerce')
    X_df.columns = X_df.columns.str.strip()

    X_train, X_test, Y_train, Y_test, Z_train, Z_test = train_test_split(
        X_df, Y, Z, test_size=0.2, random_state=42
    )
    
    train_data = {'X': X_train.to_numpy(dtype=np.float64), 'Y': np.array(Y_train, dtype=np.float64), 'Z': np.array(Z_train, dtype=int)}
    test_data = {'X':  X_test.to_numpy(dtype=np.float64), 'Y': np.array(Y_test, dtype=np.float64), 'Z': np.array(Z_test, dtype=int)}

    return train_data, test_data


""" Adds nodes and edges to G
"""
def finish_g(train_data, G, pairs_list, stratifying_feature=False):
    Z_train = train_data['Z']
    if not stratifying_feature:
        kontor = pd.read_csv('../../initial_data/kontor_kommuner.csv')
        kontorid = set(kontor['KontorId'].dropna().tolist()).union(set(Z_train))
        # Add nodes and edges
        G.add_nodes_from(kontorid)
        G.add_edges_from(pairs_list)


""" Sets the loss and regularizer functions + parameters from the arguments by reading from the config file.
"""
def setup_sm(loss, reg):
    with open('config.json', 'r') as f:
       config = json.load(f)

    loss_functions_map = {
        "sum_squares_loss": sum_squares_loss
    }

    reg_map = {
        "sum_squares_reg": sum_squares_reg,
        "L2_reg": L2_reg
    }
   
    if loss not in config["loss_functions"]:
        raise ValueError(f"Invalid loss function type: {loss}")
    loss_info = config["loss_functions"][loss]
    loss_name = loss_info["name"] # Get loss function name
    loss_params = loss_info.get("params", {}) # Get parameters

    loss_f = loss_functions_map.get(loss_name)
    loss_result = loss_f(**loss_params) if loss_f else None

    if reg not in config["regularizers"]:
        raise ValueError(f"Invalid regularizer function type: {reg}")
    reg_info = config["regularizers"][reg]
    reg_name = reg_info["name"] 
    reg_params = reg_info.get("params", {})

    reg_f = reg_map.get(reg_name)
    reg_result = reg_f(**reg_params) if reg_f else None

    return loss_result, reg_result


""" Creates the Stratified Model and prepares the graph weights
"""
def create_sm(G, loss, reg, train_data):
    Z_train = train_data['Z']
    bm = BaseModel(loss=loss, reg=reg)

    set_edge_weight(G, 2)
    sm_strat = StratifiedModel(bm, graph=G)

    missing_nodes = set(Z_train) - set(G.nodes)
    if missing_nodes:
        print("Warning: These nodes are missing in G:", missing_nodes)

    return sm_strat


if __name__ == "__main__":
    G, pairs_list = create_g()
    train_data, test_data = split_data('../data.json')
    finish_g(train_data, G, pairs_list)
    loss, reg = setup_sm("sum_squares", "L2")
    
    sm_strat = create_sm(G, loss, reg, train_data)
    kwargs = dict(rel_tol=1e-5, abs_tol=1e-5, maxiter=5000, n_jobs=1, verbose=True)

    from multiprocessing import freeze_support
    freeze_support()

    info = sm_strat.fit(train_data, **kwargs)
    score = sm_strat.scores(test_data)
    print("Stratified model")
    print("\t Info =", info)
    print("\t Loss =", score)

