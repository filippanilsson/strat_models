import argparse
import glob
import os
import networkx as nx
import pandas as pd
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import json
import numpy as np
import pickle
import random
from datetime import datetime

from strat_models.fit import *
from strat_models.models import *
from strat_models.losses import *
from strat_models.regularizers import *
from strat_models.utils import *


""" Creates G and prepares the edges. For now Z is kontorsid where an edge exists if two offices are in the same kommun.
"""
def create_g(stratify_feature=False, n_bins=50):

    if not stratify_feature: # Stratify over kontorid for now
        G = nx.Graph()
        kontor = pd.read_csv('../../initial_data/kontor_kommuner.csv')

        # Group by 'Kommunnamn' and create pairs of 'KontorId' within each group
        pairs = []
        for _, group in kontor.groupby('Kommunnamn'):
            kontors_ids = group['KontorId'].dropna().tolist()
            pairs.extend(combinations(kontors_ids, 2)) # Create all pairs within the group

        pairs_list = list(pairs)
        return G, pairs_list
    else:
        return nx.grid_2d_graph(n_bins, n_bins), [] # adjust depending on bin size


""" Get data if we using proper binning technique
"""
def get_data(df, G):
    df['log_slutpris'] = np.log(df['slutpris'])

    X = []
    Y = []
    Z = []
    for node in G.nodes():
        latbin, longbin = node
        df_node = df.query('lat_bin == %d & long_bin == %d' % (latbin, longbin))
        X_node = np.array(df_node.drop(['slutpris', 'kommunid', 'lat_bin', 'long_bin'], axis=1))
        Y_node = np.array(df_node['log_slutpris'])
        N = X_node.shape[0]
        X += [X_node]
        Y += [Y_node]
        Z.extend([node] * N)
    
    return np.concatenate(X, axis=0), np.concatenate(Y, axis=0)[:, np.newaxis], Z


""" Splits data into training and testing sets
"""
def split_data(df, G, file='../data.json', test_size=0.2, val_size=0.1, stratify_feature=False):

    if not stratify_feature:
        with open(file, 'r') as f:
            data = json.load(f)

        X = data['X']
        Y = np.array(data['Y']).reshape(-1, 1) # dependent variable slutpris
        Y = np.log1p(Y) # logtransform slutpris for smaller loss
        Z = data['Z'] # stratifying feature

        X_df = pd.DataFrame(X).apply(pd.to_numeric, errors='coerce')  # Convert all columns to numbers
        X_df.columns = X_df.columns.str.strip()

        # Train and test split
        X_train, X_test, Y_train, Y_test, Z_train, Z_test = train_test_split(
            X_df, Y, Z, test_size=test_size, random_state=42
        )

        # Split train into train and validation
        X_train, X_val, Y_train, Y_val, Z_train, Z_val = train_test_split(
            X_train, Y_train, Z_train, test_size=val_size, random_state=42
        )
        
        train_data = {'X': X_train.to_numpy(dtype=np.float64), 'Y': np.array(Y_train, dtype=np.float64), 'Z': np.array(Z_train, dtype=int)}
        val_data = {'X': X_val.to_numpy(dtype=np.float64), 'Y': np.array(Y_val, dtype=np.float64), 'Z': np.array(Z_val, dtype=int)}
        test_data = {'X':  X_test.to_numpy(dtype=np.float64), 'Y': np.array(Y_test, dtype=np.float64), 'Z': np.array(Z_test, dtype=int)}

        return train_data, val_data, test_data
    else:
        df_train, df_test = train_test_split(df)
        df_train, df_val = train_test_split(df_train)

        X_train, Y_train, Z_train = get_data(df_train, G)
        X_test, Y_test, Z_test = get_data(df_test, G)
        X_val, Y_val, Z_val = get_data(df_val, G)

        data_train = dict(X=X_train, Y=Y_train, Z=Z_train)
        data_test = dict(X=X_test, Y=Y_test, Z=Z_test)
        data_val = dict(X=X_val, Y=Y_val, Z=Z_val)

        return data_train, data_val, data_test
        

""" Format Y depending on loss function"""
def format_Y(Y, loss_name):
    if loss_name in ['nonparametric_discrete_loss', 'poisson_loss', 'bernoulli_loss', 'covariance_max_likelihood_loss']:
        return Y.ravel()
    else:
        return Y.reshape(-1, 1)

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
        "sum_squares_loss": sum_squares_loss,
        "logistic_loss": logistic_loss,
        "covariance_max_likelihood_loss": covariance_max_likelihood_loss,
        "nonparametric_discrete_loss": nonparametric_discrete_loss,
        "poisson_loss": poisson_loss,
        "bernoulli_loss": bernoulli_loss
    }

    reg_map = {
        "sum_squares_reg": sum_squares_reg,
        "L2_reg": L2_reg,
        "zero_reg": zero_reg,
        "trace_reg": trace_reg,
        "mtx_scaled_sum_squares_reg": mtx_scaled_sum_squares_reg,
        "mtx_scaled_plus_sum_squares_reg": mtx_scaled_plus_sum_squares_reg,
        "scaled_plus_sum_squares_reg": scaled_plus_sum_squares_reg,
        "L1_reg": L1_reg,
        "elastic_net_reg": elastic_net_reg,
        "neg_log_reg": neg_log_reg,
        "nonnegative_reg": nonnegative_reg,
        "simplex_reg": simplex_reg,
        "min_threshold_reg_one_elem": min_threshold_reg_one_elem,
        "clip_reg": clip_reg,
        "trace_offdiagL1Norm": trace_offdiagL1Norm
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

    return loss_result, reg_result, loss_name, reg_name


""" Creates the Stratified Model and prepares the graph weights
"""
def create_sm(G, loss, reg, train_data, stratifying_feature=False):
    Z_train = train_data['Z']
    bm = BaseModel(loss=loss, reg=reg)

    set_edge_weight(G, 15)
    sm_strat = StratifiedModel(bm, graph=G)

    if not stratifying_feature:
        missing_nodes = set(Z_train) - set(G.nodes)
        if missing_nodes:
            print("Warning: These nodes are missing in G:", missing_nodes)

    return sm_strat

""" Tuning hyperparameters
"""
def tune_hyperparameters(df, G, num_trials=10):
    train_data, val_data, _ = split_data(df, G, stratify_feature=True)

    best_score = float('inf')
    best_params = None

    for _ in range(num_trials):
        # Randomly sample hyperparameters
        reg_type = random.choice(["L2", "sum_squares", "L2", "zero", "trace", "elastic_net_reg","neg_log","nonneg","simplex","min_thres_one","clip","trace_offdiag"])
        loss_type = random.choice(["sum_squares", "logistic"])

        print(f"Trying loss={loss_type}, reg={reg_type}")
        
        loss, reg, loss_name, reg_name = setup_sm(loss_type, reg_type)
        sm_strat = create_sm(G, loss, reg, train_data, stratifying_feature=True)
        kwargs = dict(rel_tol=1e-5, abs_tol=1e-5, maxiter=300, n_jobs=1, verbose=True)
        sm_strat.fit(train_data, **kwargs)
        val_score = sm_strat.scores(val_data)
        print(f"Validation Loss: {val_score}")

        if val_score < best_score:
            best_score = val_score
            best_params = (loss_type, reg_type)

    print(f"\nBest Hyperparameters: Loss={best_params[0]}, Reg={best_params[1]}")
    print(f"Best Validation Loss: {best_score}")
    
    return best_params

""" Save the model"""
def save_model(model, loss_name, reg_name):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    filename = f'models/model_{timestamp}.pkl'

    # Save metadata
    model_data = {
        'model': model,
        'loss': loss,
        'reg': reg,
        'timestamp': timestamp
    }

    with open(filename, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"Model saved as {filename} (Loss: {loss_name}, Reg: {reg_name})")
    return filename

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and test stratified model")
    parser.add_argument("--retrain", action="store_true", help="Retrain the model", default=False, required=False)
    parser.add_argument("--stratify_feature", action="store_true", help="Stratify over lat/long bins", default=True, required=False)
    args = parser.parse_args()

    df = pd.read_csv('../../preprocessing/standardized_data/final.csv')
    if args.stratify_feature:
        print("Stratifying over lat/long bins")
        
        G, _ = create_g(stratify_feature=True)

        data_train, data_val, data_test = split_data(df, G)
        loss, reg, loss_name, reg_name = setup_sm("sum_squares", "L2")

        data_train["Y"] = format_Y(data_train["Y"], loss_name)
        data_val["Y"] = format_Y(data_val["Y"], loss_name)
        data_test["Y"] = format_Y(data_test["Y"], loss_name)

        sm_strat = create_sm(G, loss, reg, data_train, stratifying_feature=True)
    else:
        print("Stratifying over kontorid")
        G, pairs_list = create_g()
        train_data, val_data, test_data = split_data('../data.json')
        finish_g(train_data, G, pairs_list)
        loss, reg, loss_name, reg_name = setup_sm("sum_squares", "L2")
        
        # Format depending on loss function
        train_data["Y"] = format_Y(train_data["Y"], loss_name)
        val_data["Y"] = format_Y(val_data["Y"], loss_name)
        test_data["Y"] = format_Y(test_data["Y"], loss_name)
        
        sm_strat = create_sm(G, loss, reg, train_data)


    kwargs = dict(rel_tol=1e-5, abs_tol=1e-5, maxiter=5000, n_jobs=1, verbose=True)

    from multiprocessing import freeze_support
    freeze_support()

    # Train model
    if args.retrain:
        info = sm_strat.fit(train_data, **kwargs)
    else:
        models = glob.glob('models/*.pkl')
        if not models:
            raise ValueError("No models found to load")
        file = max(models, key=os.path.getctime)
        with open(file, 'rb') as f:
            model_data = pickle.load(f)
            print(f"Model loaded from {file}")
            print(f"Loss Function: {model_data['loss']}")
            print(f"Regularizer: {model_data['reg']}")
            print(f"Trained on: {model_data['timestamp']}")
        sm_strat = model_data['model']
    
    # Test and validate model
    val_score = sm_strat.scores(val_data)
    test_score = sm_strat.scores(test_data)
   
    print("Stratified model")
    if args.retrain:
        print("\t Info =", info)
    print("\t Validation Loss =", val_score)
    print("\t Test Loss =", test_score)

    model_filename = save_model(sm_strat, loss_name, reg_name)
    print(f'Model saved as {model_filename}')


    if args.retrain:
        best_hyperparams = tune_hyperparameters(df, G)
    else:
        sm_strat.eval()

        with torch.no_grad():
            y_pred = sm_strat(data_test['X']) # Get predictions
        
        y_pred_np = y_pred.cpu().numpy()
        y_true_np = data_test['Y'].cpu().numpy()

        rmse = np.sqrt(mean_squared_error(y_true_np, y_pred_np))
        r2 = r2_score(y_true_np, y_pred_np)
        print(f"RMSE: {rmse:.4f}")
        print(f"R2: {r2:.4f}")
