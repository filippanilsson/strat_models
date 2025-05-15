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


""" Builds the Laplacian graph G, putting an edge between nodes if they are
    adjacent in the same kommun
"""
def build_G(df):
    G = nx.Graph()
    for kommun in df['kommunid'].unique():
        df_kommun = df[df['kommunid'] == kommun] # Filter data for this kommun
        bins = set(zip(df_kommun['lat_bin'], df_kommun['long_bin'])) # Get unique bins for this kommun

        # Add nodes to the graph
        for node in bins:
            G.add_node((kommun, node)) # Node = (kommunid, (latbin, longbin))
        
        # Fully connect all bins within the same kommun
        bins_list = list(bins)
        for i in range(len(bins_list)):
            for j in range(i + 1, len(bins_list)):
                G.add_edge((kommun, bins_list[i]), (kommun, bins_list[j]))
    
    with open('results_from_training/bugfix_graph_TEST.pkl', 'wb') as f:
        pickle.dump(G, f)
    print("Saved Laplacian graph to 'results_from_training/bugfix_graph_TEST.pkl' ")
    return G

""" Get data if we using proper binning technique
"""
def get_data(df, G):
    df['log_slutpris'] = np.log(df['slutpris'])

    X = []
    Y = []
    Z = []
    meta = []
    for node in G.nodes():
        kommunid, (latbin, longbin) = node

        # Query data for this specific kommun and bin
        df_node = df.loc[
                    (df['kommunid'] == kommunid) & 
                    (df['lat_bin'] == latbin) & 
                    (df['long_bin'] == longbin)
                    ]
        if df_node.empty:
            continue

        X_node = np.array(df_node.drop(['log_slutpris', 'slutpris', 'kommunid', 'lat_bin', 'long_bin', 'maeklarobjektid', 'avtalsdag_full'], axis=1))
        Y_node = np.array(df_node['log_slutpris'])
        
        N = X_node.shape[0]
        X += [X_node]
        Y += [Y_node]
        Z.extend([(kommunid, (latbin, longbin))] * N)
        meta.append(df_node[['kommunid','xkoordinat','ykoordinat','avtalsdag_full','maeklarobjektid']])
    
    save_cols_filtered = pd.concat(meta, ignore_index=True)
    return np.concatenate(X, axis=0), np.concatenate(Y, axis=0)[:, np.newaxis], Z, save_cols_filtered


""" Splits data into training and testing sets
"""
def split_data(df, G):
    print("ELLO MATE")
    df_train, df_test = train_test_split(df)
    print(f"df_train size: {len(df_train)}, df_test size: {len(df_test)}")
    df_train, df_val = train_test_split(df_train)
    print(f"df_train size: {len(df_train)}, df_val size: {len(df_val)}")
    X_train, Y_train, Z_train, _ = get_data(df_train, G)
    X_test, Y_test, Z_test, save_cols = get_data(df_test, G)
    X_val, Y_val, Z_val, _ = get_data(df_val, G)

    data_train = dict(X=X_train, Y=Y_train, Z=Z_train)
    data_test = dict(X=X_test, Y=Y_test, Z=Z_test)
    data_val = dict(X=X_val, Y=Y_val, Z=Z_val)

    return data_train, data_val, data_test, save_cols
        

""" Format Y depending on loss function"""
def format_Y(Y, loss_name):
    if loss_name in ['nonparametric_discrete_loss', 'poisson_loss', 'bernoulli_loss', 'covariance_max_likelihood_loss']:
        return Y.ravel()
    else:
        return Y.reshape(-1, 1)

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
        "L2_reg": L2_reg,
        "zero_reg": zero_reg,
        "trace_reg": trace_reg,
        "scaled_plus_sum_squares_reg": scaled_plus_sum_squares_reg,
        "L1_reg": L1_reg,
        "elastic_net_reg": elastic_net_reg,
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
def create_sm(G, loss, reg, train_data):
    Z_train = train_data['Z']
    bm = BaseModel(loss=loss, reg=reg)

    set_edge_weight(G, 0.1)
    sm_strat = StratifiedModel(bm, graph=G)
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

"""
    Generates a result file
"""
def gen_result_csv(y_pred, y_true, saved_cols):
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()

    results = saved_cols.copy()
    results['slutpris (kr)'] = np.exp(y_true)
    results['prediction (kr)'] = np.exp(y_pred)
    # Percentual difference between actual and predicted value:
    results['diff (%)'] = (
    (results['prediction (kr)'] - results['slutpris (kr)']) / results['slutpris (kr)']
        ) * 100
    
    results["diff (%)"] = results["diff (%)"].round(2)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    results.to_csv(f'results_from_training/result_file_{timestamp}.csv', index=False)
    print("Successfully generated result file")

""" Save the model"""
def save_model(model, loss_name, reg_name, weight=0.1):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    filename = f'models/model_{timestamp}.pkl'

    # Save metadata
    model_data = {
        'model': model,
        'loss': loss,
        'reg': reg,
        'timestamp': timestamp,
        'weight': weight
    }

    with open(filename, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"Model saved as {filename} (Loss: {loss_name}, Reg: {reg_name})")
    return filename

"""
    Picks out 10 municipalities each from the:
        - top 33% most populated
        - middle 33% most populated
        - bottom 33% most populated
"""
def split_subset_data(df):
    # Split into three density buckets - top 33%, middle 33% and bottom 33%
    counts = df['kommunid'].value_counts().reset_index()
    counts.columns = ['kommunid', 'sample_count']
    counts = counts[counts['sample_count'] >= 100] # Filter out very unpopulated ones

    # Add quantile-based buckets
    counts['density_bucket'] = pd.qcut(counts['sample_count'], 3, labels=['low', 'medium', 'high'])
    selected_ids = counts.groupby('density_bucket').sample(n=10)['kommunid'].tolist()
    df_subset = df[df['kommunid'].isin(selected_ids)]

    return df_subset       

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and test stratified model")
    parser.add_argument("--retrain", action="store_true", help="Retrain the model", default=False, required=False)
    args = parser.parse_args()

    df = pd.read_csv('../../preprocessing/standardized_data/bugfix_final.csv')
    df = split_subset_data(df) # Get a subset of the data for testing
    print("Stratifying over lat/long bins")
    
    G = build_G(df)

    train_data, val_data, test_data, save_cols = split_data(df, G)
    loss, reg, loss_name, reg_name = setup_sm("sum_squares", "L2")

    train_data["Y"] = format_Y(train_data["Y"], loss_name)
    val_data["Y"] = format_Y(val_data["Y"], loss_name)
    test_data["Y"] = format_Y(test_data["Y"], loss_name)

    sm_strat = create_sm(G, loss, reg, train_data)
    Z = train_data['Z']
    print("Graph Nodes:", list(G.nodes())[:10])  # Print first 10 nodes
    print("First 10 Z values:", Z[:10])  # Compare to Z


    kwargs = dict(rel_tol=1e-5, abs_tol=1e-5, maxiter=300, n_jobs=1, verbose=True)

    from multiprocessing import freeze_support
    freeze_support()

    # Train model
    info = sm_strat.fit(train_data, **kwargs)
    print("Training complete")
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

    with torch.no_grad():
        y_pred = sm_strat.predict(test_data) # Get predictions
    
    #y_pred_np = y_pred.numpy()
    y_true_np = test_data['Y']

    rmse = np.sqrt(mean_squared_error(y_true_np, y_pred))
    r2 = r2_score(y_true_np, y_pred)
    print(f"RMSE: {rmse:.4f}")
    print(f"R2: {r2:.4f}")

    gen_result_csv(y_pred, y_true_np, save_cols)