"""
 Copyright 2025 Filippa Nilsson
 Based on work copyrighted by Jonathan Tuck under Apache License 2.0

 This file makes use of functions/classes from the original repository but is independently developed.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at:

     https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and limitations under the License.
"""

import argparse
import glob
import os
import networkx as nx
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
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
def build_G(df, test):
    G = nx.Graph()
    for kommun in df['kommunid'].unique():
        df_kommun = df[df['kommunid'] == kommun] # Filter data for this kommun
        bins = set(zip(df_kommun['lat_bin'], df_kommun['long_bin'])) # Get unique bins for this kommun

        # Add nodes to the graph
        for node in bins:
            G.add_node((kommun, node)) # Node = (kommunid, (latbin, longbin))

        # Fully connect all bins within the same kommun, use this for now and check performance of graphs, maybe do based
        # on adjacency later :)
        bins_list = list(bins)
        for i in range(len(bins_list)):
            for j in range(i + 1, len(bins_list)):
                G.add_edge((kommun, bins_list[i]), (kommun, bins_list[j]))

        mode = None
        # Option 1: Use only spatial adjacency across all bins
        if mode is not None:
            df['node'] = list(zip(df['kommunid'], zip(df['lat_bin'], df['long_bin'])))
            unique_nodes = df['node'].unique()
            bin_to_nodes = {}
            for kommunid, (latbin, longbin) in unique_nodes:
                bin_key = (latbin, longbin)
                bin_to_nodes.setdefault(bin_key, []).append((kommunid, (latbin, longbin)))

            # Add edges between neighboring bins
            for (latbin, longbin), nodes_in_bin in bin_to_nodes.items():
                neighbors = [
                    (latbin + 1, longbin),
                    (latbin - 1, longbin),
                    (latbin, longbin + 1),
                    (latbin, longbin - 1)
                ]
                for neighbor_bin in neighbors:
                    if neighbor_bin in bin_to_nodes:
                        for node_a in nodes_in_bin:
                            for node_b in bin_to_nodes[neighbor_bin]:
                                G.add_edge(node_a, node_b)
            
            # Optional: add full intra-kommun connectivity
            # Meaning, all nodes within a kommun are connected, as well as neighboring bins across kommuner
            # lambda will be fixed, no need to weight the edges differently for simplicity
            if mode == 'adjacent+intra':
                for kommun in df['kommun'].unique():
                    kommun_nodes = [node for node in G.nodes if node[0] == kommun]
                    for i in range(len(kommun_nodes)):
                        for j in range(i + 1, len(kommun_nodes)):
                            G.add_edge(kommun_nodes[i], kommun_nodes[j])
    # Ensure the directory exists
    os.makedirs('results_from_training', exist_ok=True) 
    
    if test:
        with open('results_from_training/laplacian_graph_TEST.pkl', 'wb') as f:
            pickle.dump(G, f)
        print("Saved Laplacian graph to 'results_from_training/laplacian_graph_TEST.pkl' ")
    else:
        with open('results_from_training/laplacian_graph.pkl', 'wb') as f:
            pickle.dump(G, f)
        print("Saved Laplacian graph to 'results_from_training/laplacian_graph.pkl'")
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

        X_node = np.array(df_node.drop(['slutpris', 'log_slutpris','kommunid', 'lat_bin', 'long_bin', 'maeklarobjektid', 'avtalsdag_full', 'xkoordinat', 'ykoordinat'], axis=1))
        Y_node = np.array(df_node['log_slutpris'])
        
        N = X_node.shape[0]
        X += [X_node]
        Y += [Y_node]
        Z.extend([(kommunid, (latbin, longbin))] * N)
        meta.append(df_node[['kommunid', 'maeklarobjektid', 'avtalsdag_full', 'xkoordinat', 'ykoordinat']])
    
    save_cols_filtered = pd.concat(meta, ignore_index=True)
    return np.concatenate(X, axis=0), np.concatenate(Y, axis=0)[:, np.newaxis], Z, save_cols_filtered


""" 
    Splits data into training and testing sets
"""
def split_data(df, G, test_size=0.2, val_size=0.1):
    print("Splitting data...")
    df_train, df_test = train_test_split(df)
    print(f"df_train size: {len(df_train)}, df_test size: {len(df_test)}")
    df_train, df_val = train_test_split(df_train)
    print(f"df_train size: {len(df_train)}, df_val size: {len(df_val)}")
    X_train, Y_train, Z_train, _ = get_data(df_train, G)
    X_test, Y_test, Z_test, save_cols  = get_data(df_test, G)
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
        "sum_squares_loss": sum_squares_loss,
    }

    reg_map = {
        "sum_squares_reg": sum_squares_reg,
        "L2_reg": L2_reg,
        "zero_reg": zero_reg,
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
def create_sm(G, loss, reg, train_data, weight=15):
    Z_train = train_data['Z']
    bm = BaseModel(loss=loss, reg=reg)

    set_edge_weight(G, weight)
    sm_strat = StratifiedModel(bm, graph=G)
    
    return sm_strat


""" Tuning hyperparameters
"""
def tune_hyperparameters(df, G, num_trials=10):
    train_data, val_data, _, _ = split_data(df, G, stratify_feature=True)

    best_score = float('inf')
    best_params = None

    for _ in range(num_trials):
        # Randomly sample hyperparameters
        reg_type = random.choice(["L2", "sum_squares", "L2", "zero", "elastic_net_reg", "scaled_plus_sum_squares_reg"])
        loss_type = random.choice(["sum_squares"])

        print(f"Trying loss={loss_type}, reg={reg_type}")
        
        loss, reg, loss_name, reg_name = setup_sm(loss_type, reg_type)
        sm_strat = create_sm(G, loss, reg, train_data)
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
def save_model(model, loss_name, reg_name, weight=15):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    filename = f'models/model_{timestamp}.pkl'

    # Save metadata
    model_data = {
        'model': model,
        'loss': loss_name,
        'reg': reg_name,
        'timestamp': timestamp,
        'edge_weight': weight
    }

    with open(filename, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"Model saved as {filename} (Loss: {loss_name}, Reg: {reg_name})")
    return filename

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

"""
    Tunes the edge weights of the graph G,
    using k-fold cross-validation to find the best edge weight
    based on the average RMSE across folds.
"""
def tune_edge_weight(df, G, weight_candidates, loss, reg, loss_name, reg_name):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    best_avg_rmse = float('inf')
    best_weight = None
    kwargs = dict(rel_tol=1e-5, abs_tol=1e-5, maxiter=300, n_jobs=1, verbose=True)

    results = []
    for weight in weight_candidates:
        fold_rms_errors = []
        print(f"Trying edge weight: {weight}")

        for fold, (train_index, val_index) in enumerate(kf.split(df), 1):
            df_train = df.iloc[train_index].copy()
            df_val = df.iloc[val_index].copy()

            # Prepare training and validationd data
            X_train, Y_train, Z_train = get_data(df_train, G)
            X_val, Y_val, Z_val = get_data(df_val, G)
            train_data = dict(X=X_train, Y=Y_train, Z=Z_train)
            val_data = dict(X=X_val, Y=Y_val, Z=Z_val)

            model = create_sm(G, loss, reg, train_data, weight=weight)
            model.fit(train_data, **kwargs)

            y_pred = model.predict(val_data)
            rmse = np.sqrt(mean_squared_error(Y_val, y_pred))
            fold_rms_errors.append(rmse)
            print(f"Fold {fold} RMSE: {rmse:.6f}")
        
        avg_rmse = np.mean(rmse)
        results.append({'edge_weight': weight, 'avg_rmse': avg_rmse})
        print(f"Average Validation Loss for weight {weight}: {avg_rmse:.6f}")

        if avg_rmse < best_avg_rmse:
            best_avg_rmse = avg_rmse
            best_weight = weight
            print(f"New best weight: {best_weight} with average RMSE: {best_avg_rmse:.6f}")
    
    results_df = pd.DataFrame(results)
    results_df.to_csv('results_from_training/edge_weights_RMSE_2.csv', index=False)
    print(f"Best edge weight: {best_weight} based on 5-fold CV with average RMSE: {best_avg_rmse:.6f}")
    
    """ 
        for weight in weight_candidates:
            print(f"Trying edge weight: {weight}")
            model = create_sm(G, loss, reg, train_data, weight)
            # Fit the model
            
            model.fit(train_data, **kwargs)
            score = model.scores(val_data)
            print(f"Validation Loss for weight {weight}: {score}")
            if score < best_score:
                best_score = score
                best_weight = weight
            
        print(f"Best edge weight: {best_weight} with validation loss: {best_score}") 
        """
    model_filename = save_model(model, loss_name, reg_name, best_weight)
    print(f'Model saved as {model_filename}')
    return best_weight

"""
    Retrain the model from scratch
"""
def retrain(df, test, regraph):
    
    if regraph:
        print("Building Laplacian graph...")
        G = build_G(df, args.test)
    else:
        if test:
            G_filepath = 'results_from_training/laplacian_graph_TEST.pkl'
        else:
            G_filepath = 'results_from_training/laplacian_graph.pkl'
        print(f"Loading graph from {G_filepath}")
        with open(G_filepath, 'rb') as f:
            G = pickle.load(f)

    
    print('Splitting data...')
    train_data, val_data, test_data, save_cols = split_data(df, G)
    print('Creating Stratified Model')
    loss, reg, loss_name, reg_name = setup_sm("sum_squares", "L2")

    train_data["Y"] = format_Y(train_data["Y"], loss_name)
    val_data["Y"] = format_Y(val_data["Y"], loss_name)
    test_data["Y"] = format_Y(test_data["Y"], loss_name)

    sm_strat = create_sm(G, loss, reg, train_data, 0.1)
    Z = train_data['Z']
   
    kwargs = dict(rel_tol=1e-5, abs_tol=1e-5, maxiter=1000, n_jobs=1, verbose=True)
    from multiprocessing import freeze_support
    freeze_support()

    print('Retraining the model from scratch...')
    info = sm_strat.fit(train_data, **kwargs)
    model_filename = save_model(sm_strat, loss_name, reg_name)
    print(f'Model saved as {model_filename}')

    # Test and validate model
    val_score = sm_strat.scores(val_data)
    test_score = sm_strat.scores(test_data)
   
    print("Stratified model")
    print("\t Info =", info)
    print("\t Validation Loss =", val_score)
    print("\t Test Loss =", test_score)

    # Now we evaluate the new model
    print("Evaluating performance...")
   
    with torch.no_grad():
        y_pred = sm_strat.predict(test_data) # Get predictions
    
    y_pred_np = y_pred
    y_true_np = test_data['Y']

    rmse = np.sqrt(mean_squared_error(y_true_np, y_pred_np))
    r2 = r2_score(y_true_np, y_pred_np)
    print(f"RMSE: {rmse:.4f}")
    print(f"R2: {r2:.4f}")

    gen_result_csv(y_pred_np, y_true_np, save_cols)

"""
    Tune hyperparameters with k-fold cross-validation
"""
def tune_model(df, edges, test, hyper):
    if test:
        G_filepath = 'results_from_training/laplacian_graph_TEST.pkl'
    else:
        G_filepath = 'results_from_training/laplacian_graph.pkl'
    print(f"Loading graph from {G_filepath}")
    with open(G_filepath, 'rb') as f:
        G = pickle.load(f)
    

    print('Splitting data...')
    train_data, val_data, test_data, _ = split_data(df, G)
    print('Creating Stratified Model')
    loss, reg, loss_name, reg_name = setup_sm("sum_squares", "L2")

    train_data["Y"] = format_Y(train_data["Y"], loss_name)
    val_data["Y"] = format_Y(val_data["Y"], loss_name)
    test_data["Y"] = format_Y(test_data["Y"], loss_name)

    if edges:
        print("Tuning edge weights...")
        weight_candidates = [0.1, 5, 10, 15, 20, 25]
        best_weight = tune_edge_weight(df, G, weight_candidates, loss, reg, loss_name, reg_name)
        print(f"Best edge weight: {best_weight}")

    if hyper:
        results = []
        best_score = float('inf')
        best_params = None

        for _ in range(10):
            # Randomly sample hyperparameters
            reg_type = random.choice(["L2", "sum_squares", "zero", "elastic_net", "scaled_plus_sum_squares"])
            loss_type = random.choice(["sum_squares"])

            print(f"Trying loss={loss_type}, reg={reg_type}")
            
            loss, reg, loss_name, reg_name = setup_sm(loss_type, reg_type)
            sm_strat = create_sm(G, loss, reg, train_data)
            kwargs = dict(rel_tol=1e-5, abs_tol=1e-5, maxiter=300, n_jobs=1, verbose=True)
            sm_strat.fit(train_data, **kwargs)
            val_score = sm_strat.scores(val_data)
            print(f"Validation Loss: {val_score}")
            results.append({'loss': loss, 'reg': reg, 'score': val_score})
            
            if val_score < best_score:
                best_score = val_score
                best_params = (loss_type, reg_type)
        
        results_df = pd.DataFrame(results)
        results_df.to_csv('results_from_training/hyperparameters_RMSE.csv', index=False)
        print(f"Best Validation Loss: {best_score}")
        loss_name = best_params[0]
        reg_name = best_params[1]
        model_filename = save_model(sm_strat, loss_name, reg_name)
        print(f"Model saved as {model_filename}")

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and test stratified model")
    parser.add_argument("--retrain", action="store_true", help="Retrain the model", default=False, required=False)
    parser.add_argument('--regraph', action="store_true", help="Regenerate the Laplacian graph G", default=False)
    parser.add_argument("--test", action="store_true", help="Test the training on a subset of data to see if the pipeline works", default=False)
    parser.add_argument("--edges", action="store_true", help="Tune edge weights for Laplacian graph", default=False)
    parser.add_argument("--hyper", action="store_true", help="Tune hyperparameters", default=False)
    args = parser.parse_args()

    print('Loading data...')
    df = pd.read_csv('../../preprocessing/standardized_data/final_TEST.csv')

    if args.regraph:
        print("Building Laplacian graph...")
        G = build_G(df, args.test)
    

    if args.test:
        print("TEST MODE ENABLED - Using top 3 municipalities only")
        top_kommuner = df['kommunid'].value_counts().nlargest(3).index.to_list()
        df = df[df['kommunid'].isin(top_kommuner)].copy()
        print(f"Selected kommuner: {top_kommuner}")
        print(f"Subset size: {len(df)} rows")

    if args.retrain:
        retrain(df, args.test, args.regraph)

    if args.edges:
        tune_model(df, args.edges, args.test, args.hyper)

    if args.hyper:
        tune_model(df, args.edges, args.test, args.hyper)
    """ if args.test:
        print("TEST MODE ENABLED - Using top 3 municipalities only")
        top_kommuner = df['kommunid'].value_counts().nlargest(3).index.to_list()
        df = df[df['kommunid'].isin(top_kommuner)].copy()
        print(f"Selected kommuner: {top_kommuner}")
        print(f"Subset size: {len(df)} rows")
    
    if args.regraph:
        print("Building Laplacian graph...")
        G = build_G(df, args.test)
    else:
        if args.test:
            G_filepath = 'results_from_training/laplacian_graph_TEST.pkl'
        else:
            G_filepath = 'results_from_training/laplacian_graph.pkl'
        print(f"Loading graph from {G_filepath}")
        with open(G_filepath, 'rb') as f:
            G = pickle.load(f)

    print('Splitting data...')
    train_data, val_data, test_data = split_data(df, G)
    print('Creating Stratified Model')
    loss, reg, loss_name, reg_name = setup_sm("sum_squares", "L2")

    train_data["Y"] = format_Y(train_data["Y"], loss_name)
    val_data["Y"] = format_Y(val_data["Y"], loss_name)
    test_data["Y"] = format_Y(test_data["Y"], loss_name)

    sm_strat = create_sm(G, loss, reg, train_data)
    Z = train_data['Z']
   
    kwargs = dict(rel_tol=1e-5, abs_tol=1e-5, maxiter=1000, n_jobs=1, verbose=True)

    from multiprocessing import freeze_support
    freeze_support()

   # Train model
    if args.retrain:
        print('Retraining the model from scratch...')
        info = sm_strat.fit(train_data, **kwargs)
        model_filename = save_model(sm_strat, loss_name, reg_name)
        print(f'Model saved as {model_filename}')
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

    # Now we evaluate the new model
    print("Evaluating performance...")
   
    with torch.no_grad():
        y_pred = sm_strat.predict(test_data) # Get predictions
    
    y_pred_np = y_pred.cpu().numpy()
    y_true_np = test_data['Y'].cpu().numpy()

    rmse = np.sqrt(mean_squared_error(y_true_np, y_pred_np))
    r2 = r2_score(y_true_np, y_pred_np)
    print(f"RMSE: {rmse:.4f}")
    print(f"R2: {r2:.4f}")    
    if not args.retrain:
        # We tune instead of retraining the whole model
        print("Tuning edge weights...")
        weight_candidates = [0.1, 5, 10, 15, 20, 25]
        best_weight = tune_edge_weight(G, weight_candidates, loss, reg, train_data, val_data)
        print(f"Best edge weight: {best_weight}") """
