import pickle
import pandas as pd
from strat_models.fit import *
from strat_models.models import *
from strat_models.losses import *
from strat_models.regularizers import *
from strat_models.utils import *
from sklearn.model_selection import train_test_split


df = pd.read_csv('../../preprocessing/standardized_data/final_TEST.csv', low_memory=False)

def load_model(model_path):
    # Open best model
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)

    model = model_data['model']
    edge_weight = model_data['edge_weight']
    loss_name = model_data["loss"]
    reg_name = model_data["reg"]

    print(f"Loss: {loss_name}, Regularizer: {reg_name}, Edge Weight: {edge_weight}")
    return model

def load_G(G_path):
    with open(G_path, 'rb') as f:
        G = pickle.load(f)
    return G

def get_data(df, G):
    df['log_slutpris'] = np.log(df['slutpris'])

    X = []
    Y = []
    Z = []
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

        drop_cols = ['log_slutpris', 'slutpris', 'kommunid', 'lat_bin', 'long_bin', 'maeklarobjektid', 'avtalsdag_full', 'xkoordinat', 'ykoordinat']
        X_node = df_node.drop(drop_cols, axis=1).to_numpy()
        Y_node = df_node['log_slutpris'].to_numpy()
        
        N = X_node.shape[0]
        X.append(X_node)
        Y.append(Y_node)
        Z.extend([(kommunid, (latbin, longbin))] * N)
    
    return np.concatenate(X, axis=0), np.concatenate(Y, axis=0)[:, np.newaxis], Z


def dump_to_csv(model, df, G):
    # Arrange test data
    _, df_test = train_test_split(df)
    X_test, Y_test, Z_test = get_data(df_test, G)
    test_data = dict(X=X_test, Y=Y_test, Z=Z_test)
    
    with torch.no_grad():
        y_pred = model.predict(test_data)

    log_y_pred_np = y_pred.cpu().numpy()
    log_y_true_np = test_data['Y'].cpu().numpy()
    # Transform it back to kr scale
    y_pred_np = np.exp(log_y_pred_np)
    y_true_np = np.exp(log_y_true_np)

    df_predictions = pd.DataFrame({
        'prediction': y_pred_np.flatten(),
        'pris': y_true_np.flatten()
    })

    cols_to_save = df[['maeklarobjektid', 'avtalsdag_full', 'xkoordinat', 'ykoordinat']]
    df_combined = pd.concat([df_predictions, cols_to_save], axis=1)

    df_combined.to_csv('results_from_training/result_file.csv', index=False)

if __name__ == "__main__":
    model_path = 'models/model_2025-04-01_161453.pkl'
    G_path = 'results_from_training/laplacian_graph.pkl'
    
    model = load_model(model_path)
    G = load_G(G_path)
    dump_to_csv(model, df, G)






