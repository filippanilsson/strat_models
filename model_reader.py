import pickle
import pandas as pd
import numpy as np

#result = pd.read_csv('../results_from_training/result_file_2025-04-14')
with open('models/model_2025-05-05_145907_GLOBAL.pkl', 'rb') as f:
    model = pickle.load(f)

loss = model['loss']
reg = model['reg']
edge_weights = model['weight']

print("Loss:", loss)
print("Regularization:", reg)
print("Edge Weights:", edge_weights)