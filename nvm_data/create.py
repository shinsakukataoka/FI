import os
import pickle
import numpy as np

# Function to normalize means between 0 and 1
def normalize(data):
    min_val = min(pair[0] for sublist in data for pair in sublist)
    max_val = max(pair[0] for sublist in data for pair in sublist)
    range_val = max_val - min_val

    normalized_data = [
        [( (pair[0]-min_val)/range_val, pair[1] ) for pair in sublist]
        for sublist in data
    ]
    return normalized_data

# Your data to be pickled
fe_de_fe_sl_hzo = normalize([
    [(-0.36,0.02),(1.58,0.03)],
    [(-0.36,0.02),(0.37,0.06),(1.00,0.02),(1.58,0.03)]
])

fe_de_fe_ss_hzo = normalize([
    [(-0.66,0.05),(1.03,0.06)],
    [(-0.66,0.05),(0.05,0.08),(0.56,0.1),(1.03,0.06)]
])

fe_m_fe_ss_hzo = normalize([
    [(-0.84,0.12),(1.73,0.04)],
    [(-0.84,0.12),(0.17,0.09),(0.96,0.09),(1.73,0.04)]
])

# Open a file for writing
with open('fe_de_fe_sl_hzo.p', 'wb') as f:
    # Use pickle's dump function to write the data object to the file
    pickle.dump(fe_de_fe_sl_hzo, f)
with open('fe_de_fe_ss_hzo.p', 'wb') as f:
    # Use pickle's dump function to write the data object to the file
    pickle.dump(fe_de_fe_ss_hzo, f)
with open('fe_m_fe_ss_hzo.p', 'wb') as f:
    # Use pickle's dump function to write the data object to the file
    pickle.dump(fe_m_fe_ss_hzo, f)