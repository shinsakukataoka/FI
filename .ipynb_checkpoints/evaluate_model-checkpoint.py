import numpy
from keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from fault_injection import dnn_fi

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score
from keras.datasets import imdb
import numpy as np
import json
from fault_injection import dnn_fi_keras
from val_config import *

# fix random seed for reproducibility
numpy.random.seed(7)

# load the dataset but only keep the top n words, and replace the rest with zeros
top_words = 5000
(_, _), (X_test, y_test) = imdb.load_data(num_words=top_words)

# truncate and/or pad input sequences
max_review_length = 600
X_test = pad_sequences(X_test, maxlen=max_review_length)

# Load the saved model
loaded_model = load_model("trained_model.h5")

# Evaluation of the loaded model
scores = loaded_model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1] * 100))

setting = {'q_type': 'signed', 'encode': 'dense', 'int_bits': 2, 'frac_bits': 6, 'rep_conf': np.array([2, 2, 2, 2, 2, 2, 2, 2])}
ber = 0.01

fault_model = dnn_fi_keras(loaded_model, seed=0, ber=ber, **setting)
fault_model_scores = fault_model.evaluate(X_test, y_test, verbose=0)
print("Accuracy after fault injection: %.2f%%" % (fault_model_scores[1] * 100))