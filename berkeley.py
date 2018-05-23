import tensorflow as tf
import numpy as np
import pandas as pd
from DeepFM import DeepFM
from sklearn.metrics import roc_auc_score
import os, sys
import argparse
import pickle


parser = argparse.ArgumentParser(description='Run DeepFM')
parser.add_argument('--dataset', type=str, nargs='?', default='berkeley0')
parser.add_argument('--iter', type=int, nargs='?', default=50)
parser.add_argument('--fm', type=bool, nargs='?', const=True, default=False)
parser.add_argument('--deep', type=bool, nargs='?', const=True, default=False)
parser.add_argument('--d', type=int, nargs='?', default=2)
parser.add_argument('--nb_layers', type=int, nargs='?', default=4)
parser.add_argument('--nb_neurons', type=int, nargs='?', default=50)
options = parser.parse_args()


print('Dataset', options.dataset)
# os.chdir(os.path.join('data', options.dataset))  # Move to dataset folder


with open('berkeley0.pickle', 'rb') as f:
    Xi_train = pickle.load(f)
    Xv_train = pickle.load(f)
    y_train = pickle.load(f)
    Xi_test = pickle.load(f)
    Xv_test = pickle.load(f)
    y_test = pickle.load(f)

nb_fields = len(Xi_train[0])
print(len(Xi_train), nb_fields, len(y_train), min(y_train), max(y_train))
data = np.array(Xv_train + Xv_test)
print('Interesting')
print(list(map(type, Xv_train[0])))
print(data.min(axis=0))
print(data.max(axis=0))

nb_features = (1 + np.array(Xi_train + Xi_test).max(axis=0)).sum()
print(nb_features, 'features over', nb_fields, 'fields')

# params
dfm_params = {
    "use_fm": options.fm,
    "use_deep": options.deep,
    "embedding_size": options.d,
    "dropout_fm": [1.0, 1.0],
    "deep_layers": [options.nb_neurons] * options.nb_layers,
    "dropout_deep": [0.6] * (options.nb_layers + 1),
    "deep_layers_activation": tf.nn.relu,
    "epoch": options.iter,
    "batch_size": 1024,
    "learning_rate": 0.001,
    "optimizer_type": "adam",
    "batch_norm": 1,
    "batch_norm_decay": 0.995,
    "l2_reg": 0.01,
    "verbose": True,
    "eval_metric": roc_auc_score,
    "random_seed": 2017,
    'feature_size': nb_features,
    'field_size': nb_fields
}

# init a DeepFM model
dfm = DeepFM(**dfm_params)

# fit a DeepFM model
dfm.fit(Xi_train, Xv_train, y_train, Xi_test, Xv_test, y_test)

# evaluate a trained model
auc_train = dfm.evaluate(Xi_train, Xv_train, y_train)
auc_test = dfm.evaluate(Xi_test, Xv_test, y_test)
print('train auc={:f} test auc={:f}'.format(auc_train, auc_test))

# make prediction on test
# y_pred = dfm.predict(Xi_test, Xv_test)
# with open('y_pred-{:.3f}-{:.3f}.txt'.format(auc_train, auc_valid), 'w') as f:
#     f.write('\n'.join(map(str, y_pred)))
