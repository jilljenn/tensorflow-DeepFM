import tensorflow as tf
import numpy as np
import pandas as pd
from DeepFM import DeepFM
from sklearn.metrics import roc_auc_score
import os, sys
import argparse
import json
import time


start = time.time()
parser = argparse.ArgumentParser(description='Run DeepFM')
parser.add_argument('--dataset', type=str, nargs='?', default='/home/jj/deepfm/data/last_fr_en')
parser.add_argument('--iter', type=int, nargs='?', default=50)
parser.add_argument('--fm', type=bool, nargs='?', const=True, default=False)
parser.add_argument('--deep', type=bool, nargs='?', const=True, default=False)
parser.add_argument('--d', type=int, nargs='?', default=20)
parser.add_argument('--nb_layers', type=int, nargs='?', default=2)
parser.add_argument('--nb_neurons', type=int, nargs='?', default=50)
parser.add_argument('--batch', type=int, nargs='?', default=128)
parser.add_argument('--rate', type=float, nargs='?', default=0.01)
options = parser.parse_args()


print('Dataset', options.dataset, time.time() - start)
os.chdir(os.path.join('data', options.dataset))  # Move to dataset folder
start = time.time()


Xi_train = list(np.load('Xi_train.npy'))
Xv_train = list(np.load('Xv_train.npy'))
y_train = list(np.load('y_train.npy').astype(np.int32))

Xi_valid = list(np.load('Xi_valid.npy'))
Xv_valid = list(np.load('Xv_valid.npy'))
y_valid = list(np.load('y_valid.npy').astype(np.int32))

Xi_test = list(np.load('Xi_test.npy'))
Xv_test = list(np.load('Xv_test.npy'))

nb_fields = len(Xi_train[0])

# nb_features = (1 + np.array(Xi_train + Xi_valid).max(axis=0)).sum()  # Old, bad version
nb_features = int(1 + np.array(Xi_train + Xi_valid + Xi_test).max())
print(nb_features, 'features over', nb_fields, 'fields', time.time() - start)

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
    "batch_size": options.batch,
    "learning_rate": options.rate,
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
dfm.fit(Xi_train, Xv_train, y_train, Xi_valid, Xv_valid, y_valid, early_stopping=True, refit=True)
print('refit done')

# evaluate a trained model
auc_train = dfm.evaluate(Xi_train, Xv_train, y_train)
auc_valid = dfm.evaluate(Xi_valid, Xv_valid, y_valid)
print('train auc={:f} valid auc={:f}'.format(auc_train, auc_valid))

# save config
del dfm_params['deep_layers_activation']
del dfm_params['eval_metric']
config = {
    'args': vars(options),
    'dfm_params': dfm_params,
    'metrics': {
        'samples_train': len(Xi_train),
        'samples_valid': len(Xi_valid),
        'samples_test': len(Xi_test),
        'auc_train': float(auc_train),
        'auc_valid': float(auc_valid),
    }
}
print(config)

# make prediction on test
y_pred = dfm.predict(Xi_test, Xv_test)
with open('y_pred-{:.3f}-{:.3f}.txt'.format(auc_train, auc_valid), 'w') as f:
    f.write('\n'.join(map(str, y_pred)))
with open('y_pred-{:.3f}-{:.3f}.config.json'.format(auc_train, auc_valid), 'w') as f:
    f.write(json.dumps(config, indent=4))
