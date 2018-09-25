from scipy.sparse import coo_matrix, vstack, save_npz
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import pywFM
from sklearn.metrics import roc_auc_score
import os, sys
import argparse
import json
import time
import getpass


start = time.time()
parser = argparse.ArgumentParser(description='Run FM')
if getpass.getuser() == 'jj':
    parser.add_argument('--base_dir', type=str, nargs='?', default='/home/jj')
    parser.add_argument('--truth_path', type=str, nargs='?', default='dataverse_files')
    parser.add_argument('--libfm', type=str, nargs='?', default='libfm')
else:
    parser.add_argument('--base_dir', type=str, nargs='?', default='/Users/jilljenn')
    parser.add_argument('--truth_path', type=str, nargs='?', default='code/sharedtask')
    parser.add_argument('--libfm', type=str, nargs='?', default='code/libfm')
parser.add_argument('--logistic', type=bool, nargs='?', const=True, default=False)
parser.add_argument('--dataset', type=str, nargs='?', default='first_fr_en')
parser.add_argument('--iter', type=int, nargs='?', default=50)
parser.add_argument('--d', type=int, nargs='?', default=20)
options = parser.parse_args()

os.environ['LIBFM_PATH'] = os.path.join(options.base_dir, options.libfm, 'bin')

print('Dataset', options.dataset, time.time() - start)
dataset_key = options.dataset[-5:]
ckpt = time.time()
os.chdir(os.path.join('data', options.dataset))  # Move to dataset folder
start = time.time()


Xi_train = np.load('Xi_train.npy')
Xv_train = np.load('Xv_train.npy')
train_samples = len(Xi_train)
y_train = np.load('y_train.npy').astype(np.int32)

Xi_valid = np.load('Xi_valid.npy')
Xv_valid = np.load('Xv_valid.npy')
valid_samples = len(Xi_valid)
y_valid = np.load('y_valid.npy').astype(np.int32)

Xi_test = np.load('Xi_test.npy')
Xv_test = np.load('Xv_test.npy')

nb_fields = len(Xi_train[0])
nb_features = int(1 + np.vstack((Xi_train, Xi_valid, Xi_test)).max())
print(nb_features, 'features over', nb_fields, 'fields', time.time() - start)

def lol_to_csr(Xi, Xv):
    nb_samples, nb_fields = Xi.shape
    rows = np.repeat(np.arange(nb_samples), nb_fields)
    cols = Xi.flatten()
    data = Xv.flatten()
    return coo_matrix((data, (rows, cols)), shape=(nb_samples, nb_features)).tocsr()

X_train = lol_to_csr(Xi_train, Xv_train)
X_valid = lol_to_csr(Xi_valid, Xv_valid)
X_test = lol_to_csr(Xi_test, Xv_test)
print('Finished converting data', time.time() - ckpt)

if options.dataset == 'dummy':
    X_fulltrain = X_train
    y_fulltrain = y_train
    y_test = [1] * (len(Xi_test) - 1) + [0]
else:
    X_fulltrain = vstack((X_train, X_valid))
    y_fulltrain = np.concatenate((y_train, y_valid))

    df = pd.read_csv(os.path.join(options.base_dir, options.truth_path, 'data_{:s}/{:s}.slam.20171218.test.key'.format(dataset_key, dataset_key)),
                     sep=' ', names=('key', 'outcome'))
    y_test = 1 - df['outcome']

# nb_features = (1 + np.array(Xi_train + Xi_valid).max(axis=0)).sum()  # Old, bad version
# nb_features = int(1 + np.array(Xi_train + Xi_valid + Xi_test).max())
# print(nb_features, 'features over', nb_fields, 'fields', time.time() - start)

# params
params = {
    'task': 'classification',
    'num_iter': options.iter,
    'rlog': True,
    'learning_method': 'mcmc',
    'k2': options.d
}

auc_train = 0
auc_valid = 0
if options.logistic or options.d == 0:

    save_npz('X_fm.npz', X_fulltrain)
    np.save('y_fm.npy', y_fulltrain)
    save_npz('X_train.npz', X_train)
    save_npz('X_valid.npz', X_valid)
    save_npz('X_test.npz', X_test)
    np.save('y_test.npy', y_test)

    model = LogisticRegression()
    model.fit(X_fulltrain, y_fulltrain)

    y_pred_train = model.predict_proba(X_train)[:, 1]
    y_pred_valid = model.predict_proba(X_valid)[:, 1]
    y_pred_test = model.predict_proba(X_test)[:, 1]

    auc_train = roc_auc_score(y_train, y_pred_train)
    auc_valid = roc_auc_score(y_valid, y_pred_valid)
else:
    # init a FM model
    fm = pywFM.FM(**params)

    # fit a FM model
    model = fm.run(X_fulltrain, y_fulltrain, X_test, y_test)
    y_pred_test = model.predictions

# evaluate a trained model
auc_test = roc_auc_score(y_test, y_pred_test)
print('train auc={:f} valid auc={:f} test auc={:f}'.format(auc_train, auc_valid, auc_test))

# save config
config = {
    'args': vars(options),
    'fm_params': params,
    'metrics': {
        'samples_train': train_samples,
        'samples_valid': valid_samples,
        'samples_test': len(Xi_test),
        'auc_test': float(auc_test),
    }
}
print(config)

# make prediction on test
with open('y_pred-0-0-{:.3f}.txt'.format(auc_test), 'w') as f:
    f.write('\n'.join(map(str, y_pred_test)))
with open('y_pred-0-0-{:.3f}.config.json'.format(auc_test), 'w') as f:
    f.write(json.dumps(config, indent=4))
