from scipy.sparse import coo_matrix, vstack, save_npz, load_npz, hstack, csr_matrix
from scipy.stats import sem, t
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import pywFM
from sklearn.metrics import roc_auc_score, ndcg_score
import os, sys
import argparse
import json
import time
import getpass


def avgstd(l):
    '''
    Given a list of values, returns a 95% confidence interval
    if the standard deviation is unknown.
    '''
    n = len(l)
    mean = sum(l) / n
    if n == 1:
        return '%.3f' % round(mean, 3)
    std_err = sem(l)
    confidence = 0.95
    h = std_err * t.ppf((1 + confidence) / 2, n - 1)
    return '%.3f ± %.3f' % (round(mean, 3), round(h, 3))


start = time.time()
parser = argparse.ArgumentParser(description='Run FM')
if getpass.getuser() == 'jj':  # Was for RAIDEN
    parser.add_argument('--base_dir', type=str, nargs='?', default='/home/jj')
    parser.add_argument('--truth_path', type=str, nargs='?', default='code/slam2018')
    parser.add_argument('--libfm', type=str, nargs='?', default='code/ktm/libfm')
else:
    parser.add_argument('--base_dir', type=str, nargs='?', default='/Users/jilljenn')
    parser.add_argument('--truth_path', type=str, nargs='?', default='code/sharedtask')
    parser.add_argument('--libfm', type=str, nargs='?', default='code/libfm')
parser.add_argument('--logistic', type=bool, nargs='?', const=True, default=False)
parser.add_argument('--countries', type=bool, nargs='?', const=True, default=False)
parser.add_argument('--dataset', type=str, nargs='?', default='first_fr_en')
parser.add_argument('--iter', type=int, nargs='?', default=20)
parser.add_argument('--d', type=int, nargs='?', default=20)
parser.add_argument('--ver', type=int, nargs='?', default=0)
options = parser.parse_args()

os.environ['LIBFM_PATH'] = os.path.join(options.base_dir, options.libfm, 'bin/')

print('Dataset', options.dataset, time.time() - start)
dataset_key = options.dataset[-5:]
ckpt = time.time()
os.chdir(os.path.join('data', options.dataset))  # Move to dataset folder
start = time.time()


slicing = [
    [0, 1],
    [0, 1, 2],
    [0, 1, 3],
    [0, 1, 4],
    [0, 1, 2, 3, 4]
][options.ver]

Xi_train = np.load('Xi_train.npy')[:, slicing]
Xv_train = np.load('Xv_train.npy')[:, slicing]
train_samples = len(Xi_train)
y_train = np.load('y_train.npy').astype(np.int32)

Xi_valid = np.load('Xi_valid.npy')[:, slicing]
Xv_valid = np.load('Xv_valid.npy')[:, slicing]
valid_samples = len(Xi_valid)
y_valid = np.load('y_valid.npy').astype(np.int32)

Xi_test = np.load('Xi_test.npy')[:, slicing]
Xv_test = np.load('Xv_test.npy')[:, slicing]

nb_fields = len(Xi_train[0])

print('min', np.vstack((Xi_train, Xi_valid, Xi_test)).min(axis=0))
print('max', np.vstack((Xi_train, Xi_valid, Xi_test)).max(axis=0))

nb_features = int(1 + np.vstack((Xi_train, Xi_valid, Xi_test)).max())
print(nb_features, 'features over', nb_fields, 'fields', time.time() - start)

def lol_to_csr(Xi, Xv, bonus=True):
    nb_samples, nb_fields = Xi.shape
    rows = np.repeat(np.arange(nb_samples), nb_fields)
    cols = Xi.flatten()
    data = Xv.flatten()
    X = coo_matrix((data, (rows, cols)), shape=(nb_samples, nb_features)).tocsr()
    if bonus:
        X_bonus = adj[Xi[:, 0]]  # Extra features per user
        print('Extra feat. (countries)', Counter(X_bonus.sum(axis=1).A1))
        return hstack((X, X_bonus))
    
    return X

adj = load_npz('adj.npz')
X_train = lol_to_csr(Xi_train, Xv_train, options.countries)
X_valid = lol_to_csr(Xi_valid, Xv_valid, options.countries)
X_test = lol_to_csr(Xi_test, Xv_test, options.countries)
print('Finished converting data', time.time() - ckpt)

if options.dataset == 'dummy':
    X_fulltrain = X_train
    y_fulltrain = y_train
    y_test = [1] * (len(Xi_test) - 1) + [0]
else:
    X_fulltrain = vstack((X_train, X_valid))
    y_fulltrain = np.concatenate((y_train, y_valid))

    df = pd.read_csv(os.path.join(options.base_dir, options.truth_path, 'data_{:s}/{:s}.slam.20190204.test.key'.format(dataset_key, dataset_key)),
                     sep=' ', names=('key', 'outcome'))
    y_test = 1 - df['outcome']

# nb_features = (1 + np.array(Xi_train + Xi_valid).max(axis=0)).sum()  # Old, bad version
# nb_features = int(1 + np.array(Xi_train + Xi_valid + Xi_test).max())
# print(nb_features, 'features over', nb_fields, 'fields', time.time() - start)

print(Counter(X_fulltrain.sum(axis=1).A1))

# params
params = {
    'task': 'classification',
    'num_iter': options.iter,
    'rlog': True,
    'learning_method': 'mcmc',
    'k2': options.d
}

def right_pad(X, expected_size):
    N, current_size = X.shape
    return hstack((X, csr_matrix(np.zeros((N, expected_size - current_size)))))


print(X_fulltrain.shape, 'fully')
print('before', Counter(X_fulltrain.sum(axis=0).A1)[0], 'are useless')
X_fulltrain_countries = adj[np.concatenate((Xi_train[:, 0], Xi_valid[:, 0]))]
X_test_countries = adj[Xi_test[:, 0]]
print('int', X_fulltrain_countries.shape)
print('int', Counter(X_fulltrain_countries.sum(axis=1).A1))
#X_fulltrain = hstack((X_fulltrain, X_fulltrain_countries))
#X_test = hstack((X_test, X_test_countries))


save_npz('X_fm.npz', vstack((X_fulltrain, X_test)))
print('final', vstack((X_fulltrain, X_test)).shape)
np.save('y_fm.npy', np.concatenate((y_fulltrain, y_test)))


auc_train = 0
auc_valid = 0
if options.logistic or options.d == 0:

    
    nb_entities, _ = adj.shape
    print(X_fulltrain.shape, type(X_fulltrain))
    print(X_train.shape, type(X_train))
    print(X_valid.shape, type(X_valid))
    print(X_test.shape, type(X_test))

    # sys.exit(0)
    """
    save_npz('X_train.npz', right_pad(X_train, nb_entities))
    save_npz('X_valid.npz', right_pad(X_valid, nb_entities))
    save_npz('X_test.npz', right_pad(X_test, nb_entities))
    np.save('y_test.npy', y_test)
    """

    model = LogisticRegression(solver='liblinear')
    model.fit(X_fulltrain, y_fulltrain)
    # print('Best', model.C_)

    y_pred_train = model.predict_proba(X_train)[:, 1]
    # y_pred_valid = model.predict_proba(X_valid)[:, 1]
    y_pred_test = model.predict_proba(X_test)[:, 1]

    auc_train = roc_auc_score(y_train, y_pred_train)
    # auc_valid = roc_auc_score(y_valid, y_pred_valid)
else:
    # init a FM model
    fm = pywFM.FM(**params)

    # fit a FM model
    model = fm.run(X_fulltrain, y_fulltrain, X_test, y_test)
    y_pred_test = model.predictions

# evaluate a trained model
auc_test = roc_auc_score(y_test, y_pred_test)

predictions_per_user = defaultdict(lambda: defaultdict(list))
metrics = defaultdict(list)

with open('test_user_ids.txt') as f:
    test_user_ids = f.read().splitlines()

for user, pred, true in zip(test_user_ids, y_pred_test, y_test):
    predictions_per_user[user]['pred'].append(pred)
    predictions_per_user[user]['y'].append(true)
    predictions_per_user[user]['opp_pred'].append(1 - pred)
    predictions_per_user[user]['opp_y'].append(1 - true)
    
for user in predictions_per_user:
    this_pred = predictions_per_user[user]['pred']
    this_true = predictions_per_user[user]['y']
    opp_this_pred = predictions_per_user[user]['opp_pred']
    opp_this_true = predictions_per_user[user]['opp_y']
    if len(this_pred) > 1:
        metrics['ndcg'].append(ndcg_score([this_true], [this_pred]))
        metrics['ndcg@10'].append(ndcg_score([this_true], [this_pred], k=10))
        metrics['ndcg-'].append(ndcg_score([opp_this_true], [opp_this_pred]))
        metrics['ndcg@10-'].append(ndcg_score([opp_this_true], [opp_this_pred], k=10))


print('train auc={:f} valid auc={:f} test auc={:f}'.format(auc_train, auc_valid, auc_test))
print('ndcg', avgstd(metrics['ndcg']))
print('ndcg@10', avgstd(metrics['ndcg@10']))
print('ndcg-', avgstd(metrics['ndcg-']))
print('ndcg@10-', avgstd(metrics['ndcg@10-']))


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
