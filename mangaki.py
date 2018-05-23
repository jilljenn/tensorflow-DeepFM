import pandas
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from DeepFM import DeepFM
import tensorflow as tf
import argparse

rating_values = {'favorite': 4, 'like': 2, 'dislike': -2, 'neutral': 0.1, 'willsee': 0.5, 'wontsee': -0.5}


parser = argparse.ArgumentParser(description='Run DeepFM')
parser.add_argument('--dataset', type=str, nargs='?', default='fren')
parser.add_argument('--iter', type=int, nargs='?', default=50)
parser.add_argument('--fm', type=bool, nargs='?', const=True, default=False)
parser.add_argument('--deep', type=bool, nargs='?', const=True, default=False)
parser.add_argument('--d', type=int, nargs='?', default=20)
parser.add_argument('--nb_layers', type=int, nargs='?', default=2)
parser.add_argument('--nb_neurons', type=int, nargs='?', default=50)
options = parser.parse_args()


df = pandas.read_csv('ratings.csv', names=('user', 'item', 'choice'))
nb_users = 1 + df['user'].max()
nb_works = 1 + df['item'].max()
print(df['user'].dtype, df['item'].dtype)


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
    "batch_size": 10240,
    "learning_rate": 0.001,
    "optimizer_type": "adam",
    "batch_norm": 1,
    "batch_norm_decay": 0.995,
    "l2_reg": 0.1,
    "verbose": True,
    "eval_metric": mean_squared_error,
    "random_seed": 2017,
    'feature_size': nb_users + nb_works,
    'field_size': 2
}

df['rating'] = df['choice'].map(rating_values)
trainval, test = train_test_split(df[['user', 'item', 'rating']], shuffle=True, test_size=0.2)
train, valid = train_test_split(trainval, shuffle=True, test_size=0.1)
print(len(train), len(valid), len(test))
print(train.head())


def prepare(df):
    n = len(df)
    Xi = df[['user', 'item']].values.tolist()
    print(Xi[:10], 'wut')
    Xv = [[1, 1] for _ in range(n)]
    y = df['rating'].values.tolist()
    return Xi, Xv, y


Xi_train, Xv_train, y_train = prepare(train)
print('wat', Xi_train[:5])
Xi_valid, Xv_valid, y_valid = prepare(valid)
Xi_test, Xv_test, y_test = prepare(test)


dfm = DeepFM(**dfm_params)
dfm.fit(Xi_train, Xv_train, y_train, Xi_valid, Xv_valid, y_valid, early_stopping=True, refit=True)

# evaluate a trained model
mse_train = dfm.evaluate(Xi_train, Xv_train, y_train)
mse_valid = dfm.evaluate(Xi_valid, Xv_valid, y_valid)
print('train rmse={:f} valid rmse={:f}'.format(mse_train ** 0.5, mse_valid ** 0.5))

# make prediction on test
y_pred = dfm.predict(Xi_test, Xv_test)
mse_test = dfm.evaluate(Xi_test, Xv_test, y_test)
print('test rmse={:f}'.format(mse_test ** 0.5))
