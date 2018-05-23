import numpy as np


N = 100
X_user = np.random.randint(0, 2, size=(N, 1))
X_item = np.random.randint(2, 5, size=(N, 1))
y = np.random.randint(0, 2, size=(N,))
Xi = np.column_stack((X_user, X_item))
Xv = np.ones((N, 2))

to_train = int(0.6 * N)
from_test = int(0.8 * N)
to_save = {
    'Xi_train': Xi[:to_train],
    'Xv_train': Xv[:to_train],
    'y_train': y[:to_train],
    'Xi_valid': Xi[to_train:from_test],
    'Xv_valid': Xv[to_train:from_test],
    'y_valid': y[to_train:from_test],
    'Xi_test': Xi[from_test:],
    'Xv_test': Xv[from_test:],
    'y_test': y[from_test:]
}
print(to_save)
for filename in to_save:
    np.save('data/dummy/{:s}.npy'.format(filename), to_save[filename])
