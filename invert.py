import numpy as np
import glob
import re
import os.path


r = re.compile(r'data/last_(.*)/y_pred-(.*).txt')
for filename in glob.glob('data/last*/y_pred-*.txt'):
    m = r.match(filename)
    if m:
        dataset, score = m.groups()
        submission = '{:s}-{:s}.pred'.format(dataset, score)
        if not os.path.isfile(submission):
            with open(filename) as f:
                lines = np.array(f.read().splitlines()).astype(np.float64)
            with open('{:s}.keys'.format(dataset)) as f:
                keys = f.read().splitlines()
            assert len(keys) == len(lines)
            with open(submission, 'w') as f:
                for key, value in zip(keys, lines):
                    f.write('{:s} {:s}\n'.format(key, str(1 - value)))
            print(submission, 'created')
