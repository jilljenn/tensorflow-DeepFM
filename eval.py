import pandas as pd
import numpy as np
import glob
import json
import re


results = []
for filename in glob.glob('data/*/*-*-*-*.json'):
    dataset = filename.split('/')[1]
    with open(filename) as f:
        result = json.load(f)
    is_lr = result['args'].get('logistic')
    is_deep = result['args'].get('deep') == True
    is_fm = result['args'].get('fm') == True
    if is_lr == True:
        model = 'LR'
    elif is_lr == False:
        model = 'Bayesian FM'
    elif is_fm and not is_deep:  # is_lr can be None
        model = 'FM'
    elif is_deep and not is_fm:
        model = 'Deep'
    else:
        model = 'DeepFM'
    results.append({
        'train_auc': result['metrics'].get('auc_train'),
        'test_auc': result['metrics']['auc_test'],
        'epoch': '{:d}/{:d}'.format(result.get('finished_at_epoch', result['args']['iter']), result['args']['iter']),
        'dataset': dataset[:dataset.index('_')],
        'data': dataset[-5:],
        'd': 0 if is_lr else result['args']['d'],
        'model': model
    })
df = pd.DataFrame.from_dict(results).sort_values(by=['data', 'test_auc'], ascending=[True, False]).round(3).reset_index()
df['first'] = np.where(df['dataset'] == 'first', df['test_auc'], None)
for data in df['data'].unique():
    df.query('data == @data')[['dataset', 'model', 'd', 'epoch', 'train_auc', 'test_auc', 'first']].to_latex('results-{:s}.tex'.format(data), index=False)
print(df.fillna('--'))
