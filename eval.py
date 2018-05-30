import pandas as pd
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
        'dataset': dataset,
        'd': 0 if is_lr else result['args']['d'],
        'model': model
    })
df = pd.DataFrame.from_dict(results).sort_values(by=['dataset', 'test_auc'], ascending=[True, False])
df.to_latex('results.tex')
print(df)
