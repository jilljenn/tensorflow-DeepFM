from string import Template
import argparse


PATH = '/home/jj/deepfm'


parser = argparse.ArgumentParser(description='Make bash script')
parser.add_argument('--dataset', type=str, nargs='?', default='fr_en')
parser.add_argument('--fm_iter', type=int, nargs='?', default=100)
parser.add_argument('--iter', type=int, nargs='?', default=1000)
parser.add_argument('--d', type=int, nargs='?', default=20)
parser.add_argument('--rate', type=float, nargs='?', default=0.001)
options = parser.parse_args()


prefix = '-'.join(map(str, vars(options).values()))
with open('template.sh') as f:
    t = Template(f.read())
    bash = t.substitute({
        'PATH': PATH,
        'dataset': options.dataset,
        'fm_iter': options.fm_iter,
        'iter': options.iter,
        'd': options.d,
        'rate': options.rate,
        'prefix': prefix
    })

with open('{:s}.sh'.format(prefix), 'w') as f:
    f.write(bash)
