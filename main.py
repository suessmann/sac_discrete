import re
import yaml
import argparse
from run import run

parser = argparse.ArgumentParser(description='Start training')
parser.add_argument("-c", "--config", type=str, help="Path to the config file")

if __name__ == '__main__':
    args = vars(parser.parse_args())

    loader = yaml.SafeLoader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
         [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))

    with open(args['config'], 'r') as ymfile:
        cfg = yaml.load(ymfile, Loader=loader)

    print(cfg)
    run(cfg)
