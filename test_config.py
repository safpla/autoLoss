import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--list', type=list, nargs='+', default=[1,2,3])

config = parser.parse_args()
print(config.list)
