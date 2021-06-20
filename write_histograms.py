from argparse import ArgumentParser
from collections import defaultdict

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-trd", "--train_data", dest="train_data_path",
                        help="path to training data file", required=True)
    parser.add_argument("-sh", "--subtoken_histogram", dest="subtoken_histogram",
                        help="subtoken histogram file", metavar="FILE", required=True)
    parser.add_argument("-nh", "--node_histogram", dest="node_histogram",
                        help="node_histogram file", metavar="FILE", required=True)
    parser.add_argument("-th", "--target_histogram", dest="target_histogram",
                        help="target histogram file", metavar="FILE", required=True)
    args = parser.parse_args()

    with open(args.train_data_path) as train_data,\
         open(args.subtoken_histogram, 'w') as subtoken_histogram,\
         open(args.node_histogram, 'w') as node_histogram,\
         open(args.target_histogram, 'w') as target_histogram:
        subtoken_dict = defaultdict(lambda: 0)
        node_dict = defaultdict(lambda: 0)
        target_dict = defaultdict(lambda: 0)
        for line in train_data:
            target, nodes, tokens, _ = line.rstrip('\n').split(' ')
            for target_subtoken in target.split('|'):
                target_dict[target_subtoken] += 1
            for node in nodes.split(','):
                node_dict[node] += 1
            for token in tokens.split(','):
                for subtoken in token.split('|'):
                    subtoken_dict[subtoken] += 1
        for f, dictionary in zip((subtoken_histogram, node_histogram, target_histogram), (subtoken_dict, node_dict, target_dict)):
            for key, value in dictionary.items():
                f.write('{} {}\n'.format(key, value))
