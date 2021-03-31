import argparse
import os

from src import hyperparameter_tuning
from src import simulate


parser = argparse.ArgumentParser(description='placeholder')

parser.add_argument(
    '-r', '--run-experiments', 
    default='',
    type=str,
    help='Run experiments. Options: "ALL", "TQL", "DQL", "MCPG"'
)

parser.add_argument(
    '-c', '--cores', 
    default=4,
    type=int,
    help='Number of cores to use for running experiments.'
)

parser.add_argument(
    '-s', '--simulate', 
    default='',
    type=str,
    help='Display simulations of TQL and DQL algorithms. Options: "TQL", "DQL".'
)

parser.add_argument(
    '-e', '--epsisodes', 
    default=4,
    type=int,
    help='Number of cores to use for running experiments.'
)


args = parser.parse_args()

if __name__ == '__main__':
    
    if not os.path.exists('log/tql'):
        os.mkdir('log/tql')
    if not os.path.exists('log/dql'):
        os.mkdir('log/dql')
    if not os.path.exists('log/mcpg'):
        os.mkdir('log/mcpg')

    if not os.path.exists('models/tql'):
        os.mkdir('models/tql')
    if not os.path.exists('models/dql'):
        os.mkdir('models/tql')
    if not os.path.exists('models/mcpg'):
        os.mkdir('models/mcpg')
    
    if not os.path.exists('output/tql'):
        os.mkdir('output/tql')
    if not os.path.exists('output/dql'):
        os.mkdir('output/dql')
    if not os.path.exists('output/mcpg'):
        os.mkdir('output/mcpg')

    if args.simulate:
        if args.simulate.lower() == 'tql':
            simulate.tql(args.epsisodes)
        elif args.simulate.lower() == 'dql':
            simulate.dql(args.epsisodes)
        else:
            raise ValueError('Unknown model "{}"'.format(args.simulate))

    if args.run_experiments and args.run_experiments.lower() == 'all':
        hyperparameter_tuning.tql(args.cores)
        hyperparameter_tuning.dql(args.cores)

    elif args.run_experiments and args.run_experiments.lower() == 'tql':
        hyperparameter_tuning.tql(args.cores)
    elif args.run_experiments and args.run_experiments.lower() == 'dql':
        hyperparameter_tuning.dql(args.cores)
    elif args.run_experiments and args.run_experiments.lower() == 'mcpg':
        pass
        # TODO josh func
    