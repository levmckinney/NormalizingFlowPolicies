from argparse import ArgumentParser
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--runs', '-r', nargs='+', required=True)
    parser.add_argument('--names', '-n', nargs='+')
    parser.add_argument('--xaxis', '-x', default='time step')
    parser.add_argument('--yaxis', '-y', default="reward")
    parser.add_argument('--reduce', '-u', default="median")
    parser.add_argument('--save_to', '-s', default='out.png')
    parser.add_argument('--title', '-t', default=None)
    args = parser.parse_args()


    if args.names is None:
        names = [os.path.basename(path) for path in args.runs]
    else:
        names = args.names

    # Collect plots
    for run_path, name in zip(args.runs, names):
        values = []
        timestep = None
        for filename in os.listdir(run_path):
            path = os.path.join(run_path, filename)
            table = pd.read_csv(path)

            if timestep is None:
                timestep = table['Step']
            else:
                assert (timestep == table['Step']).all(), 'All runs to be aggregated must share timesteps'    
            values.append(table['Value'])
        values = np.vstack(values)
        if args.reduce == 'mean':
            std = np.std(values, axis=0)
            mean = np.mean(values, axis=0)
            plt.plot(timestep, mean, label=f'{name} (N={values.shape[0]})')
            plt.fill_between(timestep, mean - std, mean + std, alpha=0.6)
        elif args.reduce == 'median':
            median = np.median(values, axis=0)
            low_quartile = np.quantile(values, 0, axis=0)
            upper_quintile = np.quantile(values, 1, axis=0)
            plt.plot(timestep, median, label=f'{name} (N={values.shape[0]})')
            plt.fill_between(timestep, low_quartile, upper_quintile, alpha=0.4)
    plt.title(args.title)
    plt.xlabel(args.xaxis)
    plt.ylabel(f'{args.reduce} {args.yaxis}')
    plt.legend(loc='upper left')

    plt.savefig(args.save_to)