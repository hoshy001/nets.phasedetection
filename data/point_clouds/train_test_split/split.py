import argparse
from dataclasses import dataclass, field, astuple
import numpy as np
import os
import random


@dataclass
class Params:
    train_ratio: float = field(
        default=0.8,
        metadata={
            "help": 'train data split ratio'
        },
    )
    num_points: int = field(
        default=3000,
        metadata={
            "help": 'number of point clouds for each phase'
        },
    )
    phases: tuple = field(
        default=('lam', 'hpc', 'hpl', 'bcc', 'dis', 'dg'),
        metadata={
            "help": 'a tuple of names of different possible phases'
        },
    )


def split(train_ratio: float, num_points: int, phases: tuple):
    # Initialize the random number generator.
    random.seed(166)

    train_ratio, n_points, phases = astuple(
        Params(train_ratio=train_ratio, num_points=num_points, phases=phases))

    n_train = int(n_points * train_ratio)
    n_test = n_points - n_train
    n_class = len(phases)
    train_id, test_id = (np.zeros((n_train, n_class)),
                         np.zeros((n_test, n_class)))
    data_train, data_test = [], []
    for idx, cls in enumerate(phases):
        cls_idx_train = random.sample(range(1, n_points + 1), n_train)
        cls_idx_test = [i for i in range(
            1, n_points + 1) if i not in cls_idx_train]
        train_id[:, idx] = cls_idx_train
        test_id[:, idx] = cls_idx_test

        for idx_train in train_id[:, idx]:
            data_train.append(
                f'"../{cls}/coord_O_{cls}_{int(idx_train)}"')

        for idx_test in test_id[:, idx]:
            data_test.append(
                f'"../{cls}/coord_O_{cls}_{int(idx_test)}"')

    with open('shuffled_train_file_list.json', 'w') as f:
        assert len(
            data_train) > 1, f'number of point clouds for training is {len(data_train)}'
        f.write('[')
        for k in range(len(data_train) - 1):
            f.write(data_train[k] + ',')
        f.write(data_train[-1] + ']')

    with open('shuffled_test_file_list.json', 'w') as f:
        assert len(
            data_train) > 1, f'number of point clouds for testing is {len(data_train)}'
        f.write('[')
        for k in range(len(data_test) - 1):
            f.write(data_test[k] + ',')
        f.write(data_test[-1] + ']')

    with open(os.path.join('..', 'phase_category.txt'), 'w') as f:
        for phase in phases:
            f.write(f'{phase.upper()}\t{phase.lower()}\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train_ratio',
        type=float,
        default=0.8,
        help='float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split.')
    parser.add_argument(
        '--num_points',
        type=int,
        default=3000,
        help='number of point clouds for each phase.')
    parser.add_argument(
        '--phases',
        type=tuple,
        default=('lam', 'hpc', 'hpl', 'bcc', 'dis', 'dg'),
        nargs='+',
        help='tuple of names of different possible phases.')
    args = parser.parse_args()
    if args.phases != ('lam', 'hpc', 'hpl', 'bcc', 'dis', 'dg'):
        for i, phase in enumerate(args.phases):
            args.phases[i] = ''.join(phase)
            args.phases[i] = args.phases[i].replace(',', '')
    args.phases = tuple(set(args.phases))

    assert 0.0 < args.train_ratio <= 1.0, f'train data split ratio = {args.train_ratio}'
    assert 0 < args.num_points, f'number of point clouds for each phase = {args.num_points}'
    assert 0 < len(
        args.phases), f'names of different possible phases = {args.phases}'

    print(f'names of different possible phases = {args.phases}')

    split(args.train_ratio, args.num_points, args.phases)
