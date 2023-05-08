from typing import Dict, Any
from functools import partial

import argparse
import os
import time

import numpy as np
import pickle as pkl

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

from qualitylib.tools import import_python_file, read_dataset
from qualitylib.runner import Runner
from qualitylib.cross_validate import random_cross_validation

np.random.seed(0)


class ScaledSVR:
    def __init__(self, *svr_args, **svr_kwargs) -> None:
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.reg = SVR(*svr_args, **svr_kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X_trans = self.scaler.fit_transform(X)
        self.reg.fit(X_trans, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.reg.predict(self.scaler.transform(X))


def print_agg_stats(stats: Dict[str, Any]) -> None:
    sample_stats = stats[list(stats.keys())[0]]
    num_samples = len(sample_stats)
    lo_ci = (0.5 - 1.96*0.5/np.sqrt(num_samples))*100
    hi_ci = (0.5 + 1.96*0.5/np.sqrt(num_samples))*100

    # Each key in dict corresponds to one set of hyperparameters for the regressor.
    # Find best hyperparameter based on median SROCC
    maxval = -1
    max_param_key = None
    for param_key in stats:
        key_stats = np.array([stat['SROCC'] for stat in stats[param_key]])
        medval = np.median(key_stats)
        if medval > maxval:
            maxval = medval
            max_param_key = param_key

    print(f'Optimal param: {max_param_key}')
    print('Stat,Median,LoCI,HiCI,Std')  # Not using spaces makes parsing text output as csv easier
    for stat_key in stats[max_param_key][0]:
        key_stats = np.array([stat[stat_key] for stat in stats[max_param_key]])
        print(f'{stat_key},{np.median(key_stats)},{np.percentile(key_stats, lo_ci)},{np.percentile(key_stats, hi_ci)},{np.std(key_stats)}')


def dict_to_str(d: Dict[Any, Any]) -> str:
    s_arr = []
    for key in d:
        s_arr.extend([str(key), str(d[key])])
    return '_'.join(s_arr)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Conduct gridsearch crossvalidation')
    parser.add_argument('--dataset', help='Path to dataset file for which to extract features', type=str)
    parser.add_argument('--fex_name', help='Name of feature extractor', type=str)
    parser.add_argument('--fex_version', help='Version of feature extractor', type=str, default=None)
    parser.add_argument('--feat_file', help='Path to csv containing features', type=str, required=True)
    parser.add_argument('--feat_names_file', help='Path to file containing feature sets', type=str, default=None)
    parser.add_argument('--regressor', help='Regressor to use', type=str, default='RandomForest')
    parser.add_argument('--splits', help='Number of parallel processes', type=int, default=100)
    parser.add_argument('--processes', help='Number of parallel processes', type=int, default=100)
    parser.add_argument('--out_file', help='Path to output pickle file', type=str, required=True)
    return parser


def main() -> None:
    args = get_parser().parse_args()

    if os.path.isfile(args.out_file):
        return

    dataset = import_python_file(args.dataset)
    assets = read_dataset(dataset, shuffle=True)
    runner = Runner(args.fex_name, args.fex_version, processes=args.processes, use_cache=True)  # Reads from stored results if available, else stores results.

    if args.feat_names_file is not None:
        mod = import_python_file(args.feat_names_file)
        feat_names_dict = mod.feat_names_dict
    else:
        feat_names_dict = {'all': None}


    if args.regressor == 'RandomForest':
        ModelClass = RandomForestRegressor
        model_params = [{'max_features': max_feat, 'n_estimators': n_est, 'n_jobs': max(100//args.processes, 1)} for max_feat in [0.25, 0.5, 0.75, 1.0] for n_est in [25, 50, 100, 200]]
    elif args.regressor == 'LinearSVR':
        ModelClass = ScaledSVR
        model_params = [{'kernel': 'linear', 'C': c} for c in [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]]
    elif args.regressor == 'GaussianSVR':
        ModelClass = ScaledSVR
        model_params = [{'kernel': 'rbf', 'C': c, 'gamma': gam} for c in [1, 1e1, 1e2, 1e3] for gam in [1e-6, 1e-4, 1e-2, 1]]
    else:
        raise ValueError('Invalid regressor')

    res_dict = {}
    for key in feat_names_dict:
        results = runner(assets, return_results=True, feat_names=np.array(feat_names_dict[key]))  # Extract features and return only specified features for cross-validation.
        start_time = time.time()
        temp_res_dict = {}
        for model_param_dict in model_params:
            agg_stats = random_cross_validation(partial(ModelClass, **model_param_dict), results, splits=args.splits, test_fraction=0.2, processes=args.processes)
            temp_res_dict[dict_to_str(model_param_dict)] = agg_stats['stats']  # Metrics computed from each split.
            print(f'Tested params: {model_param_dict}. Time elapsed {((time.time() - start_time)/60):.2f} minutes.')
        print(f'Results - {key}')
        print_agg_stats(res_dict)
        res_dict[key] = temp_res_dict

    with open(args.out_file, 'wb') as out_file:
        pkl.dump(res_dict, out_file)


if __name__ == '__main__':
    main()
