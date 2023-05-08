from typing import List, Dict, Callable, Any
import numpy as np
from joblib import Parallel, delayed

from .tools import stats
from .result import Result


def run_split(
        model_builder: Callable,
        feats: np.ndarray,
        labels: np.ndarray,
        test_mask: np.ndarray,
        test_content_labels: np.ndarray,
        eval_content_wise: bool
    ) -> Dict[str, float]:
    '''
    Run one random cross-validation split

    Args:
        model_builder: A callable that returns a new model object that contains the `fit` and `predict` methods.
        feats: Input features
        labels: Ground truth labels
        test_mask: Boolean mask denoting test samples
        test_content_labels: Labels denoting source content of each sample
        eval_content_wise: Evaluaate accuracy for each content separately

    Returns:
        Accuracy stats
    '''

    test_feats = feats[test_mask]
    train_feats = feats[~test_mask]
    test_labels = labels[test_mask]
    train_labels = labels[~test_mask]
    model = model_builder()
    model.fit(train_feats, train_labels)
    test_preds = np.squeeze(model.predict(test_feats))
    test_labels = np.squeeze(test_labels)
    test_content_labels = np.squeeze(test_content_labels)

    split_stats = {key: funct(test_preds, test_labels) for key, funct in zip(['PCC', 'SROCC', 'RMSE'], [stats.pcc, stats.srocc, stats.rmse])}

    # Sometimes, calculating accuracy for each source content and averaging is useful.
    if eval_content_wise:
        test_contents = np.unique(test_content_labels)
        cont_stats = {key: [] for key in ['Cont_PCC', 'Cont_SROCC', 'Cont_RMSE']}  
        for cont in test_contents:
            cont_mask = test_content_labels == cont
            for key, funct in zip(['Cont_PCC', 'Cont_SROCC', 'Cont_RMSE'], [stats.pcc, stats.srocc, stats.rmse]):
                cont_stats[key].append(funct(test_preds[cont_mask], test_labels[cont_mask]))
        for key in cont_stats:
            split_stats[key] = np.mean(cont_stats[key])

    return split_stats


def random_cross_validation(
        model_builder: Callable,
        results: List[Result],
        test_fraction: float = 0.2,
        splits: int = 100,
        content_aware: bool = True,
        eval_content_wise: bool = False,
        processes: int = 1
    ) -> Dict[str, Any]:
    '''
    Run random cross-validation

    Args:
        model_builder: A callable that returns a new model object that contains the `fit` and `predict` methods.
        results: List of :obj:`~qualitylib.result.Result` objects denoting each asset.
        test_fraction: Fraction of samples/contents to be used in test sets. Defaults to 0.2.
        splits: Number of random splits. Defaults to 100.
        content_aware: Flag denoting whether train-test splits must be content-aware. Defaults to True.
        eval_content_wise: Evaluaate accuracy for each content separately. Defaults to False.
        processes: Number of parallel processes to use. Defaults to 1.

    Returns:
        Accuracy stats aggregated over all splits.
    '''

    labels = np.array(list(map(lambda result: result.score, results)))
    if np.any(labels == None) or np.any(np.isnan(labels) | np.isinf(labels)):
        raise ValueError('Scores for all assets must be numerical values')

    feats = np.stack(list(map(lambda result: result.agg_feats, results)), 0)
    content_ids = np.array(list(map(lambda result: result.content_id, results)))
    test_masks = []
    test_content_ids = []
    if content_aware:
        unique_content_ids = np.unique(content_ids)
        num_contents = len(unique_content_ids)
        num_test_contents = int(np.ceil(num_contents * test_fraction))
        for split in range(splits):
            np.random.shuffle(unique_content_ids)
            test_contents = unique_content_ids[:num_test_contents]
            test_mask = np.array(list(map(lambda result: result.content_id in test_contents, results)))
            test_masks.append(test_mask)
            test_content_ids.append(content_ids[test_mask])
    else:
        num_assets = len(results)
        asset_inds = np.arange(num_assets)
        num_test_assets = int(np.ceil(num_assets * test_fraction))
        for split in range(splits):
            np.random.shuffle(asset_inds)
            test_mask = np.ones((num_assets,), dtype=bool)
            test_mask[asset_inds[:num_test_assets]] = True
            test_masks.append(test_mask)
            test_content_ids.append(np.unique(content_ids[test_mask]))

    if processes == 1:
        stats = [run_split(model_builder, feats, labels, test_mask, test_content_label) for test_mask, test_content_label in zip(test_masks, test_content_ids)]
    else:
        stats = Parallel(n_jobs=processes)(delayed(run_split)(model_builder, feats, labels, test_mask, test_content_label, eval_content_wise) for test_mask, test_content_label in zip(test_masks, test_content_ids))

    agg_stats = {'stats': stats}
    for key in stats[0]:
        agg_stats[f'med_{key}'] = np.median([stat[key] for stat in stats])

    return agg_stats
