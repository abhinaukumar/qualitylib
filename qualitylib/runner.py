from typing import Optional, List, Dict, Any, Callable

from joblib import Parallel, delayed
import numpy as np

from .feature_extractor import get_fex
from .result import Result


class Runner:
    '''
    Wrapper class around :obj:`~qualitylib.feature_extractor.FeatureExtractor` to run feature extraction.
    '''
    def __init__(self, FexClass: Callable, processes: int = 1, *args: Any, **kwargs: Any) -> None:
        '''
        Args:
            FexClass: Class derived from :obj:`~qualitylib.feature_extractor.FeatureExtractor`.
            processes: Number of parallel processes to use. Defaults to 1.
        '''
        self.fex = FexClass(*args, **kwargs)
        self.processes = processes

    def run(self, asset_dicts: List[Dict[str, Any]], return_results: bool = True, feat_names: Optional[np.ndarray] = None) -> List[Result]:
        '''
        Run feature extraction on assets.

        Args:
            asset_dicts: List of assets.
            return_results: Flag denoting whether results must be returned. Defaults to True.
            feat_names: Names of features to return. Defaults to None - all features are returned.

        Returns:
            List of :obj:`~qualitylib.result.Result` objects, each containing the result for one asset.
        '''
        results = Parallel(n_jobs=self.processes)(delayed(self.fex.run)(asset_dict, return_results, feat_names) for asset_dict in asset_dicts)
        return results

    def __call__(self, *args: List[Any], **kwargs: Dict[str, Any]) -> List[Result]:
        '''
        Wrapper around :obj:`~qualitylib.runner.Runner.run` to make :obj:`~qualitylib.runner.Runner` object callable.

        Args:
            args: Positional arguments passed to :obj:`~qualitylib.runner.Runner.run`
            kwargs: Keyword arguments passed to :obj:`~qualitylib.runner.Runner.run`

        Returns:
            List of :obj:`~qualitylib.result.Result` objects, each containing the result for one asset.
        '''
        return self.run(*args, **kwargs)
