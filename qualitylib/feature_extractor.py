from typing import Any, Dict, List, Type, Optional

import numpy as np

from .result import Result


class FeatureExtractor:
    '''
    Class defining a feature extractor.
    '''
    NAME = 'DefaultFex'  #: Unique name of feature extractor.
    VERSION = '1.0'  #: Unique version of feature extractor.
    feat_names = np.empty(shape=(0,), dtype='<U1')  #: Optional array of feature names.

    def __init__(self, use_cache: bool = True, sample_rate: Optional[int] = None) -> None:
        '''
        Args:
            use_cache: Flag denoting whether to save/load results to/from cache. Defaults to True.
            sample_rate: Framerate at which features are to be extracted. Defaults to None, i.e., extracting from all frames.
        '''
        self.use_cache = use_cache
        self.sample_rate = sample_rate

    @property
    def name_version(self) -> str:
        '''
        '<name>_V<version>' formatted string
        '''
        return f'{self.NAME}_V{self.VERSION}'

    @staticmethod
    def _to_result(asset_dict: Dict[str, Any], feats: np.ndarray, feat_names: np.ndarray = np.empty(shape=(0,), dtype='<U1')) -> Result:
        '''
        Method to create :obj:`~qualitylib.results.Result` from extracted features and asset data.

        Args:
            asset_dict: Asset for which features were extracted
            feats: Extracted features
            feat_names: Optional array of feature names.

        Returns:
            Result: :obj:`~qualitylib.results.Result` object containing extracted features and asset data
        '''
        return Result(
            dataset_name=asset_dict['dataset_name'],
            content_id=asset_dict['content_id'],
            asset_id=asset_dict['asset_id'],
            feats=feats,
            feat_names=feat_names,
            score=asset_dict['score']
        )

    def _get_sample_interval(self, asset_dict: Dict[str, Any]) -> int:
        '''
        Calculate sampling interval.

        Args:
            asset_dict: Asset

        Returns:
            Sampling interval
        '''
        framerate = asset_dict.get('fps', 30)  # Default frame rate is 30 fps if unspecified
        return framerate // self.sample_rate if self.sample_rate is not None else 1

    def _run_on_asset(self, asset_dict: Dict[str, Any]) -> Result:
        '''
        Run feature extractor on asset.

        Args:
            asset_dict: Input asset.

        Returns:
            Result containing extracted features and asset data.
        '''
        pass

    def run(self, asset_dict: Dict[str, Any], return_results: bool = True, feat_names: Optional[np.ndarray] = None) -> Result:
        '''
        Extract features from asset.

        Args:
            asset_dict: Input asset
            return_results: Flag denoting whether :obj:`~qualitylib.result.Result` is to be returned. Defaults to True.
            feat_names: Names of features to be returned. Defaults to None - all features are returned.

        Raises:
            RuntimeError: When feature extraction encounters any errors.

        Returns:
            Result containing extracted features and asset data.
        '''
        if self.use_cache:
            try:
                if return_results:
                    result = Result.from_asset_and_fex(asset_dict, self.name_version)
                    if feat_names is not None:
                        result.feats = result[feat_names, 'feats']
                        result.feat_names = feat_names
                else:
                    Result.exists_from_asset_and_fex(asset_dict, self.name_version)
                    result=None
                print(f'Loaded result from cache for {asset_dict["dis_path"]}')
            except OSError:
                result = self._run_on_asset(asset_dict)
                result.save(Result.get_path(asset_dict, self.name_version))
            except Exception as err:
                raise RuntimeError(f'Unexpected error when loading result for {str(asset_dict)} from fex: {self.name_version}.') from err
        else:
            result = self._run_on_asset(asset_dict)

        return result

    def __call__(self, *args: List[Any], **kwargs: Dict[str, Any]) -> Result:
        '''
        Wrapper around :obj:`~qualitylib.feature_extractor.FeatureExtractor.run` to make :obj:`~qualitylib.feature_extractor.FeatureExtractor` object callable.

        Args:
            args: Positional arguments passed to :obj:`~qualitylib.feature_extractor.FeatureExtractor.run`
            kwargs: Keyword arguments passed to :obj:`~qualitylib.feature_extractor.FeatureExtractor.run`

        Returns:
            Result containing extracted features and asset data.
        '''
        return self.run(*args, **kwargs)

    @classmethod
    def _get_all_subclasses(cls) -> List[Type]:
        '''
        Recursively find all subclasses

        Returns:
            List of subclasses
        '''
        return cls.__subclasses__() + sum([sub._get_all_subclasses() for sub in cls.__subclasses__()], [])


def get_fex(name: str, version: Optional[str] = None):
    '''
    Get :obj:`~qualitylib.feature_extractor.FeatureExtractor` class from name and version.

    Args:
        name: Name of the feature extractor.
        version: Optional version of the feature extractor.
    '''
    fexs = FeatureExtractor._get_all_subclasses()
    matching_fexs = list(filter(lambda x: x.NAME == name, fexs))
    if version is not None:
        matching_fexs = list(filter(lambda x: x.VERSION == version, matching_fexs))
    if len(matching_fexs) == 0:
        raise RuntimeError(f'Found no matching feature extractors for (name={name}, version={version})')
    elif len(matching_fexs) != 1:
        raise RuntimeError(f'Found more than one matching feature extractor for (name={name}, version={version})')
    else:
        return matching_fexs[0]
