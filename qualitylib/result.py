from __future__ import annotations
from typing import Dict, Tuple, Any, Optional, Union

import os

import numpy as np
from scipy.io import loadmat, savemat


result_store_dir = 'result_store'  #: Cache path where all results are stored.


class Result:
    '''
    Result object containing extracted features and asset data.
    '''
    def __init__(
            self,
            dataset_name: str,
            content_id: int,
            asset_id: int,
            feats: np.ndarray,
            feat_names: np.ndarray = np.empty(shape=(0,), dtype='<U1'),
            agg_method: Union[str, callable] = 'mean',
            score: Optional[float] = None
        ) -> None:
        '''
        Args:
            dataset_name: Name of dataset.
            content_id: ID of source content of input asset.
            asset_id: ID of input asset from which features were extracted.
            feats: Array of features extracted from asset.
            feat_names: Optional names of extracted features.
            agg_method: Temporal aggregation method. Defaults to 'mean'.
            score: Optional ground-truth subjective score. Defaults to None.

        Raises:
            ValueError: If `feats` is not a 2D array.
            ValueError: If `agg_method` is neither a recognized string, nor a :obj:`Callable`.
            ValueError: If the number of features in `feats` does not match the number of feature names, if provided. 
        '''

        self.dataset_name = dataset_name
        self.content_id = content_id
        self.asset_id = asset_id
        self.score = score

        if not isinstance(feats, np.ndarray):
            feats = np.array(feats)
        if not isinstance(feat_names, np.ndarray):
            feat_names = np.array(feat_names)

        if feats.ndim != 2:
            raise ValueError(f'Expected 2-D feats array, received {feats.ndim}-D array.')
        self.feats = feats  # N x F numpy array, where N is number of frames, F is number of feats

        if isinstance(agg_method, str):
            if agg_method not in ['mean', 'std', 'cov']:
                raise ValueError(f'When agg_method is a string, it must be one of \'mean\', \'std\' or \'cov\'. Provided {agg_method}')
        elif not callable(agg_method):
            raise ValueError(f'agg_method must be either a valid string or a callable. Provided {agg_method} of type {type(agg_method)}')

        self.agg_method = agg_method
        self._agg_feats = None

        if feat_names.ndim > 0 and len(feat_names) > 0 and len(feat_names) != feats.shape[1]:
            raise ValueError(f'Length of feat_names {len(feat_names)} does not match number of features {feats.shape[1]}')
        self.feat_names = feat_names

    @staticmethod
    def get_path(asset_dict: Dict[str, Any], fex_name_version: str) -> os.PathLike:
        '''
        Get path in cache to/from which result is saved/loaded.

        Args:
            asset_dict: Input asset.
            fex_name_version: Name-version string of feature extractor.

        Returns:
            Path of result in cache.
        '''
        return os.path.join(result_store_dir, fex_name_version, asset_dict['dataset_name'], str(asset_dict['content_id']), str(asset_dict['asset_id']) + '.mat')

    @classmethod
    def from_dict(cls, result_dict: Dict[str, Any]) -> Result:
        '''
        Create result from dictionary.

        Args:
            result_dict: Dictionary containing result attributes.

        Returns:
            Result object.
        '''
        return cls(**result_dict)

    @classmethod
    def load(cls, path: os.PathLike) -> Result:
        '''
        Load result from path.

        Args:
            path: Path to result in cache.

        Raises:
            OSError: If path is not a valid readable file path

        Returns:
            Loaded result object
        '''
        if os.path.isfile(path):
            result_dict = loadmat(path)
            unwanted_keys = list(filter(lambda key: key[:2] == '__', result_dict.keys()))
            for key in unwanted_keys:
                del result_dict[key]
            for key in ['dataset_name', 'content_id', 'asset_id', 'agg_method', 'score']:
                if key in result_dict:
                    result_dict[key] = result_dict[key].ravel()[0]
            result_dict['feat_names'] = np.array(list(map(lambda s: s.rstrip(), result_dict['feat_names'])))
            return cls.from_dict(result_dict)
        else:
            raise OSError(f'Result file {path} does not exist')

    @classmethod
    def from_asset_and_fex(cls, asset_dict: Dict[str, Any], fex_name_version: str) -> Result:
        '''
        Load result from input asset and name-version string of the :obj:`~qualitylib.feature_extractor.FeatureExtractor` class.

        Args:
            asset_dict: Input asset
            fex_name_version: Name-version string of the :obj:`~qualitylib.feature_extractor.FeatureExtractor` class.

        Raises:
            OSError: If path is not a valid readable file path

        Returns:
            Loaded result object
        '''
        result_path = Result.get_path(asset_dict, fex_name_version)
        result = cls.load(result_path)
        if result.score is None:
            result.score = asset_dict['score']
        return result

    @staticmethod
    def exists(asset_dict: Dict[str, Any], fex_name_version: str) -> bool:
        '''
        Check if result file exists in cache.

        Args:
            asset_dict: Input asset.
            fex_name_version: Name-version string of the :obj:`~qualitylib.feature_extractor.FeatureExtractor` class.

        Returns:
            bool: Flag denoting whether the result file exists in the cache.
        '''
        return os.path.isfile(Result.get_path(asset_dict, fex_name_version))

    def save(self, path: os.PathLike) -> None:
        '''
        Save result to path.

        Args:
            path: Path to which result must be saved.
        '''
        os.makedirs(os.path.dirname(path), exist_ok=True)
        result_dict = {
            'dataset_name': self.dataset_name,
            'content_id': self.content_id,
            'asset_id': self.asset_id,
            'feats': self.feats,
            'feat_names': self.feat_names,
            'agg_method': self.agg_method
        }
        if self.score is not None:
            result_dict['score'] = self.score
        savemat(path, result_dict)

    @property
    def num_frames(self) -> int:
        '''
        Number of frames for which features are extracted.
        '''
        return self.feats.shape[0]

    @property
    def num_feats(self) -> int:
        '''
        Number of extracted features.
        '''
        return self.feats.shape[1]

    @property
    def feats(self) -> np.ndarray:
        '''
        Array of extracted features from each frame.
        '''
        return self._feats

    @feats.setter
    def feats(self, value) -> None:
        '''
        Setter for `feats`.
        '''
        self._feats = value
        self._agg_feats = None

    @property
    def agg_feats(self) -> np.ndarray:
        '''
        Aggregated features, using the `agg_method`.
        '''
        if self._agg_feats is None:
            if self.agg_method == 'mean':
                self._agg_feats = np.mean(self.feats, 0)
            elif self.agg_method == 'std':
                self._agg_feats = np.std(self.feats, 0)
            elif self.agg_method == 'cov':
                self._agg_feats = np.std(self.feats, 0) / np.mean(self.feats, 0)
            else:
                self._agg_feats = self.agg_method(self.feats)

        return self._agg_feats

    def __getitem__(self, key: Union[slice, int, str, np.ndarray, Tuple[Union[slice, int, str, np.ndarray], str]]) -> np.ndarray:
        '''
        Selecting and slicing features from result.

        Args:
            key: Input key/index/slice.

        Raises:
            ValueError: If index/key/slice is of an invalid format.
            KeyError: If index/key/slice is not found.
            TypeError: If index/key/slice is of an invalid datatype.

        Returns:
            Selected (aggregated) features
        '''
        default_feat_type = 'agg_feats'
        if isinstance(key, tuple):
            if len(key) > 2:
                raise ValueError('Expected at most 2 indices/keys')
            if len(key) == 1:
                key = key[0]
                feat_type = default_feat_type
            else:
                key, feat_type = key
        else:
            feat_type = default_feat_type

        if isinstance(key, slice):
            if feat_type == 'agg_feats':
                return self.feats[slice.start:slice.stop:slice.step]
            elif feat_type == 'feats':
                return self.feats[:, slice.start:slice.stop:slice.step]
            else:
                raise KeyError(f'Invalid feature type {feat_type}')
        elif isinstance(key, int):
            if feat_type == 'agg_feats':
                return self.agg_feats[key]
            elif feat_type == 'feats':
                return self.feats[:, key]
            else:
                raise KeyError(f'Invalid feature type {feat_type}')
        elif isinstance(key, str):
            if key not in self.feat_names:
                raise KeyError(f'Could not find key {key} in feat_names')
            ind = np.where(self.feat_names == key)[0][0]
            if feat_type == 'agg_feats':
                return self.agg_feats[ind]
            elif feat_type == 'feats':
                return self.feats[:, ind]
            else:
                raise KeyError(f'Invalid feature type {feat_type}')
        elif isinstance(key, np.ndarray):
            if key.dtype.type == np.integer:
                return self.feats[:, key]
            elif key.dtype.type == np.str_:
                inds = np.concatenate([np.where(self.feat_names == keyval)[0] for keyval in key])
                if len(inds) != len(key):
                    raise KeyError('Not all provided keys exist in feat_names')
                if feat_type == 'agg_feats':
                    return self.agg_feats[inds]
                elif feat_type == 'feats':
                    return self.feats[:, inds]
                else:
                    raise KeyError(f'Invalid feature type {feat_type}')
            else:
                raise TypeError('Index/key array must be an integer or string array')
        else:
            raise TypeError(f'Index/key must be an (array of) integer(s) or string(s), or a slice. Got {type(key)} instead')

# TODO: Create infrastructure to handle collections of results
