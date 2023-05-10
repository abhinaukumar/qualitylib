from typing import List, Dict, Any
from types import ModuleType
import random


def read_dataset(dataset: ModuleType, shuffle: bool = True) -> List[Dict[str, Any]]:
    '''
    Read dataset module to create a list of assets in the dataset.

    Args:
        dataset: Module describing attributes of the dataset
        shuffle: Optional flag denoting whether assets should be returned in a random order. Defaults to True.

    Raises:
        AttributeError: If `dataset` does not have a `dataset_name`.
        AttributeError: If `dataset` does not have a list of `dis_videos`
        KeyError: If any element of `dis_videos` does not have a 'path'
        ValueError: If any element of `dis_videos` does not have exactly one of 'score', 'mos', or 'dmos' 
        ValueError: If a :obj:`~videolib.standards.Standard` is not provided either globally or for each video.

    Returns:
        List of asset dictionaries
    '''
    if not hasattr(dataset, 'dataset_name'):
        raise AttributeError('Dataset must have a name \'dataset_name\'.')
    dataset_name = dataset.dataset_name

    glob_standard = getattr(dataset, 'standard', None)
    glob_ref_standard = getattr(dataset, 'ref_standard', None)
    glob_dis_standard = getattr(dataset, 'dis_standard', None)
    glob_width = getattr(dataset, 'width', None)
    glob_height = getattr(dataset, 'height', None)

    if not hasattr(dataset, 'dis_videos'):
        raise AttributeError('Dataset must have a list of distorted videos \'dis_videos\'.')
    dis_videos = dataset.dis_videos

    ref_videos = getattr(dataset, 'ref_videos', {})

    asset_dicts = []
    score_key = 'score'
    valid_score_keys = ['score', 'mos', 'dmos']
    for asset_id, dis_video in dis_videos.items():
        asset_dict = {}

        asset_dict['dataset_name'] = dataset_name

        asset_dict['asset_id'] = asset_id

        asset_dict['content_id'] = dis_video.get('content_id', asset_dict['asset_id'])

        asset_dict['ref_path'] = ref_videos.get(asset_dict['content_id'], {}).get('path', None)

        if 'path' not in dis_video:
            raise KeyError('Every distorted video must have a path.')
        else:
            asset_dict['dis_path'] = dis_video['path']

        present_score_keys = set(valid_score_keys) & set(dis_video.keys())
        if len(present_score_keys) == 0:
            asset_dict['score'] = None
        elif len(present_score_keys) == 1:
            score_key = present_score_keys.pop()
            asset_dict['score'] = dis_video[score_key]
        else:
            raise ValueError('Expected exactly one of \'score\', \'mos\', or \'dmos\' to specify subjective scores.')

        asset_dict['dis_standard'] = dis_video.get('standard', glob_dis_standard if glob_dis_standard is not None else glob_standard)
        if asset_dict['dis_standard'] is None:
            raise ValueError('Standard must be specified either per video or globally for the dataset.')

        asset_dict['ref_standard'] = ref_videos.get(asset_dict['content_id'], {}).get('standard', glob_ref_standard if glob_ref_standard is not None else glob_standard)

        asset_dict['width'] = dis_video.get('width', glob_width)

        asset_dict['height'] = dis_video.get('height', glob_height)

        extra_keys = set(dis_video.keys()) - set(asset_dict.keys())
        for key in extra_keys:
            asset_dict[key] = dis_video[key]

        asset_dicts.append(asset_dict)

    if shuffle:
        random.shuffle(asset_dicts)

    return asset_dicts
