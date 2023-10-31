from typing import Dict, Any
import numpy as np

from videolib import Video

from qualitylib.feature_extractor import FeatureExtractor
from features import ssim
from qualitylib.result import Result


class SsimFeatureExtractor(FeatureExtractor):
    NAME = 'SSIM_fex'
    VERSION = '1.0'
    feat_names = np.array(['lum', 'cs', 'ssim'])  # Leave empty if feature names are not to be provided

    def _run_on_asset(self, asset_dict: Dict[str, Any]) -> Result:
        sample_interval = self._get_sample_interval(asset_dict)  # Temporal subsampling
        feats_dict = {key: [] for key in self.feat_names}  # Use list/array if no feat_names are used
        with Video(
                asset_dict['ref_path'], mode='r',
                standard=asset_dict['ref_standard'],
                width=asset_dict['width'], height=asset_dict['height']
        ) as v_ref:
            with Video(
                asset_dict['dis_path'], mode='r',
                standard=asset_dict['dis_standard'],
                width=asset_dict['width'], height=asset_dict['height']
             ) as v_dis:
                # for i, (frame_ref, frame_dis) in enumerate(zip(v_ref, v_dis)):
                #     if i % sample_interval:
                #         continue
                #     lum_val, cs_val, ssim_val = ssim(frame_ref, frame_dis, full=True)
                #     feats_dict['lum'].append(lum_val)
                #     feats_dict['cs'].append(cs_val)
                #     feats_dict['ssim'].append(ssim_val)
                feats = np.random.randn(v_ref.num_frames, 3)

        # feats = np.array(list(feats_dict.values())).T
        print(f'Processed {asset_dict["dis_path"]}')
        return self._to_result(asset_dict, feats, list(feats_dict.keys()))
