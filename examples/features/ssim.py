from typing import Union, Tuple
from scipy.ndimage import gaussian_filter1d
import numpy as np

def ssim(x: np.ndarray, y: np.ndarray, range: int = 1, full: bool = False) -> Union[float, Tuple]:
    C1 = 0.01
    C2 = 0.03

    x = x / range
    y = y / range

    mu_x = gaussian_filter1d(gaussian_filter1d(x, 1.5, 0), 1.5, 1)
    mu_y = gaussian_filter1d(gaussian_filter1d(y, 1.5, 0), 1.5, 1)

    mu_x2 = mu_x*mu_x
    mu_y2 = mu_y*mu_y

    mu_xx = gaussian_filter1d(gaussian_filter1d(x*x, 1.5, 0), 1.5, 1)
    mu_yy = gaussian_filter1d(gaussian_filter1d(y*y, 1.5, 0), 1.5, 1)
    mu_xy = gaussian_filter1d(gaussian_filter1d(x*y, 1.5, 0), 1.5, 1)

    var_x = np.abs(mu_xx - mu_x2)
    var_y = np.abs(mu_yy - mu_y2)
    cov_xy = mu_xy - mu_x*mu_y

    l_map = (2*mu_x*mu_y + C1) / (mu_x2 + mu_y2 + C1)
    cs_map = (2*cov_xy + C2) / (var_x + var_y + C2)
    ssim_map = l_map * cs_map

    if full:
        return np.mean(l_map), np.mean(cs_map), np.mean(ssim_map)
    else:
        return np.mean(ssim_map)