from scipy.stats import spearmanr, pearsonr
import numpy as np
import numpy.typing

def pcc(y_preds: numpy.typing.ArrayLike, y_trues: numpy.typing.ArrayLike, err_default: float = 0) -> float:
    '''
    Pearson Correlation Coefficient (PCC)

    Args:
        y_preds: Predicted values
        y_trues: True values
        err_default: Value to return if PCC computation fails. Defaults to 0.

    Returns:
        PCC between predicted and true values
    '''
    ret = pearsonr(y_preds, y_trues)[0]
    if np.isnan(ret) or np.isinf(ret):
        ret = err_default
    return ret


def srocc(y_preds: numpy.typing.ArrayLike, y_trues: numpy.typing.ArrayLike, err_default: float = 0) -> float:
    '''
    Spearmanr Rank Order Correlation Coefficient (SROCC)

    Args:
        y_preds: Predicted values
        y_trues: True values
        err_default: Value to return if SROCC computation fails. Defaults to 0.

    Returns:
        SROCC between predicted and true values
    '''
    ret = spearmanr(y_preds, y_trues)[0]
    if np.isnan(ret) or np.isinf(ret):
        ret = err_default
    return ret


def rmse(y_preds: numpy.typing.ArrayLike, y_trues: numpy.typing.ArrayLike, err_default: float = 100) -> float:
    '''
    Root Mean Squared Error (RMSE)

    Args:
        y_preds: Predicted values
        y_trues: True values
        err_default: Value to return if RMSE computation fails. Defaults to 100.

    Returns:
        RMSE between predicted and true values
    '''
    ret = np.sqrt(np.mean((y_preds - y_trues)**2))
    if np.isnan(ret) or np.isinf(ret):
        ret = err_default
    return ret
