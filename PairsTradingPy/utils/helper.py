import datetime as dt
from operator import itemgetter
from typing import Union, List

import numpy as np
import pandas as pd


# %%
def foward_fillna_2darray(array, axis=1):
    """
    An implementation of forward fill NA for 2-d array
    """
    mask = np.isnan(array)
    if axis == 1:
        indices = np.where(~mask, np.arange(mask.shape[1]), 0)
        np.maximum.accumulate(indices, axis=1, out=indices)
        return array[np.arange(indices.shape[0])[:, None], indices]
    else:
        indices = np.where(~mask, np.arange(mask.shape[0])[:, None], 0)
        np.maximum.accumulate(indices, axis=0, out=indices)
        return array[indices, np.arange(indices.shape[1])]


# %%
def shift_2darray(array, by=1, axis=1, fill=np.nan):
    """
    Similar to LQuant wLag
    """
    assert axis in (0, 1)
    new_arr = np.roll(array, shift=by, axis=axis)
    if axis == 0:
        if by > 0:
            new_arr[:by, :] = fill
        else:
            new_arr[by:, :] = fill
    else:
        if by > 0:
            new_arr[:, :by] = fill
        else:
            new_arr[:, by:] = fill
    return new_arr


# %%
def frequency_convert(factor_dates: Union[np.ndarray, pd.Index, List], output_freq='M', period_end=True, skip=0):
    """
    An replication of LQuant frequency_convert
    """
    dates_obj = [dt.datetime.strptime(x, "%Y-%m-%d").date() for x in factor_dates]
    if output_freq == 'A':
        period_symb = np.array([x.strftime('%Y') for x in dates_obj])
    elif output_freq == 'Q':
        map_func = lambda x: ['Q1', 'Q2', 'Q3', 'Q4'][(x - 1) // 3]
        period_symb = np.array([map_func(int(x.strftime('%m'))) for x in dates_obj])
    elif output_freq == 'M':
        period_symb = np.array([x.strftime('%B') for x in dates_obj])
    elif output_freq == 'W':
        period_symb = np.array([int(x.strftime('%U')) for x in dates_obj])
    else:
        period_symb = factor_dates.copy()
    rebalance_dates = np.argwhere(period_symb[1:] != period_symb[:-1]).flatten()
    if len(rebalance_dates) == 0:
        rebalance_dates = np.argwhere(period_symb == period_symb[0]).flatten()[[-1]]
    else:
        if period_end:
            avg_gap = np.mean(rebalance_dates[1:] - rebalance_dates[:-1])
            if np.abs(len(period_symb) - 1 - rebalance_dates.max() - avg_gap) <= avg_gap / 3.:
                rebalance_dates = np.r_[(rebalance_dates, len(period_symb) - 1)]
        else:
            rebalance_dates = np.r_[(0, rebalance_dates + 1)]
    if skip:
        assert len(rebalance_dates) >= skip + 1, "skip too many periods"
        rebalance_dates = rebalance_dates[skip:]
    if isinstance(factor_dates, List):
        if len(rebalance_dates) == 1:
            return [factor_dates[rebalance_dates[0]], ]
        else:
            return list(itemgetter(*rebalance_dates)(factor_dates))
    else:
        if len(rebalance_dates) == 1:
            return factor_dates[[rebalance_dates[0]]]
        else:
            return factor_dates[rebalance_dates]


# %%
def remove_outliers(input_data: Union[np.ndarray, pd.DataFrame, pd.Series], k=3., set_na=True, log_scale=False,
                    pruning='both'):
    """
        An replication of LQuant remove_outliers
    """
    rem_top = pruning == 'both' or pruning == 'top_only'
    rem_bot = pruning == 'both' or pruning == 'bottom_only'
    assert rem_top or rem_bot
    is_Series, is_DF = False, False
    dim = None
    if input_data.ndim == 1:
        dim = 1
        if isinstance(input_data, pd.Series):
            Series_idx = input_data.index
            Series_name = input_data.name
            input_data = input_data.values
            is_Series = True
    elif input_data.ndim == 2:
        dim = 2
        if isinstance(input_data, pd.DataFrame):
            DF_idx = input_data.index
            DF_columns = input_data.columns
            input_data = input_data.values
            is_DF = True
    else:
        raise Exception('cannot handle data with 3 or more dimensions')
    if log_scale:
        input_data = np.log(input_data)
    thresh = np.nanquantile(input_data, (0.25, 0.75), axis=0)
    upper_bound = thresh[1] + (thresh[1] - thresh[0]) * k
    lower_bound = thresh[0] - (thresh[1] - thresh[0]) * k
    if set_na:
        if rem_top:
            input_data[input_data > upper_bound] = np.nan
        if rem_bot:
            input_data[input_data < lower_bound] = np.nan
    else:
        if rem_top:
            if dim == 1:
                input_data[input_data > upper_bound] = upper_bound
            else:
                mask = input_data > upper_bound
                input_data[mask] = np.tile(upper_bound, (input_data.shape[0], 1))[mask]
        if rem_bot:
            if dim == 1:
                input_data[input_data < lower_bound] = lower_bound
            else:
                mask = input_data < lower_bound
                input_data[mask] = np.tile(lower_bound, (input_data.shape[0], 1))[mask]
    if log_scale:
        input_data = np.exp(input_data)
    if is_Series:
        return pd.Series(input_data, index=Series_idx, name=Series_name)
    elif is_DF:
        return pd.DataFrame(input_data, index=DF_idx, columns=DF_columns)
    else:
        return input_data
