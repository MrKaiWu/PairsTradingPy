from collections import defaultdict
from typing import Union, Sequence, Dict

import bottleneck as bn
import numpy as np
import pandas as pd
from joblib import delayed
from numba import njit
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from tqdm import tqdm

from .helper import shift_2darray, frequency_convert, remove_outliers, foward_fillna_2darray
from .parallel import ProgressParallel


# %%
def calculate_beta(returns: pd.DataFrame, in_flag: pd.DataFrame, mkt: pd.DataFrame = None,
                   class_df: pd.DataFrame = None, output_freq="D", window_size=252, k=5, universe=True,
                   target_dates=None, len_beta=None, minimum_coverage=None) -> pd.DataFrame:
    """
    Beta calculation is very fast in nature. No need to utilize parallel computation
    :param returns: daily stock return DataFrame
    :param in_flag: daily in flag DataFrame
    :param mkt: daily market return DataFrame; this return is used as the benchmark (regressor) in a CAPM model;
                if it is an one row dataframe, then there is only one benchmark (e.g., equal weighted market return);
                if it has the same shape as returns, then different stocks may correspond to different benchmarks (e.g., industry return)
    :param class_df: DataFrame, the class each stock corresponds to (e.g., GICS industry);
                     by default, returns, in_flag and class_df should be of the same shape
                     and have their indexes (ID) & columns (dates) aligned
    :param output_freq: {'M', 'D'}; 'M' for monthly, 'D' for daily
    :param window_size: used in combination with output_freq; usually 252D (252 days) or 12M (12 months)
    :param k: a parameter used to determine outlier returns; only effective when mkt is None
    :param universe: if True, in_flag will be used to tell whether a stock is in the corresponding universe on a specific day
    :param target_dates: list-like, the specific dates that betas are calculated for
    :param len_beta: number of cross-sections that betas are calculated for
    :param minimum_coverage: the minimum percentage of non-missing returns needed to calculate betas
    :return: DataFrame, beta matrix
    """
    assert output_freq.lower() in ('d', 'm')
    if minimum_coverage is None or minimum_coverage > 1 or minimum_coverage < 0:
        minimum_coverage = 0.75
    if class_df is None:
        class_df = returns.values.copy()
        if universe:
            class_df[in_flag.values != 1] = np.nan
        class_df[np.isfinite(class_df)] = 1
        class_df = pd.DataFrame(class_df, index=returns.index, columns=returns.columns)
    diff_class = np.unique(class_df.values[~np.isnan(class_df)])
    period_end = frequency_convert(returns.columns, output_freq)
    period_end_idx = np.array([np.argwhere(returns.columns == x)[0, 0] for x in period_end])
    selected_idx = range(
        window_size + np.argwhere(
            (period_end_idx + 1) > np.argwhere(np.sum(np.isfinite(returns.values), axis=0) > 1)[0, 0])[0, 0],
        len(period_end_idx))
    if target_dates is None and len_beta is not None and len_beta > 0:
        selected_idx = range(selected_idx.stop - len_beta, selected_idx.stop)
    if target_dates is not None:
        target_idx = [np.argwhere(period_end == x).flatten()[0] for x in target_dates]
        selected_idx_2 = [x for x in target_idx if x in selected_idx]
        selected_idx = target_idx
    output = np.full((returns.shape[0], len(selected_idx)), np.nan)
    for col_id, c_sel_idx in tqdm(enumerate(selected_idx)):
        if target_dates is not None and c_sel_idx not in selected_idx_2:
            continue
        for c_class in diff_class:
            if universe:
                c_mask = ((in_flag.loc[:, period_end[c_sel_idx]] == 1) & (
                        class_df.loc[:, period_end[c_sel_idx]] == c_class)).values
            else:
                c_mask = (class_df.loc[:, period_end[c_sel_idx]] == c_class).values
            if c_mask.any():
                c_idx = in_flag.values[c_mask,
                        (period_end_idx[c_sel_idx - window_size] + 1):(period_end_idx[c_sel_idx] + 1)] == 1
                c_rtn = returns.values[c_mask,
                        (period_end_idx[c_sel_idx - window_size] + 1):(period_end_idx[c_sel_idx] + 1)]
                if mkt is None:
                    # if mkt is not provided, calculate the ew-market return as the market return benchmark
                    cc_rtn = c_rtn.copy()
                    cc_rtn[~c_idx] = np.nan
                    if k > 1:
                        cc_rtn = remove_outliers(cc_rtn, k=k, set_na=False)
                    else:
                        cc_rtn[cc_rtn > k] = k
                    mkt_rtn = bn.nanmean(cc_rtn, axis=0)
                else:
                    c_rtn_columns = returns.columns[
                                    (period_end_idx[c_sel_idx - window_size] + 1):(period_end_idx[c_sel_idx] + 1)]
                    dates = np.intersect1d(mkt.columns, c_rtn_columns, assume_unique=True)
                    c_rtn = pd.DataFrame(c_rtn, columns=c_rtn_columns)
                    c_rtn = c_rtn.loc[:, dates].values
                    mkt_rtn = mkt.loc[:, returns.columns[period_end_idx[c_sel_idx - window_size]]:dates[-1]]
                    mkt_rtn_cols = mkt_rtn.columns
                    mkt_rtn = mkt_rtn.values.copy()
                    if mkt_rtn.shape[0] > 1:
                        mkt_rtn = mkt_rtn[c_mask, :]
                    mkt_rtn[np.isnan(mkt_rtn)] = 0
                    wealth = np.exp(np.log(1 + mkt_rtn).cumsum(axis=1))
                    mkt_rtn = pd.DataFrame(wealth / shift_2darray(wealth, 1, axis=1) - 1, columns=mkt_rtn_cols)
                    mkt_rtn = mkt_rtn.loc[:, dates].values
                mask_beta = np.sum(np.isfinite(c_rtn), axis=1) >= (
                        period_end_idx[c_sel_idx] - period_end_idx[c_sel_idx - window_size]) * minimum_coverage
                if mask_beta.any():
                    mkt_var = bn.nanvar(mkt_rtn, ddof=1, axis=1)
                    if mkt_rtn.shape[0] == 1:
                        c_beta = pairwise_covariance(c_rtn[mask_beta], mkt_rtn) / mkt_var
                    else:
                        c_beta = pairwise_covariance(c_rtn[mask_beta], mkt_rtn[mask_beta]) / mkt_var[mask_beta]
                    output[np.argwhere(c_mask).flatten()[mask_beta], col_id] = c_beta

    return pd.DataFrame(output, index=returns.index, columns=period_end[selected_idx])


# %%
def _calculate_pairwise_statistics(values: pd.DataFrame, in_flag: pd.DataFrame, pairwise_function,
                                   class_df: pd.DataFrame = None, class_subset: Dict = None,
                                   output_freq="M", window_size=12, wealth_df: pd.DataFrame = None, universe=True,
                                   target_dates=None, log_scale=True, minimum_coverage=None, n_jobs=-1,
                                   **kwargs) -> Dict[str, Dict]:
    """
    A wrapper for the calculation of pairwise statistics, like correlation, ADF, etc.
    :param values: DataFrame (daily prices, daily returns, etc.)
    :param in_flag: daily in flag DataFrame
    :param pairwise_function: a user-defined function that implements pairwise statistics calculation
    :param class_df: DataFrame, the class each stock corresponds to (e.g., GICS industry);
                     by default, values, in_flag and class_df should be of the same shape
                     and have their indexes (ID) & columns (dates) aligned
    :param class_subset: a dict-like object used to constrain the calculation to specific subclasses; the keys are dates,
                         and the values are list-like objects containing the subset of classes that are of interest to
                         this calculation; note that calculate_beta does not have this parameter because beta calculation
                         is usually fast
    :param output_freq: {'M', 'D'}; 'M' for monthly, 'D' for daily
    :param window_size: used in combination with output_freq; usually 252D (252 days) or 12M (12 months)
    :param wealth_df: Cumulative wealth curve DataFrame used to adjust historic prices; usually needed when values is
                      the daily prices DataFrame
    :param universe: if True, in_flag will be used to tell whether a stock is in the corresponding universe on a specific day
    :param target_dates: list-like, the specific dates that betas are calculated for
    :param log_scale: if True, the values and wealth_df DataFrames will go through a log transformation
    :param minimum_coverage: the minimum percentage of non-missing returns needed to calculate betas
    :param n_jobs: number of cpu processes; -1 means using all; 1 means single process (no parallel computation);
    :param kwargs: additional keyword parameters to pass into the pairwise_function
    :return: a dictionary of dictionaries, the keys are dates and the values are dictionaries. For the sub dicts, the
             keys are classes and the values are pairwise statistics DataFrames
    """
    assert output_freq.lower() in ('d', 'm')
    if log_scale:
        values = np.log(values)
        if wealth_df is not None:
            wealth_df = np.log(wealth_df)
    if class_df is None:
        class_df = values.values.copy()
        if universe:
            class_df[in_flag.values != 1] = np.nan
        class_df[np.isfinite(class_df)] = 1
        class_df = pd.DataFrame(class_df, index=values.index, columns=values.columns)
    diff_class = np.unique(class_df.values[~np.isnan(class_df)])
    period_end = frequency_convert(values.columns, output_freq)
    period_end_idx = np.array([np.argwhere(values.columns == x)[0, 0] for x in period_end])
    selected_idx = range(
        window_size +
        np.argwhere((period_end_idx + 1) > np.argwhere(np.sum(np.isfinite(values.values), axis=0) > 1)[0, 0])[0, 0],
        len(period_end_idx))

    if target_dates is not None:
        target_idx = [np.argwhere(period_end == x).flatten()[0] for x in target_dates]
        selected_idx_2 = [x for x in target_idx if x in selected_idx]
        selected_idx = target_idx

    # An internal generator to feed data into different processes
    def map_data_generator():
        for c_sel_idx in selected_idx:
            if target_dates is not None and c_sel_idx not in selected_idx_2:
                yield None
            else:
                if class_subset is None:
                    subset = None
                else:
                    subset = class_subset[period_end[c_sel_idx]]
                if wealth_df is None:
                    wealth = None
                else:
                    wealth = wealth_df.iloc[:,
                             (period_end_idx[c_sel_idx - window_size] + 1):(period_end_idx[c_sel_idx] + 1)]
                yield (
                    values.iloc[:, (period_end_idx[c_sel_idx - window_size] + 1):(period_end_idx[c_sel_idx] + 1)],
                    in_flag.iloc[:, (period_end_idx[c_sel_idx - window_size] + 1):(period_end_idx[c_sel_idx] + 1)],
                    class_df.iloc[:, (period_end_idx[c_sel_idx - window_size] + 1):(period_end_idx[c_sel_idx] + 1)],
                    wealth,
                    subset
                )

    def map_func(data, **kwargs):
        result = {}
        if data is None:
            return result

        values, in_flag, class_df, wealth_df, class_subset = data

        for c_class in diff_class:
            if class_subset is not None:
                if c_class not in class_subset:  #
                    continue
            if universe:
                c_mask = ((in_flag.iloc[:, -1] == 1) & (
                        class_df.iloc[:, -1] == c_class)).values
            else:
                c_mask = (class_df.iloc[:, -1] == c_class).values
            if c_mask.any():
                c_vals = values.values[c_mask]
                if wealth_df is not None:
                    cum_wealth = wealth_df.values[c_mask]
                    if log_scale:
                        c_vals = c_vals[:, :1] + cum_wealth - cum_wealth[:, :1]
                    else:
                        c_vals = c_vals[:, :1] * cum_wealth / cum_wealth[:, :1]
                if minimum_coverage is not None and (0 < minimum_coverage < 1):
                    mask_stats = np.sum(np.isfinite(c_vals), axis=1) >= c_vals.shape[1] * minimum_coverage
                else:
                    mask_stats = np.sum(~np.isfinite(c_vals), axis=1) == 0
                if mask_stats.any():
                    c_stats = pairwise_function(c_vals[mask_stats], **kwargs)
                    kept_IDs = values.index[c_mask][mask_stats]
                    if isinstance(c_stats, np.recarray):
                        stat_dict = {}
                        for name in c_stats.dtype.names:
                            stat_dict[name] = pd.DataFrame(getattr(c_stats, name), index=kept_IDs,
                                                           columns=kept_IDs)
                        result[c_class] = stat_dict
                    else:
                        result[c_class] = pd.DataFrame(c_stats, index=kept_IDs, columns=kept_IDs)
        return result

    if n_jobs == -1:
        import multiprocessing
        n_jobs = multiprocessing.cpu_count()
    n_jobs = min(len(selected_idx), n_jobs)
    if n_jobs == 0:
        return {}

    with ProgressParallel(n_jobs=n_jobs, total=len(selected_idx)) as parallel:
        result_list = parallel(delayed(map_func)(data, **kwargs) for data in map_data_generator())

    result_dict = {}
    for c_sel_idx, result in zip(selected_idx, result_list):
        result_dict[period_end[c_sel_idx]] = result

    return result_dict


# %%
def calculate_corr(values, in_flag, class_df=None, class_subset=None, output_freq="M",
                   window_size=12, universe=True, wealth_df=None, target_dates=None,
                   minimum_coverage=None):
    return _calculate_pairwise_statistics(values, in_flag, pairwise_covariance, class_df=class_df,
                                          class_subset=class_subset, output_freq=output_freq,
                                          window_size=window_size, universe=universe, wealth_df=wealth_df,
                                          target_dates=target_dates, log_scale=False,
                                          minimum_coverage=minimum_coverage, correlation=True,
                                          n_jobs=1)  # use n_jobs = 1 for correlation calculation


# %%
def calculate_ADF(prices, in_flag, class_df=None, class_subset=None, output_freq="M",
                  window_size=12, universe=True, wealth_df=None, target_dates=None, log_scale=True,
                  minimum_coverage=None, n_jobs=-1, max_lag=None, auto_lag=False):
    """
    :param max_lag: if auto_lag is False, max_lag is # of lagged variables used in regression; if None, use default #
                    given by rule of thumb (see https://www.statsmodels.org/dev/generated/statsmodels.tsa.stattools.adfuller.html)
    :param auto_lag: if True, ADF test will iterate over candidates lag numbers to search for the best one,
                     which is discouraged because it is very slow
    """
    return _calculate_pairwise_statistics(prices, in_flag, pairwise_ADF_stats, class_df=class_df,
                                          class_subset=class_subset, output_freq=output_freq,
                                          window_size=window_size, universe=universe, wealth_df=wealth_df,
                                          target_dates=target_dates, log_scale=log_scale,
                                          minimum_coverage=minimum_coverage, n_jobs=n_jobs,
                                          max_lag=max_lag, auto_lag=auto_lag)


# %%
def calculate_coint_EG(prices, in_flag, class_df=None, class_subset=None, output_freq="M",
                       window_size=12, universe=True, wealth_df=None, target_dates=None, log_scale=True,
                       minimum_coverage=None, n_jobs=-1, max_lag=None, auto_lag=False):
    """
    Engel-Granger cointegration test
    :param max_lag: if auto_lag is False, max_lag is # of lagged variables used in regression; if None, use default #
                    given by rule of thumb (see https://www.statsmodels.org/dev/generated/statsmodels.tsa.stattools.adfuller.html)
    :param auto_lag: if True, ADF test (the first step of EG test )will iterate over candidates lag numbers to search for
                     the best one, which is discouraged because it is very slow
    """
    return _calculate_pairwise_statistics(prices, in_flag, pairwise_EG_coint_test, class_df=class_df,
                                          class_subset=class_subset, output_freq=output_freq,
                                          window_size=window_size, universe=universe, wealth_df=wealth_df,
                                          target_dates=target_dates, log_scale=log_scale,
                                          minimum_coverage=minimum_coverage, n_jobs=n_jobs, max_lag=max_lag,
                                          auto_lag=auto_lag)


# %%
def calculate_coint_Johansen(prices, in_flag, class_df=None, class_subset=None, output_freq="M",
                             window_size=12, universe=True, wealth_df=None, target_dates=None, log_scale=True,
                             minimum_coverage=None, n_jobs=-1, k_ar_diff=None):
    """
    :param k_ar_diff: Number of lagged differences in the model.
                      See https://www.statsmodels.org/dev/generated/statsmodels.tsa.vector_ar.vecm.coint_johansen.html
    """
    return _calculate_pairwise_statistics(prices, in_flag, pairwise_Johansen_coint_test, class_df=class_df,
                                          class_subset=class_subset, output_freq=output_freq,
                                          window_size=window_size, universe=universe, wealth_df=wealth_df,
                                          target_dates=target_dates, log_scale=log_scale,
                                          minimum_coverage=minimum_coverage, n_jobs=n_jobs,
                                          k_ar_diff=k_ar_diff)


# %%
def calculate_Hurst_exponent(prices, in_flag, class_df=None, class_subset=None, output_freq="M",
                             window_size=12, universe=True, wealth_df=None, target_dates=None, log_scale=True,
                             n_jobs=-1, minimum_coverage=None):
    return _calculate_pairwise_statistics(prices, in_flag, pairwise_Hurst_exponent, class_df=class_df,
                                          class_subset=class_subset, output_freq=output_freq,
                                          window_size=window_size, universe=universe, wealth_df=wealth_df,
                                          target_dates=target_dates, log_scale=log_scale, n_jobs=n_jobs,
                                          minimum_coverage=minimum_coverage)


# %%
def pairwise_ADF_stats(x_mat: np.ndarray, max_lag, auto_lag, is_log_price: bool = True, append_pvalue=True):
    if append_pvalue:
        adfs = np.core.records.fromarrays(
            [np.full((x_mat.shape[0], x_mat.shape[0]), fill_value=np.nan),
             np.full((x_mat.shape[0], x_mat.shape[0]), fill_value=np.nan)],
            names=['statistics', 'pvalue'])
        # adfs = np.full((x_mat.shape[0], x_mat.shape[0]), fill_value=np.nan)
    x_mat_fillna = foward_fillna_2darray(x_mat, axis=1)
    for i in range(1, x_mat.shape[0]):
        for j in range(0, i):
            if is_log_price:
                test_series = x_mat_fillna[i] - x_mat_fillna[j]
            else:
                test_series = np.log(x_mat_fillna[i] / x_mat_fillna[j])
            if np.isnan(test_series).any():
                if append_pvalue:
                    fill = -100, 1.
                else:
                    fill = -100
            else:
                adf_res = adfuller(test_series, regression='ct', maxlag=max_lag, autolag=auto_lag)
                try:
                    if append_pvalue:
                        fill = -adf_res[0], adf_res[1]
                    else:
                        fill = -adf_res[0]
                except:
                    if append_pvalue:
                        fill = -100, 1.
                    else:
                        fill = -100
            adfs[i, j] = fill
            adfs[j, i] = fill
    return adfs


# %%
def coint_EG(y: np.ndarray, x: np.ndarray, trend: str = 'c', max_lag=None, auto_lag=None, return_pvalue=True):
    if return_pvalue:
        res = coint(y, x, trend=trend, maxlag=max_lag, autolag=auto_lag)
        return -res[0], res[1]
    else:
        if len(y.shape) < 2:
            y = y[:, None]
        if len(x.shape) < 2:
            x = x[:, None]
        x = np.column_stack([np.ones_like(x), x])
        beta = np.linalg.pinv(x.T @ x) @ x.T @ y
        residual = (y - x @ beta)[:, 0]
        adf_stat = -adfuller(residual, regression='nc', maxlag=max_lag, autolag=auto_lag)[0]
        return adf_stat


def pairwise_EG_coint_test(x_mat: np.ndarray, max_lag, auto_lag, append_pvalue=True):
    if append_pvalue:
        coints = np.core.records.fromarrays(
            [np.full((x_mat.shape[0], x_mat.shape[0]), fill_value=np.nan),
             np.full((x_mat.shape[0], x_mat.shape[0]), fill_value=np.nan)],
            names=['statistics', 'pvalue'])
        # coints = np.full((2, x_mat.shape[0], x_mat.shape[0]), fill_value=np.nan)
    else:
        coints = np.full((x_mat.shape[0], x_mat.shape[0]), fill_value=np.nan)
    x_mat_fillna = foward_fillna_2darray(x_mat, axis=1)
    for i in range(1, x_mat.shape[0]):
        for j in range(0, i):
            y = x_mat_fillna[i]
            x = x_mat_fillna[j]
            if np.any(np.isnan(x)) or np.any(np.isnan(y)):
                if append_pvalue:
                    fill = -100, 1.
                else:
                    fill = -100
            else:
                try:
                    if append_pvalue:
                        fill = coint_EG(y, x, trend='c', max_lag=max_lag, auto_lag=auto_lag, return_pvalue=True)
                    else:
                        fill = coint_EG(y, x, trend='c', max_lag=max_lag, auto_lag=auto_lag, return_pvalue=False)
                except:
                    if append_pvalue:
                        fill = -100, 1.
                    else:
                        fill = -100
            coints[i, j] = fill
            coints[j, i] = fill
    return coints


# %%
def coint_Johansen(data, det_order, k_ar_diff, return_pvalue=True):
    res = coint_johansen(data, det_order, k_ar_diff)
    stat = res.lr1[0]
    if not return_pvalue:
        return stat
    else:
        levels = (0.1 - 1e-6, 0.05 - 1e-6, 0.01 - 1e-6)
        critical_values = res.cvt[0]
        where = (stat > critical_values).nonzero()[0]
        if len(where):
            pvalue = levels[where[-1]]
        else:
            pvalue = 1.
        return stat, pvalue


def pairwise_Johansen_coint_test(x_mat: np.ndarray, k_ar_diff: int = None, append_pvalue=True):
    # coints = np.full((3, x_mat.shape[0], x_mat.shape[0]), fill_value=np.nan)
    if k_ar_diff is None:
        k_ar_diff = 4
    if append_pvalue:
        coints = np.core.records.fromarrays(
            [np.full((x_mat.shape[0], x_mat.shape[0]), fill_value=np.nan),
             np.full((x_mat.shape[0], x_mat.shape[0]), fill_value=np.nan)],
            names=['statistics', 'pvalue'])
        # coints = np.full((2, x_mat.shape[0], x_mat.shape[0]), fill_value=np.nan)
    else:
        coints = np.full((x_mat.shape[0], x_mat.shape[0]), fill_value=np.nan)
    x_mat_fillna = foward_fillna_2darray(x_mat, axis=1)
    for i in range(1, x_mat.shape[0]):
        for j in range(0, i):
            data = x_mat_fillna[[i, j]]
            if np.any(np.isnan(data)):
                if append_pvalue:
                    fill = -100, 1.
                else:
                    fill = -100
            else:
                try:
                    if append_pvalue:
                        fill = coint_Johansen(data.T, 0, k_ar_diff, return_pvalue=True)
                    else:
                        fill = coint_Johansen(data.T, 0, k_ar_diff, return_pvalue=False)
                except:
                    if append_pvalue:
                        fill = -100, 1.
                    else:
                        fill = -100
            coints[i, j] = fill
            coints[j, i] = fill
    return coints


# %%
def pairwise_covariance(x_mat, y=None, correlation=False):
    x_mat = x_mat.copy()
    x_nan = np.isnan(x_mat)
    if y is not None:
        if y.shape[0] != 1:
            assert y.shape == x_mat.shape, 'y and x_mat should be of the same shape if y has more than 1 rows'
            y_mat = y
        else:
            y_mat = np.tile(y, (x_mat.shape[0], 1))
        y_nan = np.isnan(y_mat)
        x_mat[y_nan] = np.nan
        y_mat[x_nan] = np.nan
        pw_multiply = np.multiply(x_mat - bn.nanmean(x_mat, axis=1).reshape(-1, 1),
                                  y_mat - bn.nanmean(y_mat, axis=1).reshape(-1, 1))
        cov = bn.nansum(pw_multiply, axis=1) / (pw_multiply.shape[1] - np.isnan(pw_multiply).sum(axis=1) - 1)
        if correlation:
            return cov / np.multiply(bn.nanstd(x_mat, axis=1, ddof=1), bn.nanstd(y_mat, axis=1, ddof=1))
        return cov
    else:
        if correlation:
            return pd.DataFrame(x_mat).T.corr().values
        return pd.DataFrame(x_mat).T.cov().values


# %%
@njit
def hurst_exponent(series, q=2):
    L = len(series)
    H = np.zeros((len(range(5, 20)), 1))
    k = 0

    for Tmax in range(5, 20):
        x = np.arange(1, Tmax + 1, 1)
        mcord = np.zeros((Tmax, 1))

        for tt in range(1, Tmax + 1):
            dV = series[tt:L:tt] - series[0:(L - tt):tt]
            VV = series[0:L:tt]
            N = len(dV) + 1
            X = np.arange(1, N + 1)
            Y = VV
            mx = (1 + N) / 2
            SSxx = np.sum(X ** 2) - N * mx ** 2
            my = VV.mean()
            SSxy = np.sum(np.multiply(X, Y)) - N * mx * my
            cc1 = SSxy / SSxx
            cc2 = my - cc1 * mx
            ddVd = dV - cc1
            VVVd = VV - np.multiply(cc1, X) - cc2
            mcord[tt - 1] = np.mean(np.abs(ddVd) ** q) / np.mean(np.abs(VVVd) ** q)

        mx = np.mean(np.log10(x))
        SSxx = np.sum(np.log10(x) ** 2) - Tmax * mx ** 2
        my = np.mean(np.log10(mcord))
        SSxy = np.sum(np.multiply(np.log10(x), np.transpose(np.log10(mcord)))) - Tmax * mx * my
        H[k] = SSxy / SSxx
        k = k + 1

    mH = np.mean(H) / q

    return mH


def pairwise_Hurst_exponent(x_mat: np.ndarray, is_log_price: bool = True):
    hursts = np.full((x_mat.shape[0], x_mat.shape[0]), fill_value=np.nan)
    x_mat_fillna = foward_fillna_2darray(x_mat, axis=1)
    for i in range(1, x_mat.shape[0]):
        for j in range(0, i):
            if is_log_price:
                test_series = x_mat_fillna[i] - x_mat_fillna[j]
            else:
                test_series = np.log(x_mat_fillna[i] / x_mat_fillna[j])
            if np.isnan(test_series).any():
                fill = 1.
            else:
                fill = hurst_exponent(test_series, 2)
            hursts[i, j] = fill
            hursts[j, i] = fill
    return hursts


# %%
def restructure_distance_mat(distance_dict: Dict[str, Dict[str, pd.DataFrame]],
                             target_dates: Union[Sequence[str], np.ndarray], in_flag: pd.DataFrame=None,
                             universe=True):
    """
    Used in PairFilter._distance_metric; Will only keep the data for stocks within the universe in a point-in-time
    manner if universe is True and in_flag DataFrame is provided
    """
    result_dict = defaultdict(dict)
    for date in tqdm(target_dates):
        sub_dict = distance_dict[date]
        if universe and in_flag is not None:
            for c, df in sub_dict.items():
                mark = (in_flag.loc[df.index, date] == 1).values
                if any(mark):
                    result_dict[date][c] = df.loc[mark, mark]
        else:
            result_dict[date] = sub_dict
    return result_dict
