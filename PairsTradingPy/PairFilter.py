import re
from collections import defaultdict
from functools import reduce
from functools import wraps
from hashlib import md5
from operator import and_, or_
from typing import Union, Sequence, Dict

import numpy as np
import pandas as pd

from .DataLoader import DataLoader
from .utils import calculate_beta, calculate_corr, frequency_convert, calculate_ADF, calculate_coint_EG, \
    restructure_distance_mat, calculate_coint_Johansen, calculate_Hurst_exponent


class FilterConditions:
    """
    Support the composition of pair filter conditions in a chained fashion
    """
    # make sure to extend this list if you add another statistical test with pvalue available
    OPERATIONS_WITH_PVALUE = ['ADF', 'coint_EG', 'coint_Johansen']

    # TIME_CONSUMING_OPERATIONS = ['ADF']

    class _DataRepr:
        """
        A helper class to support customized data (e.g., distance matrices)
        """

        def __init__(self, data, name):
            self.data = data
            self.name = name

        def __repr__(self):
            return self.name

        def __call__(self):
            return self.data

    def __init__(self, PairFilterObj):
        self.PF = PairFilterObj
        # each condition is expressed with a tuple of two elements: the first element is the name of the condition,
        # the second element is the attribute dict (usually named attr somewhere else in the package)
        self.conditions = []

        self._stack = []
        self.same_class_conditions = []
        self.primary_class_conditions = []
        self.subset_class_conditions = []
        self._added_conds = []

    def __repr__(self):
        return repr(self.same_class_conditions + self.subset_class_conditions + self.conditions + self._stack)

    def __hash__(self):
        """
        This hash function is used when storing pair selection results in the PairFilter.pair_chache dict. The hash value
        serves as a key. It is not user friendly to use a potentially verbose string representation of this FilterCondition
        object (given by __repr__) as the key.
        """
        a = sorted(self.primary_class_conditions, key=lambda x: x[0])
        b = sorted(self.same_class_conditions, key=lambda x: x[0])
        c = sorted(self.subset_class_conditions, key=lambda x: x[0])
        d = sorted(self.conditions, key=lambda x: x[0])
        return int(md5(bytes(repr(a + b + c + d), 'utf8')).hexdigest(), 16)

    def _clear_stack(self):
        if self._stack:
            self.conditions.append(tuple(self._stack.pop()))

    def _check_stack(self, func_name):
        assert len(
            self._stack) == 1, '"{}" cannot be used. The corresponding main condition has not been specified.'.format(
            func_name)

    def _check_pending_cond(self):
        if self._stack:
            assert any(x in self._stack[0][1] for x in ['le', 'lt', 'ge', 'gt', 'top', 'bottom', 'pvalue']), \
                'at least one of the following should be called to make the previous {} condition valid: \n\t\t\t\t{}'.format(
                    self._stack[0][0], ['greater_than', 'less_than', 'top', 'bottom'])

    def _check_existing_conds(self, cond_name):
        assert cond_name not in self._added_conds, '{} condition already exists'.format(cond_name)

    def same_industry(self, gics_level=3, as_primary=True):
        """
        :param gics_level: int, 1 to 4
        :param as_primary: if True, this "same class" condtion will be the primary "same class" condition so that all
                           other conditions specified in the chain will by default use it when possible
                           (which means no need to call .by()) unless you specified a different class in .by();
                            for current implementation, you can always set it to True
        """
        assert gics_level in (1, 2, 3, 4), 'incorrect GICS level specified, should be in (1, 2, 3, 4)'
        self._check_existing_conds('industry')
        self._check_pending_cond()
        self._clear_stack()
        self.same_class_conditions.append(('industry', {'gics_level': gics_level}))
        if as_primary:
            self.primary_class_conditions.append(('industry', {'gics_level': gics_level}))
        self._added_conds.append('industry')
        return self

    def industry_subset(self, subset: Union[int, Sequence[int]] = None):
        """
        :param subset: an integer corresponding to a GICS code (e.g. 40 for the Financial sector) or a list of
                       integers representing different GICS codes; during pair selection, only companies belonging to
                       these GICS codes will be considered
        """
        self._check_existing_conds('industry_subset')
        self._check_pending_cond()
        self._clear_stack()
        if isinstance(subset, int):
            subset = (str(subset),)
        elif subset is not None:
            assert len(set(len(str(x)) for x in subset)) == 1, 'GICS codes in "subset" should be of the same level'
            subset = tuple(sorted(str(x) for x in subset))
        assert len(subset[0]) in (2, 4, 6, 8), 'GICS code {} does not exist'.format(subset[0])
        self.subset_class_conditions.append(('industry_subset', {'subset': subset}))
        self._added_conds.append('industry_subset')
        return self

    def _check_industry_conds(self):
        if 'industry_subset' in self._added_conds:
            assert 'industry' in self._added_conds, \
                '"industry_subset" can only be used in combination with "same_industry"'
            level_map = {2: 1, 4: 2, 6: 3, 8: 4}
            industry_subset_cond = [x for x in self.subset_class_conditions if x[0] == 'industry_subset'][0]
            same_industry_cond = [x for x in self.same_class_conditions if x[0] == 'industry'][0]
            assert level_map[len(industry_subset_cond[1]['subset'][0])] <= same_industry_cond[1]['gics_level'], \
                'the industry subsets provided has a gics level lower than the "gics_level" specified for the ' \
                '"same_class" condition, which is currently prohibited '

    def correlation(self, freq='D', window=252, corr_type='raw'):
        """
        :param freq: {'D', 'M'}
        :param window: usually 12M or 252D
        :param corr_type: {"raw", "residual", "industry"}
        """
        assert corr_type in (
            'raw', 'residual', 'industry'), 'the corr_type argument should be either "raw" or "residual"'
        assert freq.upper() in ['D', 'M'], 'the freq argument should be either "D" for daily or "M" for monthly'
        self._check_existing_conds('correlation')
        self._check_pending_cond()
        self._clear_stack()
        self._stack.append(['correlation', {'freq': freq.upper(), 'window': window, 'corr_type': corr_type}])
        self._added_conds.append('correlation')
        return self

    def beta_diff(self, freq='D', window=252):  # cheap to compute all betas
        assert freq.upper() in ['D', 'M'], 'the freq argument should be either "D" for daily or "M" for monthly'
        self._check_existing_conds('beta_diff')
        self._check_pending_cond()
        self._clear_stack()
        self._stack.append(['beta_diff', {'freq': freq.upper(), 'window': window}])
        self._added_conds.append('beta_diff')
        return self

    def ADF(self, freq='D', window=252, max_lag: int = 4, auto_lag: bool = None):
        """
        :param max_lag: if auto_lag is False, max_lag is # of lagged variables used in regression; if None, use default #
                        given by rule of thumb (see https://www.statsmodels.org/dev/generated/statsmodels.tsa.stattools.adfuller.html)
        :param auto_lag: if True, ADF test will iterate over candidates lag numbers to search for the best one,
                         which is discouraged because it is very slow
        """
        assert freq.upper() in ['D', 'M'], 'the freq argument should be either "D" for daily or "M" for monthly'
        self._check_existing_conds('ADF')
        self._check_pending_cond()
        self._clear_stack()
        self._stack.append(['ADF', {'freq': freq.upper(), 'window': window, 'max_lag': max_lag, 'auto_lag': auto_lag}])
        self._added_conds.append('ADF')
        return self

    def hurst(self, freq='D', window=252):
        assert freq.upper() in ['D', 'M'], 'the freq argument should be either "D" for daily or "M" for monthly'
        self._check_existing_conds('hurst')
        self._check_pending_cond()
        self._clear_stack()
        self._stack.append(['hurst', {'freq': freq.upper(), 'window': window}])
        self._added_conds.append('hurst')
        return self

    def coint_EG(self, freq='D', window=252, max_lag=None, auto_lag: bool = None):
        """
        :param max_lag: if auto_lag is False, max_lag is # of lagged variables used in regression; if None, use default #
                        given by rule of thumb (see https://www.statsmodels.org/dev/generated/statsmodels.tsa.stattools.adfuller.html)
        :param auto_lag: if True, ADF test (the first step of EG test )will iterate over candidates lag numbers to search for
                         the best one, which is discouraged because it is very slow
        """
        assert freq.upper() in ['D', 'M'], 'the freq argument should be either "D" for daily or "M" for monthly'
        self._check_existing_conds('coint_EG')
        self._check_pending_cond()
        self._clear_stack()
        self._stack.append(
            ['coint_EG', {'freq': freq.upper(), 'window': window, 'max_lag': max_lag, 'auto_lag': auto_lag}])
        self._added_conds.append('coint_EG')
        return self

    def coint_Johansen(self, freq='D', window=252, k_ar_diff=4):
        """
        :param k_ar_diff: Number of lagged differences in the model.
                              See https://www.statsmodels.org/dev/generated/statsmodels.tsa.vector_ar.vecm.coint_johansen.html
        """
        assert freq.upper() in ['D', 'M'], 'the freq argument should be either "D" for daily or "M" for monthly'
        self._check_existing_conds('coint_Johansen')
        self._check_pending_cond()
        self._clear_stack()
        self._stack.append(['coint_Johansen', {'freq': freq.upper(), 'window': window, 'k_ar_diff': k_ar_diff}])
        self._added_conds.append('coint_Johansen')
        return self

    def add_class(self, label, as_primary=False):  # sector
        # TODO
        return self

    # ADF, correlation, industry_exposure
    def custom_distance(self, distance_name, distance_dict: Dict[str, Dict[str, pd.DataFrame]], notes=None):
        """
        This condition enables you to use your customized distance matrices. In fact, any pairwise statistic can be
        regarded as a distance measure, like correlation, beta difference, ADF statistics, etc.. You can first use this
        condition to test whether a pairwise statistic works before adding it as a stand-alone condition
        :param distance_name: the name you give to this condition
        :param distance_dict: a dict of dicts; the keys are dates, and the values are sub-dictionaries. For sub dicts, the
                              keys are classes and values are the distance DataFrame corresponding to that class on that
                              day; If the distance matrices are not grouped by classes, feel free to make all class values
                              1 (to make sure distance_dict is a dict of dicts)
        :param notes: additional notes, if any
        """
        self._check_existing_conds(distance_name)
        self._check_pending_cond()
        self._clear_stack()
        self._stack.append(
            [distance_name, {'data': self._DataRepr(distance_dict,
                                                    'distance_dict'),
                             'notes': notes,
                             'custom_type': 'distance'}])
        self._added_conds.append(distance_name)
        return self

    def add_attributes(self, label):
        # TODO
        return self

    def add_connection(self, label):  # client-customer
        # TODO
        return self

    def _absolute_value(self, direction, value, inclusive):
        self._check_stack(direction)
        assert 'pvalue' not in self._stack[0][1], \
            '"{}" cannot be used together with "pvalue"'.format(direction)
        operator = 'lt' if direction == 'less_than' else 'gt'
        operator_opposite = 'gt' if operator == 'lt' else 'lt'
        operator_inclusive = 'le' if direction == 'less_than' else 'ge'
        operator_inc_opposite = 'ge' if operator == 'le' else 'le'
        assert operator not in self._stack[0][1] and operator_inclusive not in self._stack[0][1], \
            '{} condtion already exists'.format(direction)
        value2, oppo_key = None, None
        if operator_opposite in self._stack[0][1]:
            value2 = self._stack[0][1][operator_opposite]
            oppo_key = operator_opposite
        elif operator_inc_opposite in self._stack[0][1]:
            value2 = self._stack[0][1][operator_inc_opposite]
            oppo_key = operator_inc_opposite
        if value2 is not None:
            error_msg = 'conflicting with the [{}_{}_{}] condition'.format(self._stack[0][0],
                                                                           oppo_key, self._stack[0][1][oppo_key])
            if direction == 'less_than':
                assert value > value2, error_msg
            else:
                assert value < value2, error_msg
        if inclusive:
            self._stack[0][1][operator_inclusive] = value
        else:
            self._stack[0][1][operator] = value
        return self

    def less_than(self, value, inclusive=False):
        return self._absolute_value('less_than', value, inclusive)

    def greater_than(self, value, inclusive=False):
        return self._absolute_value('greater_than', value, inclusive)

    def _quantile(self, direction, quantile):
        self._check_stack(direction)
        assert 0 < quantile < 1, 'invalid quantile value'
        opposite_direction = 'bottom' if direction == 'top' else 'top'
        assert opposite_direction not in self._stack[0][1], \
            'conflicting with the [{}_{}_{}] condition'.format(self._stack[0][0],
                                                               opposite_direction,
                                                               self._stack[0][1][opposite_direction])
        assert direction not in self._stack[0][1], \
            '{} condtion already exists'.format(direction)
        assert 'pvalue' not in self._stack[0][1], \
            '"{}" cannot be used together with "pvalue"'.format(direction)
        self._stack[0][1][direction] = quantile
        return self

    def bottom(self, quantile):
        return self._quantile('bottom', quantile)

    def top(self, quantile):
        return self._quantile('top', quantile)

    def pvalue(self, cutoff=0.05):
        self._check_stack('pvalue')
        assert 'pvalue' not in self._stack[0][1], 'pvalue condition already exists'
        assert self._stack[0][0] in self.OPERATIONS_WITH_PVALUE, '"pvalue" is not available for the "%s" condition' % \
                                                                 self._stack[0][0]
        assert all(x not in self._stack[0][1] for x in ('less_than', 'greater_than', 'top',
                                                        'bottom')), '"pvalue" cannot be used together with less_than/greater_than/top/bottom'
        assert cutoff < 0.1 + 1e-6, '"cutoff" can not be greater than 0.1'

        self._stack[0][1]['pvalue'] = cutoff
        return self

    def by(self, gics_level: int = None, class_label: str = None):
        """
        This method is used to ensure that the calculation of pairwise statistics between any two stocks only happen if
        these two stocks come from the same class (category/group)
        :param gics_level: {1, 2, 3, 4}; by default we assume you'd like to calculate pairwise statistics among different
                           GICS, hence this parameter
        :param class_label: designed to be used in combination with "add_class" condition which is not yet implemented
        """
        self._check_stack('by')
        assert 'by' not in self._stack[0][1], 'by condition already exists'
        assert 'conditional_on' not in self._stack[0][1], '"by" and "conditional_on" cannot be used simultaneously'
        if gics_level:
            assert gics_level in (1, 2, 3, 4), 'incorrect GICS level specified, should be in (1, 2, 3, 4)'
            self._stack[0][1]['by'] = ('industry', {'gics_level': gics_level})
        else:
            assert class_label, 'must specify a class/category'
            self._stack[0][1]['by'] = class_label
        return self

    # kept for reference
    # def conditional_sort(self):
    #     self._check_stack('conditional_sort')
    #     assert self._stack[0][
    #                0] in self.TIME_CONSUMING_OPERATIONS, '"conditional_sort" is only enabled for time-consuming calculations, including {}'.format(
    #         self.TIME_CONSUMING_OPERATIONS)
    #     assert 'conditional_sort' not in self._stack[0][1], '"conditional_sort" condition already exists'
    #     assert 'by' not in self._stack[0][1], '"conditional_sort" and "by" cannot be used simultaneously'
    #     self._stack[0][1]['conditional_sort'] = None
    #     return self

    # kept for reference
    # def _complete_conditional_sort_conditions(self):
    #     conditional_filters = [x[0] for x in self.conditions if 'conditional_sort' in x[1]]
    #     unconditional_filters = [x[0] for x in self.conditions if 'conditional_sort' not in x[1]]
    #     for cf in conditional_filters:
    #         for cond, attr in self.conditions:
    #             if cond == cf:
    #                 if not unconditional_filters:
    #                     del attr['conditional_sort']
    #                 else:
    #                     attr['conditional_sort'] = tuple(unconditional_filters)
    #                 unconditional_filters.append(cf)
    #                 break

    # kept for reference
    # deserted conditional_on, kept for reference
    # def conditional_on(self, all_previous=False, conditions: Union[str, Sequence[str]] = None):
    #     self._check_stack('conditional_on')
    #     assert 'conditional_on' not in self._stack[0][1], '"conditional_on" condition already exists'
    #     assert 'by' not in self._stack[0][1], '"conditional_on" and "by" cannot be used simultaneously'
    #     assert all_previous is True or conditions is not None, \
    #         'must specify the "conditions" parameter if "all_previous" is False'
    #     if all_previous:
    #         conds = [x for x in self._added_conds if
    #                  x not in [self._stack[0][0]] + [x[0] for x in self._same_class_conditions]]
    #     else:
    #         conds = []
    #         if isinstance(conditions, str):
    #             conds.append(conditions)
    #         else:
    #             conds.extend([x for x in conditions])
    #         for c in conds:
    #             assert c in self._added_conds, '"{}" is either an invalid (misspelled) condition name, or is referring to a condtion not added yet. Available options are {}'.format(
    #                 c, tuple(self._added_conds))
    #             assert c != self._stack[0][0], '"{}" cannot be conditioned on itself'.format(self._stack[0][0])
    #             assert c not in [x[0] for x in self._same_class_conditions], \
    #                 'Class condition "{}" is not supposed to be conditioned on, use "by" to specify the class instead'.format(c)
    #     if conds:
    #         self._stack[0][1]['conditional_on'] = tuple(conds)
    #     return self

    def start(self):
        """
        Need to call this explicitly before starting the composition of a chained condition series.
        All contents previously saved in internal lists (_stack, same_class_conditions, etc.) will be discarded and
        the "_ready_for_filter" attribute of the PairFilter instance passed in initialization will be set to False so that
        it cannot start a pair selection process
        """
        for x in [self._stack, self.same_class_conditions, self._added_conds, self.conditions]:
            if x:
                del x[:]
        self.PF._ready_for_filter = False
        return self

    def end(self):
        """
        Need to call this explicitly to finish the composition of a chained condition series.
        the "_ready_for_filter" attribute of the PairFilter instance passed in initialization will be set to True so that
        the pair filter process is ready to start
        """
        if self._stack:
            self._check_pending_cond()
            self._clear_stack()
        self._check_industry_conds()
        # self._complete_conditional_sort_conditions()
        self.PF._ready_for_filter = True


# %%
def restore_return_raw(f):
    """
    A decorator. Set the "return_raw" attribute of the DataLoader object to False before executing the decorated
    function. Reset "return_raw" to True afterwards. Used when we need to make sure that during the execution of the
    decorated function, datasets returned by the DataLoader object are DataFrames (so that we can have access to their
    indices and columns)
    """

    @wraps(f)
    def wrapper(self, *args, **kwargs):
        restore_return_raw = True if self.dl.return_raw else False
        if restore_return_raw:
            self.dl.return_raw = False
        result = f(self, *args, **kwargs)
        if restore_return_raw:
            self.dl.return_raw = True
        return result

    return wrapper


def restore_return_DF(f):
    """
    A decorator. Set the "return_raw" attribute of the DataLoader object to True before executing the decorated
    function. Reset "return_raw" to False afterwards. Used when we need to make sure that during the execution of the
    decorated function, datasets returned by the DataLoader object are numpy arrays (for speed consideration).
    """

    @wraps(f)
    def wrapper(self, *args, **kwargs):
        restore_return_DF = True if not self.dl.return_raw else False
        if restore_return_DF:
            self.dl.return_raw = True
        result = f(self, *args, **kwargs)
        if restore_return_DF:
            self.dl.return_raw = False
        return result

    return wrapper


# %%
def format_dict_to_string(dictionary: Dict) -> str:
    """
    Convert a Python dict into a string so that it can serve as a dict key
    """
    new_dict = {k: v for k, v in sorted(dictionary.items(), key=lambda x: x[0])}
    string = re.subn(r"[{'}:]", '', str(new_dict))[0]
    string = string.replace(' ', '#')
    string = string.replace(',', '|')
    return string


# %%
def extract_pairwise_value_diff(values: pd.DataFrame, class_df: pd.DataFrame = None, class_subset: Dict = None,
                                colname: str = None, **kwargs) -> Dict[str, pd.DataFrame]:
    """
    Used to extract the stock pairs if their differences in a variable/statistic satisfy certain filtering conditions
    :param values: raw values DataFrame
    :param class_df: class DataFrame
    :param class_subset: a dict-like object used to constrain the calculation to specific subclasses; the keys are dates,
                         and the values are list-like objects containing the subset of classes that are of interest to
                         this calculation; if None, all stocks belong to class 1 -- an artificial class
    :param colname: if given, will become the column name in the returned DataFrames corresponding to the difference
                    variable; otherwise that column is named "diff" by default
    :param kwargs: parameters to be passed into gen_comparison_func
    :return: a dict;the keys are dates and the values are DataFrames of pairs that pass the filtering condition;
             column "A" and column "B" are the stock ids
    """
    comp_filter = gen_comparison_func(**kwargs)
    dates = values.columns
    has_class = True
    ids = values.index.values
    if class_df is None:
        class_df = pd.DataFrame(np.ones_like(values), columns=dates)
        has_class = False
    res = {}
    for date in dates:
        value_date = values.loc[:, date].values
        class_date = class_df.loc[:, date].values
        diff_class = np.unique(class_date[~np.isnan(class_date)])
        if class_subset is not None:
            diff_class = [c for c in class_subset[date]]
        data_dict = defaultdict(list)
        for c in diff_class:
            c_mask = (class_date == c) & (~np.isnan(value_date))
            c_value = value_date[c_mask]
            if not len(c_value):
                continue
            low_tri_indices = np.tril_indices(len(c_value), k=-1)
            value_diff = np.abs((c_value - c_value.reshape(-1, 1))[low_tri_indices])
            if not len(value_diff):
                continue
            filter_mask = comp_filter(value_diff)
            value_diff = value_diff[filter_mask]
            if not len(value_diff):
                continue
            data_dict['A'].append(ids[c_mask][low_tri_indices[0]][filter_mask])
            data_dict['B'].append(ids[c_mask][low_tri_indices[1]][filter_mask])
            if colname is not None:
                data_dict[colname].append(value_diff)
            else:
                data_dict['diff'].append(value_diff)
            if has_class:
                data_dict['class'].append(np.full_like(value_diff, c))
        if data_dict['A']:
            pair_sorted = np.sort(np.column_stack([np.concatenate(data_dict['A']), np.concatenate(data_dict['B'])]),
                                  axis=1)
            pairs_df = pd.DataFrame(pair_sorted, columns=['A', 'B'])
            data_dict2 = {k: np.concatenate(v) for k, v in data_dict.items() if k not in ('A', 'B')}
            pairs_df = pd.concat([pairs_df, pd.DataFrame(data_dict2)], axis=1)
            res[date] = pairs_df
        else:
            res[date] = None
    return res


# %%
def extract_pairwise_similarity(similarities: Dict, key: str = None, record_all_keys=True,
                                colname_prefix: str = None, **kwargs) -> Dict[str, pd.DataFrame]:
    """
    Used to extract the stock pairs if their similarity-alike statistics (e.g. correlation, ADF stats, cointegration stats)
    satisfy certain filtering conditions
    :param similarities: depending on the statistics, can be either 1) a dict of dicts: the first layer is dates - dicts;
                         the second layer is classes - DataFrames; 2) a dict of dicts of dicts: the first layer is dates - dicts;
                         the second layer is classes - dicts; the third layer is stats keys - DataFrames. The second case
                         can happen, for example, if both the statistics and the corresponding p-value are saved
    :param key: if not None and "similarities" satisfies case 2), the key is used to extract DataFrames from the
                third layer and this particular type of similarities DataFrame is used in the filtering condition;
                currently available options: {'statistics', 'pvalue'}
    :param record_all_keys: if True and "similarities" satisfies case 2), all available stats will be appended (although
                            only the one specified by "key" participate in pair selection); for example, if "key" is
                            "statistics", then "statistics" will participate in pair selection but "pvalue" will also
                            appear in the returned pairs DataFrames
    :param colname_prefix: the prefix before each stats key that appears in the returned DataFrames
    :param kwargs: parameters to be passed into gen_comparison_func
    :return: a dict;the keys are dates and the values are DataFrames of pairs that pass the filtering condition;
             column "A" and column "B" are the stock ids
    """
    comp_filter = gen_comparison_func(**kwargs)
    res = {}
    has_class = True

    for date, sim_date in similarities.items():
        data_dict = defaultdict(list)
        for c, corr_matrix in sim_date.items():
            if c == 1. and has_class and len(sim_date) == 1:
                has_class = False
            if key is not None and isinstance(corr_matrix, dict):
                corr_mat = corr_matrix[key]
            else:
                corr_mat = corr_matrix
            ids = corr_mat.index.values
            c_sim = corr_mat.values
            low_tri_indices = np.tril_indices(len(c_sim), k=-1)
            c_sim = c_sim[low_tri_indices]
            if not len(c_sim):
                continue
            filter_mask = comp_filter(c_sim)
            c_sim = c_sim[filter_mask]
            if not len(c_sim):
                continue
            data_dict['A'].append(ids[low_tri_indices[0]][filter_mask])
            data_dict['B'].append(ids[low_tri_indices[1]][filter_mask])
            if key is not None:
                if colname_prefix is not None:
                    data_dict[colname_prefix + '_' + key].append(c_sim)
                else:
                    data_dict[key].append(c_sim)
            else:
                if colname_prefix is not None:
                    data_dict[colname_prefix].append(c_sim)
                else:
                    data_dict['score'].append(c_sim)
            if key is not None and isinstance(corr_matrix, dict) and record_all_keys:
                other_keys = sorted([k for k in corr_matrix.keys() if k != key])
                for k in other_keys:
                    if colname_prefix is not None:
                        data_dict[colname_prefix + '_' + k].append(corr_matrix[k].values[low_tri_indices][filter_mask])
                    else:
                        data_dict[k].append(corr_matrix[k].values[low_tri_indices][filter_mask])
            if has_class:
                data_dict['class'].append(np.full_like(c_sim, c))
        if data_dict['A']:
            pair_sorted = np.sort(np.column_stack([np.concatenate(data_dict['A']), np.concatenate(data_dict['B'])]),
                                  axis=1)
            pairs_df = pd.DataFrame(pair_sorted, columns=['A', 'B'])
            data_dict2 = {k: np.concatenate(v) for k, v in data_dict.items() if k not in ('A', 'B')}
            pairs_df = pd.concat([pairs_df, pd.DataFrame(data_dict2)], axis=1)
            res[date] = pairs_df
        else:
            res[date] = None
    return res


# %%
def gen_comparison_func(**kwargs):
    func_list = []
    if 'le' in kwargs:
        func_list.append(lambda x: x <= kwargs['le'])
    if 'ge' in kwargs:
        func_list.append(lambda x: x >= kwargs['ge'])
    if 'lt' in kwargs:
        func_list.append(lambda x: x < kwargs['lt'])
    if 'gt' in kwargs:
        func_list.append(lambda x: x > kwargs['gt'])
    if 'top' in kwargs:
        func_list.append(lambda x: x >= np.quantile(x, 1 - kwargs['top']))
    elif 'bottom' in kwargs:
        func_list.append(lambda x: x <= np.quantile(x, kwargs['bottom']))
    if 'pvalue' in kwargs:
        func_list.append(lambda x: x <= kwargs['pvalue'])
    if func_list:
        def func(x):
            return reduce(and_, [f(x) for f in func_list])
    else:
        def func(x):
            return slice(None)
    return func


# %%
class PairFilter:

    def __init__(self, DataLoaderObj: DataLoader, n_jobs: int = None):
        """
        :param DataLoaderObj: a DataLoader object
        :param n_jobs: number of cpu processes; -1 means using all; 1 means single process (no parallel computation)
        """
        self.dl = DataLoaderObj
        self.filter_conditions = None
        # interim_result stores the computation results of beta, correlation, ADF, etc.
        self.interim_result = defaultdict(dict)
        # cache stores the pairs DataFrames corresponding to the ones that pass beta_diff condition,
        # or correlation condition, or ADF condition, etc.
        self.cache = defaultdict(dict)
        # pairs_cache stores the pairs DataFrames corresponding to the ones that pass all conditions (i.e. selected pairs)
        self.pairs_cache = defaultdict(dict)
        self._ready_for_filter = False
        self._primary_class = []
        self._subset_class = []

        from multiprocessing import cpu_count
        if n_jobs is None:
            self.n_jobs = cpu_count() // 2
        else:
            assert n_jobs is int and (n_jobs == -1 or n_jobs >= 1), 'invalid "n_jobs" value: {}'.format(n_jobs)
            if n_jobs == -1:
                self.n_jobs = cpu_count()
            else:
                self.n_jobs = min(n_jobs, cpu_count())
        print('Using %d cpu processes...' % self.n_jobs)

    def new_filter(self) -> FilterConditions:
        """
        Start creating a new chained series of conditions
        :return: return a FilterConditions object. The previous FilterConditions Object (if any) will be overridden.
        """
        self.filter_conditions = FilterConditions(self)
        self._ready_for_filter = False
        del self._primary_class[:]
        del self._subset_class[:]
        return self.filter_conditions

    def show_conditions(self):
        print(self.filter_conditions)

    def _map_dates_to_nearest_trading_day(self, dates: Union[Sequence[str], np.ndarray, pd.Index], freq='D',
                                          verbose=True):
        reference_dates = frequency_convert(self.dl.dates, freq)
        if min(dates) < reference_dates[0] and verbose:
            print('[Warning] {} is earlier than {}, the first day when data is available at frequency {}'.format(
                min(dates), reference_dates[0], freq))
        if max(dates) > reference_dates[-1] and verbose:
            print('[Warning] {} is later than {}, the last day when data is available at frequency {}'.format(
                max(dates), reference_dates[-1], freq))
        res = []
        for x in dates:
            where = np.argwhere(reference_dates <= x)
            if len(where):
                res.append(reference_dates[where[-1, 0]])
            else:
                res.append(None)
        return res

    @restore_return_DF
    def _generate_industry_flag(self, subset):
        """
        generating a flag array (its length is equal to the total number of stock IDs) indicating whether a stock has been
        in any classes that belong to the specified subset of classes
        """
        gics_dataset_map = {2: 'QES_GSECTOR', 4: 'GGROUP', 6: 'GIND', 8: 'GSUBIND'}
        dataset = self.dl[gics_dataset_map[len(subset[0])]]
        industry_flag = reduce(or_, [np.sum(dataset == int(ind_code), axis=1) > 0 for ind_code in subset])
        return industry_flag

    @restore_return_DF
    def _generate_industry_subset(self, industry_subset_cond):
        """
        If the "subset" parameter you specify in FilterConditions.ndustry_subset() is 40 and you also require that pairs
        should come from the same GICS level 3 industry (which will by default constrain the computation of any pairwise
        statistics within GICS level 3), then this method gives you a list of GICS-level3 under 40 (i.e. 401010, 401020,
        etc.) for each date
        :param industry_subset_cond: the industry subset condition from the FilterConditions object
        """
        key = format_dict_to_string(industry_subset_cond)
        if key not in self.cache['subset_class']:
            gics_dataset_map = {1: 'QES_GSECTOR', 2: 'GGROUP', 3: 'GIND', 4: 'GSUBIND'}
            industry_mat = self.dl[gics_dataset_map[industry_subset_cond['gics_level']]].copy()
            subset = industry_subset_cond['subset']
            gics_dataset_map = {2: 'QES_GSECTOR', 4: 'GGROUP', 6: 'GIND', 8: 'GSUBIND'}
            dataset = self.dl[gics_dataset_map[len(subset[0])]]
            mark = reduce(or_, [dataset == int(ind_code) for ind_code in subset])
            industry_mat[~mark] = np.nan
            industry_subset_dict = {}
            for i, date in enumerate(self.dl.dates):
                inds = industry_mat[:, i]
                inds = inds[~np.isnan(inds)]
                industry_subset_dict[date] = np.unique(inds)
            self.cache['subset_class'][key] = industry_subset_dict

    def _complete_condition_dict(self, cond_dict, add_subset_class=False, add_primary_class=False):
        """
        This method exists for two reasons: 1) if you want to constrain the computation of any statistics within the same
        class (say, GICS level 3), you are allowed to not call .by() after each condition if you call
        same_industry(as_primary=True) somewhere. If this is the case then the condition string generated by
        format_dict_to_string() will not be complete. Then this method helps by adding the primary class to each
        condition. 2) it also helps by adding the subset class to each condition so that the condition string generated
        will reflect all the information considered in pair selection
        """
        assert not (add_primary_class and add_subset_class)
        cond_dict = cond_dict.copy()
        if add_primary_class:
            if self._primary_class:
                if self._primary_class[0][0] == 'industry':  # TODO
                    cond_dict['by'] = self._primary_class[0]
                    if self._subset_class:
                        if self._subset_class[0][0] == 'industry_subset':  # TODO
                            cond_dict['by'][1]['subset'] = self._subset_class[0][1]['subset']
        if add_subset_class:
            if self._subset_class:
                if self._subset_class[0][0] == 'industry_subset':  # TODO
                    cond_dict['subset'] = self._subset_class[0][1]['subset']
        return cond_dict

    @restore_return_raw
    def _read_class_conds(self):
        """
        Read and set all class conditions (including subset conditions) specified in FilterConditions object.
        """

        for cond in self.filter_conditions.subset_class_conditions:
            if cond[0] == 'industry_subset':
                self._subset_class.append(cond)
            else:
                # TODO
                ...

        def set_industry_class_matrix(cond_dict):
            gics_dataset_map = {1: 'QES_GSECTOR', 2: 'GGROUP', 3: 'GIND', 4: 'GSUBIND'}
            cond_dict = self._complete_condition_dict(cond_dict, add_subset_class=True)
            key = format_dict_to_string(cond_dict)
            if key not in self.cache['class']:
                dataset = gics_dataset_map[cond_dict['gics_level']]
                ind_df = self.dl[dataset].copy()
                if self._subset_class:
                    industry_mark = self._generate_industry_flag(cond_dict['subset'])
                    ind_df.loc[~industry_mark, :] = np.nan
                self.cache['class'][key] = ind_df

        def set_industry_subset(cond_dict):
            if self._subset_class:
                cond_dict = self._complete_condition_dict(cond_dict, add_subset_class=True)
                self._generate_industry_subset(cond_dict)

        for cond in self.filter_conditions.primary_class_conditions:
            if cond[0] == 'industry':
                set_industry_class_matrix(cond[1])
                set_industry_subset(cond[1])
                self._primary_class.append(cond)
            else:
                # TODO
                ...
        for cond in self.filter_conditions.same_class_conditions:
            if cond[0] == 'industry':
                set_industry_class_matrix(cond[1])
                set_industry_subset(cond[1])
            else:
                # TODO
                ...
        for cond, attr in self.filter_conditions.conditions:
            if 'by' in attr:
                if attr['by'][0] == 'industry':
                    set_industry_class_matrix(attr['by'][1])
                    set_industry_subset(attr['by'][1])
                else:
                    # TODO
                    ...

    def _load_class_matrix(self, attr):
        """
        Load the corresponding class DataFrame based on the condition attribute dict you pass
        You can notice that all condition expressions are stored in the form of a tuple of two elements.
        The first element is the name of that condition and the second is the attribute of that condition, called "attr"
        """

        if 'by' in attr:
            if attr['by'][0] == 'industry':
                cond_dict = self._complete_condition_dict(attr['by'][1], add_subset_class=True)
                key = format_dict_to_string(cond_dict)
                return self.cache['class'][key]
            else:
                # TODO
                ...
        elif self._primary_class:
            for cond in self._primary_class:
                if cond[0] == 'industry':
                    cond_dict = self._complete_condition_dict(cond[1], add_subset_class=True)
                    key = format_dict_to_string(cond_dict)
                    return self.cache['class'][key]
                else:
                    # TODO
                    ...
        return None

    def _load_class_subset(self, attr):
        """
        Load the corresponding class subset based on the condition attribute dict you pass
        """
        if not self._subset_class:
            return None

        if 'by' in attr:
            if attr['by'][0] == 'industry':
                cond_dict = self._complete_condition_dict(attr['by'][1], add_subset_class=True)
                key = format_dict_to_string(cond_dict)
                return self.cache['subset_class'][key]
            else:
                # TODO
                ...
        elif self._primary_class:
            for cond in self._primary_class:
                if cond[0] == 'industry':
                    cond_dict = self._complete_condition_dict(cond[1], add_subset_class=True)
                    key = format_dict_to_string(cond_dict)
                    return self.cache['subset_class'][key]
                else:
                    # TODO
                    ...
        return None

    def _read_function_conds(self, dates):
        """
        read the filtering conditions from FilterConditions object and call corresponding filtering functions
        """
        for cond, attr in self.filter_conditions.conditions:
            # if 'conditional_on' in attr:
            #     if cond in FilterConditions.TIME_CONSUMING_OPERATIONS:  # new
            #         continue
            if cond == 'beta_diff':
                print('applying "difference of beta" condition')
                self._beta_diff(attr, dates)
            elif cond == 'correlation':
                print('applying "correlation" condition')
                self._correlation(attr, dates)
            elif cond == 'ADF':
                print('applying "Augmented DF test" condition')
                self._ADF(attr, dates)
            elif cond == 'coint_EG':
                print('applying "Engle-Granger cointegration test" condition')
                self._coint_EG(attr, dates)
            elif cond == 'coint_Johansen':
                print('applying "Johansen cointegration test" condition')
                self._coint_Johansen(attr, dates)
            elif cond == 'hurst':
                print('applying "Hurst Exponent" condition')
                self._hurst(attr, dates)
            elif 'custom_type' in attr:
                custom_type = attr['custom_type']
                if custom_type == 'distance':
                    print('applying "{}" condition'.format(cond))
                    self._distance_metric(cond, attr, dates)
                else:
                    # TODO
                    ...
            else:
                # TODO
                ...

    def _condition_attr_dict_to_key_n_by(self, attr_dict, add_subset: bool = True):
        """
        Transform the condition attribute dict to a string (so that it can be used as a key in self.cache or
        self.interim_result). Before generating the key, the primary class (if any) will be appended. The subset class
        (if any) will also be appended if add_subset is True.
        If (after appending primary class & subset class) there is a "by" key in the attribute dict, the corresponding value
        will also be returned (otherwise return None)
        """
        if 'by' not in attr_dict:
            if 'custom_type' not in attr_dict:  # TODO
                attr_dict = self._complete_condition_dict(attr_dict, add_primary_class=True)
                if 'by' in attr_dict and add_subset:  # TODO
                    attr_dict['by'][1].update(self._complete_condition_dict(attr_dict['by'][1], add_subset_class=True))
            key = format_dict_to_string(attr_dict)
        else:
            if attr_dict['by'][0] == 'industry':
                attr_dict = attr_dict.copy()
                attr_dict['by'][1].update(self._complete_condition_dict(attr_dict['by'][1], add_subset_class=True))
            else:  # TODO
                ...
            key = format_dict_to_string(attr_dict)
        by = None if 'by' not in attr_dict else attr_dict['by']
        return key, by

    @restore_return_raw
    def _beta_diff(self, attr, dates):
        """
        beta difference filter
        """
        key, _ = self._condition_attr_dict_to_key_n_by(attr)
        available_dates = []
        if key in self.cache['beta_diff']:
            available_dates.extend(list(self.cache['beta_diff'][key].keys()))
        freq, window = attr['freq'], attr['window']
        if freq != 'D':
            target_dates = self._map_dates_to_nearest_trading_day(dates, freq, verbose=False)
        else:
            target_dates = dates
        target_dates = sorted(set([x for x in target_dates if x is not None and x not in available_dates]))
        mkt = self.dl.get_processed_data('MKT_RTN_EQUI')
        if target_dates:
            betas = self._load_or_calculate_beta(target_dates=target_dates, freq=freq, window=window, mkt=mkt,
                                                 sort_dates=False)
            class_matrix = self._load_class_matrix(attr)
            class_subset = self._load_class_subset(attr)
            # return betas, class_matrix
            beta_diffs = extract_pairwise_value_diff(betas, class_matrix, class_subset, colname='beta_diff', **attr)
            if key in self.cache['beta_diff']:
                self.cache['beta_diff'][key].update(beta_diffs)
            else:
                self.cache['beta_diff'][key] = beta_diffs

    @restore_return_raw
    def _find_dates_for_beta_calculation(self, target_dates, freq, window):
        reference_dates = frequency_convert(self.dl.dates, freq)
        reference_dates_idx = np.array([np.argwhere(self.dl.dates == x)[0, 0] for x in reference_dates])
        target_dates_idx = np.array([np.argwhere(reference_dates == x)[0, 0] for x in target_dates])
        min_idx = window + np.argwhere(
            (reference_dates_idx + 1) > np.argwhere(np.sum(np.isfinite(self.dl['RTN'].values), axis=0) > 1)[0, 0])[0, 0]
        idxs = []
        for x in target_dates_idx:
            if (x - window + 1) >= min_idx:
                idxs.extend(list(range(reference_dates_idx[x - window] + 1, reference_dates_idx[x] + 1)))
        dates_for_beta_calculation_idx = sorted(set(idxs))
        return self.dl.dates[dates_for_beta_calculation_idx]

    def _load_or_calculate_beta(self, target_dates, freq, window, indep_vars='mkt_ew', mkt=None,
                                sort_dates=False) -> pd.DataFrame:
        """
        load beta from self.interim_result or calculate beta
        :param indep_vars: {'mkt_ew', 'ind_ew'}
        :param sort_dates: if True, the returned beta DataFrame will have the column (dates) sorted
        """
        use_old_data = False
        if freq == 'M':
            freq = 'D'
            window = 21 * window
        key_i = format_dict_to_string({'freq': freq, 'window': window, 'indep_vars': indep_vars})
        if key_i in self.interim_result['beta']:
            common_dates = self.interim_result['beta'][key_i].columns.intersection(target_dates)
            if len(common_dates):
                use_old_data = True
                betas_in_memory = self.interim_result['beta'][key_i].loc[:, common_dates]
                target_dates = [x for x in target_dates if x not in common_dates]
        if not len(target_dates):
            betas = betas_in_memory
        else:
            betas = calculate_beta(self.dl['RTN'], self.dl['IN_US_1'], mkt=mkt, output_freq=freq,
                                   window_size=window, target_dates=target_dates)
            if key_i not in self.interim_result['beta']:
                self.interim_result['beta'][key_i] = betas
            else:
                self.interim_result['beta'][key_i] = self.interim_result['beta'][key_i].merge(betas, left_index=True,
                                                                                              right_index=True)
                if use_old_data:
                    betas = betas_in_memory.merge(betas, left_index=True, right_index=True)
        if sort_dates:
            betas = betas.reindex(columns=sorted(betas.columns.values))
        return betas

    @restore_return_raw
    def _correlation(self, attr, dates):
        """
        correlation filter
        Note that a filter method (self._correlation(), self._adf_like(), self._distance_metric(), etc. ) usually does the following things:
        1. check for self.cache. Identify the dates when pairs extraction is needed. If none of the dates requires new
        pairs extraction, simply return the results stored in self.cache
        2. check for self.interim_result. Identify the dates when computation of statistics is needed. If none of the dates
        requires new computation, simply use the results stored in self.interim_result to extract pairs according to
        the condition attribute dict (attr)
        3. start new computation of statistics, which is the most time consuming
        4. update self.interim_result and self.cache for future use. the computation of statistics corresponding to attr
        is stored in self.interim_result, and the extraction of paris corresponding to attr is stored in self.cache which
        will further be merged with other extraction results to get the final pairs that pass a series of conditions
        5. you'd better follow the paradigms if you'd like to add new functions
        """
        key, by = self._condition_attr_dict_to_key_n_by(attr)
        available_dates = []
        if key in self.cache['correlation']:
            available_dates.extend(list(self.cache['correlation'][key].keys()))
        freq, window, corr_type = attr['freq'], attr['window'], attr['corr_type']
        if freq != 'D':
            target_dates = self._map_dates_to_nearest_trading_day(dates, freq, verbose=False)
        else:
            target_dates = dates
        target_dates = sorted(set([x for x in target_dates if x is not None and x not in available_dates]))
        if corr_type in ('residual', 'industry') and len(target_dates):
            dates_for_beta_cal = self._find_dates_for_beta_calculation(target_dates, freq, window)
            mkt = self.dl.get_processed_data('MKT_RTN_EQUI') if corr_type == 'residual' else self.dl.get_processed_data(
                'IND_RTN_EQUI')
            indep_vars = 'mkt_ew' if corr_type == 'residual' else 'ind_ew'
            betas = self._load_or_calculate_beta(target_dates=dates_for_beta_cal, freq=freq, window=window,
                                                 indep_vars=indep_vars, mkt=mkt, sort_dates=True)
        if target_dates:
            use_old_data = False
            if by is not None:
                key_i = format_dict_to_string({'freq': freq, 'window': window,
                                               'corr_type': corr_type, 'by': by})
            else:
                key_i = format_dict_to_string({'freq': freq, 'window': window, 'corr_type': corr_type})
            if key_i in self.interim_result['correlation']:
                common_dates = list(set(self.interim_result['correlation'][key_i].keys()).intersection(target_dates))
                if len(common_dates):
                    use_old_data = True
                    correlation_in_memory = {d: self.interim_result['correlation'][key_i][d] for d in common_dates}
                    target_dates = [x for x in target_dates if x not in common_dates]
            if not target_dates:
                correlations = correlation_in_memory
            else:
                class_matrix = self._load_class_matrix(attr)
                class_subset = self._load_class_subset(attr)
                if corr_type in ('residual', 'industry'):
                    returns = self.dl['RTN'].copy()
                    returns.loc[:, dates_for_beta_cal] = returns.loc[:, dates_for_beta_cal].values - np.multiply(
                        betas.loc[:, dates_for_beta_cal].values, mkt.loc[:, dates_for_beta_cal].values)
                    correlations = calculate_corr(returns, self.dl['IN_US_1'], class_df=class_matrix,
                                                  class_subset=class_subset, output_freq=freq,
                                                  window_size=window, target_dates=target_dates)
                else:
                    correlations = calculate_corr(self.dl['RTN'], self.dl['IN_US_1'], class_df=class_matrix,
                                                  class_subset=class_subset,
                                                  output_freq=freq, window_size=window, target_dates=target_dates)
                if key_i not in self.interim_result['correlation']:
                    self.interim_result['correlation'][key_i] = correlations
                else:
                    self.interim_result['correlation'][key_i].update(correlations)
                if use_old_data:
                    correlations.update(correlation_in_memory)
            corrs = extract_pairwise_similarity(correlations, colname_prefix='correlation', **attr)
            if key in self.cache['correlation']:
                self.cache['correlation'][key].update(corrs)
            else:
                self.cache['correlation'][key] = corrs

    @restore_return_raw
    def _adf_like(self, cond_name, stat_function, attr, dates):
        """
        a wrapper filter for all conditions similar to ADF (e.g. ADF, cointegration, hurst, etc.)
        :param cond_name: condition name; namely the first element of a condition tuple from FilterConditions
        :param stat_function: a function that can take care of the computation of desired statistics; corresponding to the
                              calculate_xxx family in utils/computation_funcs.py
        """
        key, by = self._condition_attr_dict_to_key_n_by(attr)
        available_dates = []
        if key in self.cache[cond_name]:
            available_dates.extend(list(self.cache[cond_name][key].keys()))
        # freq, window, max_lag, auto_lag = attr['freq'], attr['window'], attr['max_lag'], attr['auto_lag']
        freq, window = attr['freq'], attr['window']
        if freq != 'D':
            target_dates = self._map_dates_to_nearest_trading_day(dates, freq, verbose=False)
        else:
            target_dates = dates
        target_dates = sorted(set([x for x in target_dates if x is not None and x not in available_dates]))
        if target_dates:
            use_old_data = False
            stat_in_memory = {}
            key_i_dict = {'freq': freq, 'window': window}
            kwargs = {k: attr[k] for k in ['max_lag', 'auto_lag', 'k_ar_diff'] if k in attr}
            key_i_dict.update(kwargs)
            if by is not None:
                key_i_dict['by'] = by
            key_i = format_dict_to_string(key_i_dict)
            if key_i in self.interim_result[cond_name]:
                common_dates = list(set(self.interim_result[cond_name][key_i].keys()).intersection(target_dates))
                if len(common_dates):
                    use_old_data = True
                    stat_in_memory.update({d: self.interim_result[cond_name][key_i][d] for d in common_dates})
                    target_dates = [x for x in target_dates if x not in common_dates]
            if not target_dates:
                stat = stat_in_memory
            else:
                class_matrix = self._load_class_matrix(attr)
                class_subset = self._load_class_subset(attr)
                stat = stat_function(self.dl['PRCCD'], self.dl['IN_US_1'], class_df=class_matrix,
                                     class_subset=class_subset, output_freq=freq, window_size=window,
                                     wealth_df=self.dl['CUM_WEALTH'], target_dates=target_dates, log_scale=True,
                                     n_jobs=self.n_jobs, **kwargs)
                if key_i not in self.interim_result[cond_name]:
                    self.interim_result[cond_name][key_i] = stat
                else:
                    self.interim_result[cond_name][key_i].update(stat)
                if use_old_data:
                    stat.update(stat_in_memory)
            if 'pvalue' in attr:
                stat_dfs = extract_pairwise_similarity(stat, key='pvalue', colname_prefix=cond_name, **attr)
            else:
                stat_dfs = extract_pairwise_similarity(stat, key='statistics', colname_prefix=cond_name, **attr)
            if key in self.cache[cond_name]:
                self.cache[cond_name][key].update(stat_dfs)
            else:
                self.cache[cond_name][key] = stat_dfs

    def _ADF(self, attr, dates):
        """
        ADF filter
        """
        self._adf_like('ADF', calculate_ADF, attr, dates)

    def _coint_EG(self, attr, dates):
        """
        Engel-Granger cointegration filter
        """
        self._adf_like('coint_EG', calculate_coint_EG, attr, dates)

    def _coint_Johansen(self, attr, dates):
        """
        Johansen cointegration filter
        """
        self._adf_like('coint_Johansen', calculate_coint_Johansen, attr, dates)

    def _hurst(self, attr, dates):
        """
        Hurst filter
        """
        self._adf_like('hurst', calculate_Hurst_exponent, attr, dates)

    @restore_return_raw
    def _distance_metric(self, distance_name, attr, dates, **kwargs):
        """
        custom distance filter
        """
        distance_dict = attr['data']()
        # key = format_dict_to_string()
        key, _ = self._condition_attr_dict_to_key_n_by(attr)
        available_dates = []
        if key in self.cache[distance_name]:
            available_dates.extend(list(self.cache[distance_name][key].keys()))
        dates_existing = np.array(sorted(distance_dict.keys()))
        target_dates = []
        for d in dates:
            mark = dates_existing <= d
            if mark.any():
                target_dates.append(dates_existing[mark][-1])  # TODO
        target_dates = sorted(set([x for x in target_dates if x is not None and x not in available_dates]))
        if target_dates:
            use_old_data = False
            distance_in_memory = {}
            key_i = format_dict_to_string({'notes': attr['notes']})
            if key_i in self.interim_result[distance_name]:
                common_dates = list(set(self.interim_result[distance_name][key_i].keys()).intersection(target_dates))
                if len(common_dates):
                    use_old_data = True
                    distance_in_memory.update({d: self.interim_result[distance_name][key_i][d] for d in common_dates})
                    target_dates = [x for x in target_dates if x not in common_dates]
            if not target_dates:
                distance = distance_in_memory
            else:
                distance = restructure_distance_mat(distance_dict, target_dates=target_dates,
                                                    in_flag=self.dl['IN_US_1'], universe=True)
                if key_i not in self.interim_result[distance_name]:
                    self.interim_result[distance_name][key_i] = distance
                else:
                    self.interim_result[distance_name][key_i].update(distance)
                if use_old_data:
                    distance.update(distance_in_memory)
            distance_dfs = extract_pairwise_similarity(distance, colname_prefix=distance_name, **attr)
            if key in self.cache[distance_name]:
                self.cache[distance_name][key].update(distance_dfs)
            else:
                self.cache[distance_name][key] = distance_dfs

    def filter_given_dates(self, dates: Union[str, Sequence[str], np.ndarray, pd.Index] = None,
                           start_from: str = None, end_at: str = None) -> dict:
        """
        :param dates: dates that you would like to get pairs selection results
        :param start_from: if dates is None, then you can specify the start date (via start_from) and end date (via end_at)
        :param end_at: see above
        :return: a dict with dates as keys and pairs DataFrames as values
        """
        if not self._ready_for_filter:
            raise Exception(
                'Pair filtering not ready: need adding filter conditions beforehand. Try calling the new_filter() method')
        if dates is None and start_from is None and end_at is None:
            dates = self.dl.dates
        elif isinstance(dates, str):
            dates = [dates, ]
        elif dates is not None:
            pass
        elif start_from is not None and end_at is not None:
            dates = pd.date_range(start_from, end_at).strftime('%Y-%m-%d')
        else:
            raise Exception('the dates/start_from/end_at arguments are wrongly specified')
        # insert here
        use_old_data = False
        dates_not_cached = dates
        if hash(self.filter_conditions) in self.pairs_cache.keys():
            completed_dates = list(self.pairs_cache[hash(self.filter_conditions)].keys())
            common_dates = set(completed_dates).intersection(set(dates))
            if len(common_dates):
                use_old_data = True
                dates_not_cached = sorted(set(dates).difference(common_dates))
        if len(dates_not_cached):
            dates_ = self._map_dates_to_nearest_trading_day(dates_not_cached)
            dates_unique = sorted([x for x in set(dates_) if x is not None])
            self._read_class_conds()
            self._read_function_conds(dates_unique)

        res = {}
        merge_func = lambda df_left, df_right: pd.merge(df_left,
                                                        df_right[[c for c in df_right.columns if c != 'class']],
                                                        how='inner', on=['A', 'B'])
        primary_class_mat = self._load_class_matrix({})
        for date in dates:
            if use_old_data and date not in dates_not_cached:
                # self.pairs_cache[hash(self.filter_conditions)]
                res[date] = self.pairs_cache[hash(self.filter_conditions)][date]
                continue
            pair_df_list = []  # change
            skip = False
            for cond, attr in self.filter_conditions.conditions:  # TODO
                cond_key, _ = self._condition_attr_dict_to_key_n_by(attr)
                avail_dates = np.array(sorted(self.cache[cond][cond_key].keys()))
                if date < avail_dates[0]:  # TODO
                    res[date] = None
                    skip = True
                    break
                nearest_date = [x for x in avail_dates if x <= date][-1]
                pair_df = self.cache[cond][cond_key][nearest_date]
                if pair_df is None or len(pair_df) == 0:
                    res[date] = None
                    skip = True
                    break
                pair_df_list.append(pair_df)
            if skip:
                continue
            if len(pair_df_list) > 1:
                pairs = reduce(merge_func, pair_df_list)
            elif len(pair_df_list) == 1:
                pairs = pair_df_list[0]
            else:
                res[date] = None
                continue

            nearest_date = [x for x in self.dl.dates if x <= date][-1]
            if primary_class_mat is not None:
                cls = primary_class_mat[[nearest_date]]
                a_cls = cls.loc[pairs['A'].values, nearest_date].values
                b_cls = cls.loc[pairs['B'].values, nearest_date].values
                c_mask = a_cls == b_cls
                pairs = pairs[c_mask]
            if len(pairs):
                pairs['date'] = date
                res[date] = pairs
            else:
                res[date] = None

        self.pairs_cache[hash(self.filter_conditions)].update(
            {k: v for k, v in res.items() if k not in self.pairs_cache[hash(self.filter_conditions)]})
        return res
