import datetime
import re
from collections import defaultdict
from functools import wraps

import numpy as np
import pandas as pd
from tqdm import tqdm

from .Pair import Pair
from .PairFilter import PairFilter
from .utils import frequency_convert


# %%
def restore_Pairs_params(f):
    """
    A decorator. The class Pair has default values for MAX_OBS_DAYS (max observation days) and MAX_HLD_DAYS (max
    holding days). During backtesting the actual max observation days and max holding days can be larger than default.
    If this is the case then before the execution of the decorated function, this decorator will temporarily change the
    default values.
    """

    @wraps(f)
    def wrapper(self, *args, **kwargs):
        max_holding_days = int(self.holding_window[:-1])
        max_obs_days_temp = int(self.tracking_window[:-1]) * 21 if self.tracking_window[-1] == 'M' else int(
            self.tracking_window[:-1])
        restore_default = False
        max_holding_days_orig, max_obs_days_orig = Pair.MAX_HLD_DAYS, Pair.MAX_OBS_DAYS
        if max_holding_days > Pair.MAX_HLD_DAYS:
            Pair.MAX_HLD_DAYS = max_holding_days
            restore_default = True
        if max_obs_days_temp > Pair.MAX_OBS_DAYS:
            Pair.MAX_OBS_DAYS = max_obs_days_temp
            restore_default = True
        result = f(self, *args, **kwargs)
        if restore_default:
            Pair.MAX_HLD_DAYS, Pair.MAX_OBS_DAYS = max_holding_days_orig, max_obs_days_orig
        return result

    return wrapper


# %%
class PairBacktesting:

    def __init__(self, pair_filter: PairFilter,
                 start_date, identification_freq: str = '1M', end_date=None, holding_window: str = None,
                 tracking_window: str = None, cooling_window: str = '1D', use_half_life=True,
                 max_reactivate_rounds=0, same_type_constraint=False):
        """
        :param pair_filter: a PairFilter object; note that you must create an effective filter_condition for the
                            PairFilter object via PairFilter.new_filter() before passing it as the parameter; depending
                            on your preference, you may call PairFilter.filter_given_days() beforehand; however, it is
                            discouraged because PairBacktesting will automatically calculate the dates based on your input
                            of identification frequency, start date, and end date, and then call PairFilter.filter_given_days()
                            for you
        :param start_date: the start date of the backtesting
        :param identification_freq: 'xM' or 'xD' or 'M' (equivalent to '1M') or 'D' (equivalent to '1D')
        :param end_date: the end date of the backtesting
        :param holding_window: the maximum holding period; 'xM' or 'xD'
        :param tracking_window: the period after identification when the identified pairs remain valid; 'xM' or 'xD';
                                must use 'xM' if identification_freq is 'xM'; note that in other places of this package
                                this is also called observation days
        :param cooling_window: ; the minimum allowed days between the consecutive activations of two positions involving
                                 the same pair; 'xD'
        :param use_half_life: if True, half life conditions (both the absolute value and the coefficient significance) apply
        :param max_reactivate_rounds: the maximum allowed times during the "tracking window" an already activated pair
                                      can open new positions (reactivation); if None, there is no limit
        :param same_type_constraint: if True, and if max_reactivate_rounds is not zero, during "reactivation" a new position
                                     is not legitimate if the last opened position involving the same pair is of the same direction
        """
        self.filter = pair_filter
        self.dl = pair_filter.dl
        self.start_date = self.check_date_format(start_date)
        self.end_date = self.check_date_format(end_date)
        self.identification_freq = self.check_freq_format(identification_freq, '1M')

        self.holding_window, self.tracking_window, self.cooling_window = self._check_windows(holding_window,
                                                                                             tracking_window,
                                                                                             cooling_window,
                                                                                             self.identification_freq)

        self.identification_dates = self._get_identification_days(self.start_date, self.identification_freq,
                                                                  self.end_date)

        self.use_half_life = use_half_life
        self.max_reactivate_rounds = max_reactivate_rounds
        self.same_type_constraint = same_type_constraint
        self.activated_pairs = defaultdict(list)
        self.identified_pairs = None
        # self.closed_positions = {}
        self._non_activated_data = []

    def __repr__(self):
        return 'PairBacktesting(' + ', '.join(['{}: {}'.format(x, getattr(self, x)) for x in (
            'start_date', 'end_date', 'identification_freq', 'holding_window',
            'tracking_window',
            'cooling_window', 'max_reactivate_rounds', 'same_type_constraint')]) + ')'

    def get_summary_table(self, include_non_activated=True, append_stats=True) -> pd.DataFrame:
        """
        :param include_non_activated: if True, non-activated pairs will also be reflected in the returned DataFrame
        :param append_stats: if True, all the statistics used during pair selection will be appended
        """
        activated_data_list = []
        for date, pairs in self.activated_pairs.items():
            for p in pairs:
                if append_stats:
                    activated_data_list.append((date, p.identified_date, p.pair[0],
                                                p.pair[1], p.type, p.open_date, p.close_date, p.exit_reason,
                                                p.holding_days, p.performance['holding_period_return'],
                                                p.performance['mdd'],
                                                p.performance['max_negative_return'], p.half_life,
                                                p.half_life_reg_pvalue))
                else:
                    activated_data_list.append((date, p.identified_date, p.pair[0],
                                                p.pair[1], p.type, p.open_date, p.close_date, p.exit_reason,
                                                p.holding_days, p.performance['holding_period_return'],
                                                p.performance['mdd'],
                                                p.performance['max_negative_return']))
        if append_stats:
            df = pd.DataFrame(activated_data_list,
                              columns=['first_identified_date', 'identified_date', 'A', 'B',
                                       'type', 'open_date', 'close_date', 'exit_reason',
                                       'holding_days', 'return', 'mdd', 'max_neg_return', 'half_life',
                                       'half_life_reg_pvalue'])
        else:
            df = pd.DataFrame(activated_data_list,
                              columns=['first_identified_date', 'identified_date', 'A', 'B',
                                       'type', 'open_date', 'close_date', 'exit_reason',
                                       'holding_days', 'return', 'mdd', 'max_neg_return'])
        if include_non_activated:
            non_activated_df = pd.DataFrame(self._non_activated_data,
                                            columns=['A', 'B', 'first_identified_date', 'half_life',
                                                     'half_life_reg_pvalue'])
            df = df.append(non_activated_df, sort=False)

        if append_stats:
            df = df.merge(pd.concat(self.identified_pairs.values()), left_on=['A', 'B', 'first_identified_date'],
                          right_on=['A', 'B', 'date'])
            df.drop(['class', 'date'], axis=1, inplace=True)

        comp_names = self.dl.get_processed_data('COMP_NAMES')
        df = df.merge(comp_names, how='left', left_on='A', right_on='id')
        df = df.loc[(df['first_identified_date'] <= df['datelast']) & (df['first_identified_date'] >= df['datefirst'])]
        df.drop(['id', 'datefirst', 'datelast'], axis=1, inplace=True)
        df = df.merge(comp_names, how='left', left_on='B', right_on='id', suffixes=['_A', '_B'])
        df = df.loc[(df['first_identified_date'] <= df['datelast']) & (df['first_identified_date'] >= df['datefirst'])]
        df.drop(['id', 'datefirst', 'datelast'], axis=1, inplace=True)
        df.sort_values(by=['first_identified_date', 'open_date', 'A', 'B'], inplace=True)
        df.reset_index(drop=True, inplace=True)

        return df

    def _record_data(self, pair: Pair):
        A, B = pair.pair
        variables = ['identified_date', 'half_life', 'half_life_reg_pvalue']
        self._non_activated_data.append((A, B, *(getattr(pair, var, None) for var in variables)))

    @restore_Pairs_params
    def run(self):
        """
        run the backtesting
        """
        self._non_activated_data.clear()
        self.activated_pairs.clear()
        pairs_dict = self.filter.filter_given_dates(dates=self.identification_dates)
        self.identified_pairs = pairs_dict
        reference_dates = frequency_convert(self.dl.dates, self.identification_freq[-1])
        max_holding_days = int(self.holding_window[:-1])
        cooling_window = None if self.cooling_window is None else int(self.cooling_window[:-1])
        previous_activate_dates = defaultdict(list)
        tqdm_bar = tqdm(pairs_dict.items())
        for date, pairs_df in tqdm_bar:
            tqdm_bar.set_description(date)
            date_idx = self.dl.dates_to_indices(date)
            if self.tracking_window[-1] == 'M':
                max_obs_days = None
                idtf_idx = np.argwhere(reference_dates == date)[0, 0]
                next_month_end = reference_dates[
                    idtf_idx + int(self.tracking_window[:-1])]  # TODO: potential outofbound error here
                nme_idx = self.dl.dates_to_indices(next_month_end)
                last_day_to_activate = self.dl.dates[nme_idx - 1]
            else:
                max_obs_days = int(self.tracking_window[:-1])
                last_day_to_activate = None
            for _, row in pairs_df.iterrows():
                pair_tuple = tuple(row[['A', 'B']])
                if cooling_window is not None and len(
                        previous_activate_dates) and pair_tuple in previous_activate_dates:
                    idtf_date_idx = max(
                        self.dl.dates_to_indices(max(previous_activate_dates[pair_tuple])) + cooling_window, date_idx)
                    p = Pair(*pair_tuple, identified_date=self.dl.dates[idtf_date_idx], DataLoaderObj=self.dl)
                    if max_obs_days is None:
                        p.update_to_termination(max_holding_days=max_holding_days,
                                                last_day_to_activate=last_day_to_activate,
                                                use_half_life=self.use_half_life)
                    else:
                        p.update_to_termination(max_holding_days=max_holding_days,
                                                max_obs_days=max_obs_days - (idtf_date_idx - date_idx),
                                                use_half_life=self.use_half_life)
                else:
                    p = Pair(*pair_tuple, identified_date=date, DataLoaderObj=self.dl)
                    p.update_to_termination(max_holding_days=max_holding_days, max_obs_days=max_obs_days,
                                            last_day_to_activate=last_day_to_activate,
                                            use_half_life=self.use_half_life)
                if p.is_activated:
                    self.activated_pairs[date].append(p)
                else:
                    self._record_data(p)
            if cooling_window is not None and self.max_reactivate_rounds != 0:
                self._reactivate(self.activated_pairs[date], date, max_obs_days, last_day_to_activate, cooling_window,
                                 max_holding_days, max_reactivate_rounds=self.max_reactivate_rounds,
                                 same_type_constraint=self.same_type_constraint)
                previous_activate_dates.clear()
                for p in self.activated_pairs[date]:
                    previous_activate_dates[p.pair].append(p.activate_date)

    def _reactivate(self, activated_pairs, first_identified_date, max_obs_days, last_day_to_activate, cooling_window,
                    max_holding_days, max_reactivate_rounds=None, same_type_constraint=False):
        """
        implementation of the "reactivation"
        """
        first_idtf_idx = self.dl.dates_to_indices(first_identified_date)
        pairs_multi_acts = []
        pairs_multi_acts_same_type = []
        activated_pairs = activated_pairs.copy()
        if max_reactivate_rounds is None:
            max_reactivate_rounds = 999
        for round in range(max_reactivate_rounds):
            for p in activated_pairs:
                activate_idx = self.dl.dates_to_indices(p.activate_date)
                if max_obs_days is not None:
                    if activate_idx - first_idtf_idx + cooling_window < max_obs_days:
                        p_new = Pair(*p.pair, identified_date=self.dl.dates[activate_idx + cooling_window],
                                     DataLoaderObj=self.dl)
                        p_new.update_to_termination(max_holding_days=max_holding_days,
                                                    max_obs_days=max_obs_days - (
                                                            activate_idx - first_idtf_idx) - cooling_window,
                                                    use_half_life=self.use_half_life)
                        if p_new.is_activated:
                            p_new.parent_Pairs = p if not hasattr(p, 'parent_Pairs') else p.parent_Pairs
                            # p_new.first_identified_date = p.parent_Pairs.identified_date
                            if same_type_constraint and p_new.type == p.type:
                                pairs_multi_acts_same_type.append(p_new)
                            else:
                                pairs_multi_acts.append(p_new)
                else:
                    if activate_idx + cooling_window <= self.dl.dates_to_indices(last_day_to_activate):
                        p_new = Pair(*p.pair, identified_date=self.dl.dates[activate_idx + cooling_window],
                                     DataLoaderObj=self.dl)
                        p_new.update_to_termination(max_holding_days=max_holding_days,
                                                    last_day_to_activate=last_day_to_activate,
                                                    use_half_life=self.use_half_life)
                        if p_new.is_activated:
                            p_new.parent_Pairs = p if not hasattr(p, 'parent_Pairs') else p.parent_Pairs
                            # p_new.first_identified_date = p.parent_Pairs.identified_date
                            if same_type_constraint and p_new.type == p.type:
                                pairs_multi_acts_same_type.append(p_new)
                            else:
                                pairs_multi_acts.append(p_new)
            if pairs_multi_acts:
                self.activated_pairs[first_identified_date].extend(pairs_multi_acts)
                if round + 1 < max_reactivate_rounds:
                    activated_pairs = pairs_multi_acts + pairs_multi_acts_same_type
                    del pairs_multi_acts[:]
                    del pairs_multi_acts_same_type[:]
            else:
                break

    @classmethod
    def check_date_format(cls, date):
        if date is not None:
            try:
                datetime.datetime.strptime(date, '%Y-%m-%d')
            except ValueError:
                raise ValueError("Incorrect date format, should be YYYY-MM-DD")
        return date

    @classmethod
    def check_freq_format(cls, freq, default):
        if freq is not None:
            assert re.match(r'([1-9]\d{0,3})?[mMdD]', freq), 'Wrong freq format, should be like "M", "D", "2M", "252D"'
            return freq.upper()
        else:
            return default

    def _check_windows(self, holding_window, tracking_window, cooling_window, idtf_freq):
        """
        Check whether identification_freq, holding_window, tracking_window, and their combination, are valid
        """
        current_freq_unit = idtf_freq[-1]
        if holding_window is None:
            holding_window = '189D'
        elif isinstance(holding_window, str):
            holding_window = self.check_freq_format(holding_window, None)
            _ = holding_window
            if holding_window[-1] == 'M':
                holding_window = str(int(holding_window[:-1]) * 21) + 'D'
                print(
                    'Automatically convert holding_window from {} to {} (21 days per month)'.format(_, holding_window))
        else:
            raise ValueError(
                'holding_window {} (type: {}) is unsupported '.format(holding_window, type(holding_window)))

        if tracking_window is None:
            tracking_window = idtf_freq  # pay attention to this
        elif isinstance(tracking_window, str):
            tracking_window = self.check_freq_format(tracking_window, None)
            if tracking_window[-1] == 'M' and current_freq_unit == 'D':
                _ = tracking_window
                tracking_window = str(int(tracking_window[:-1]) * 21) + 'D'
                print(
                    'Automatically convert tracking_window from {} to {} (21 days per month)'.format(_,
                                                                                                     tracking_window))
        else:
            raise ValueError(
                'tracking_window {} (type: {}) is unsupported '.format(tracking_window, type(tracking_window)))

        if cooling_window is None:
            pass
        elif isinstance(cooling_window, str):
            cooling_window = self.check_freq_format(cooling_window, None)
            _ = cooling_window
            if cooling_window[-1] == tracking_window[-1]:
                assert int(cooling_window[:-1]) < int(
                    tracking_window[:-1]), 'cooling_window {} is greater than or equal to tracking_window {}'.format(
                    cooling_window, tracking_window)
            elif cooling_window[-1] == 'D':  # tracking_window[-1] == 'M'
                assert int(cooling_window[:-1]) < 21 * int(tracking_window[
                                                           :-1]), 'since tracking_window is {}, the maximum cooling_window allowed is {}'.format(
                    tracking_window, '%dD' % (21 * int(tracking_window[:-1]) - 1))
            else:  # cooling_window[-1] == 'M' & tracking_window[-1] == 'D'
                assert 21 * int(cooling_window[:-1]) < int(tracking_window[
                                                           :-1]), 'since tracking_window is {}, a cooling_window of {} is not allowed which might be greater than {}'.format(
                    tracking_window, cooling_window, tracking_window)
            if cooling_window[-1] == 'M':
                cooling_window = str(int(cooling_window[:-1]) * 21) + 'D'
                print(
                    'Automatically convert cooling_window from {} to {} (21 days per month)'.format(_, cooling_window))
        else:
            raise ValueError(
                'cooling_window {} (type: {}) is unsupported '.format(cooling_window, type(cooling_window)))

        return holding_window, tracking_window, cooling_window

    @restore_Pairs_params
    def _get_identification_days(self, start_date, identification_freq: str = None, end_date=None):
        """
        Automatically get the list of dates to be passed into PairFilter.filter_given_dates()
        """
        step = 1 if len(identification_freq) == 1 else int(identification_freq[:-1])
        unit = 'M' if identification_freq[-1] == 'M' else 'D'
        dates_resampled = frequency_convert(self.dl.dates, unit)
        assert start_date <= dates_resampled[-1], \
            '{} is later than {}, the last day when data is available at frequency {}'.format(start_date,
                                                                                              dates_resampled[-1],
                                                                                              identification_freq[-1])
        if start_date < dates_resampled[0]:
            print('[Warning] {} is earlier than {}, the first day when data is available at frequency {}'.format(
                start_date, dates_resampled[0], identification_freq[-1]))
            new_start_date = dates_resampled[0]
        else:
            new_start_date = dates_resampled[np.argwhere(dates_resampled >= start_date)[0, 0]]

        new_flag = False
        if new_start_date != start_date:
            new_flag = True
            print(
                'Adjust the start date from {} to {} according to the availability of data and the market calendar'.format(
                    start_date, new_start_date))

        if new_start_date < self.dl.dates[252]:
            print(
                '[WARNING]: The {}start date {} is earlier than {}, the first day when any time-series statistics requiring 252 days of data might be available, e.g. beta calculated using return series of the last 252 days'.format(
                    'new ' if new_flag else '', new_start_date, self.dl.dates[252]))
        if new_start_date < self.dl.dates[504]:
            print(
                '[WARNING]: The {}start date {} is earlier than {}, the first day when any time-series statistics requiring 504 (252 x 2) days of data might be available, e.g. 252-day residual return correlations between stocks'.format(
                    'new ' if new_flag else '', new_start_date, self.dl.dates[504]))

        if end_date is not None:
            assert end_date >= new_start_date, 'The backtesting end date {} is earlier than the {}start date {}'.format(
                end_date, 'new' if new_flag else '', new_start_date)

            if end_date > dates_resampled[-1]:
                print('[Warning] {} is later than {}, the last day when data is available at frequency {}'.format(
                    end_date, dates_resampled[-1], identification_freq[-1]))
                new_end_date = dates_resampled[-1]
            else:
                new_end_date = dates_resampled[np.argwhere(dates_resampled <= end_date)[-1, 0]]
        else:
            new_end_date = dates_resampled[
                np.argwhere(dates_resampled <= self.dl.dates[-(Pair.MAX_OBS_DAYS + Pair.MAX_HLD_DAYS)])[
                    -1, 0]]  # TODO

        s_idx = list(dates_resampled).index(new_start_date)
        e_idx = list(dates_resampled).index(new_end_date)
        identification_dates = dates_resampled[slice(s_idx, e_idx + 1, step)]
        new_end_date = identification_dates[-1]

        if end_date is not None:
            if new_end_date != end_date:
                print(
                    'Adjust the end date from {} to {} according to the availability of data, the frequency setting, and the market calendar'.format(
                        end_date, new_end_date))
        else:
            print('The end date of backtesting is set to {} as default'.format(new_end_date))

        return identification_dates
