import bottleneck as bn
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

from .PairFilter import restore_return_DF
from .utils import foward_fillna_2darray
from .DataLoader import DataLoader


# %%
class Pair:
    # __slots__ helps save memory consumption for this class. This is necessary because it is likely that we generate
    # many thousands of Pair instances. If you'd like to add new attributes or methods, add their names to __slots__ first
    __slots__ = ('_data_dict', '_identified_date_id', 'activate_date', 'close_date', 'dl', 'exit_date', 'exit_reason',
                 'holding_days', 'identified_date', 'is_activated', 'is_terminated', 'open_date', 'pair', 'performance',
                 'type', 'parent_Pairs', 'first_identified_date', 'half_life', 'half_life_reg_pvalue')


    MAX_OBS_DAYS = 21
    MAX_HLD_DAYS = 189
    TRANSACTION_COST = 0
    STOP_LOSS = -0.6

    def __init__(self, pair_a: str, pair_b: str, identified_date: str, DataLoaderObj: DataLoader):
        """
        :param pair_a: the stock ID of stock A
        :param pair_b: the stock ID of stock B
        :param identified_date: the date when this pair is identified
        :param DataLoaderObj: an DataLoader object
        """
        self.pair = (pair_a, pair_b)
        self.identified_date = identified_date
        self.dl = DataLoaderObj
        self._identified_date_id = None
        self._data_dict = {}
        self._set_data()

        self.is_activated = False
        self.type = None
        self.activate_date = None
        self.exit_date = None
        self.open_date = None
        self.close_date = None
        self.holding_days = None
        self.exit_reason = None
        self.performance = {}
        self.is_terminated = False

    def __repr__(self):
        string = 'Pair {} and {} identified at {}'.format(*self.pair, self.identified_date)
        return string

    def __contains__(self, item):
        if item in self.pair:
            return True
        return False

    def _get_date(self, relative_idx):
        """
        return a date string given the relative index with respect to the identification date
        """
        return self.dl.dates[self._identified_date_id + relative_idx]

    @restore_return_DF
    def _set_data(self):
        try:
            endcol1 = self.dl.dates_to_indices(self.identified_date)
        except KeyError:
            argwhere = np.argwhere(self.dl.dates > self.identified_date)
            if not len(argwhere):
                raise KeyError('It seems that {} is neither a valid date, nor a date when data is available'.format(
                    self.identified_date))
            else:
                endcol1 = argwhere[0][0]
                self.identified_date = self.dl.dates[endcol1]
        # endcol = min(endcol1 + self.MAX_OBS_DAYS + self.MAX_HLD_DAYS + 1, len(self.dl))
        endcol = min(endcol1 + int((self.MAX_OBS_DAYS + self.MAX_HLD_DAYS) * 1.1), len(self.dl))
        startcol = endcol1 - 252
        assert startcol >= 0, '%s is too early to have enough data required for computation' % self.identified_date
        # according to R implementation, should minus 251, but here change it to 252 so the identified day can also be
        # trading trigger/activation day
        self._identified_date_id = endcol1
        pair_prices = self.dl['PRCCD', self.pair][:, startcol:endcol]
        pair_wealth = self.dl['CUM_WEALTH', self.pair][:, startcol:endcol]
        pair_prices = pair_prices[:, :1] * pair_wealth / pair_wealth[:, :1]

        has_na = np.isnan(pair_prices[:, 252:]).any(axis=0)
        self._data_dict['has_na'] = has_na  # start from identified
        self._data_dict['cum_na'] = has_na.cumsum()  # start from identified
        # Note: actually no missing values were found during my experiment.
        # this block might be redundant

        pair_prices = foward_fillna_2darray(pair_prices)
        ratio = np.log(pair_prices[0] / pair_prices[1])
        self._data_dict['ratio_history'] = ratio
        mean_mv = bn.move_mean(ratio, window=252, min_count=200)[251:]
        sd_mv = bn.move_std(ratio, window=252, min_count=200, ddof=1)[251:]
        # min_count is used to address the extreme case where the first 50 days are all missing data.
        # this is likely under the parameter settings of correlation computation
        ub_mv = mean_mv + 2. * sd_mv  # start from identified - 1
        lb_mv = mean_mv - 2. * sd_mv  # start from identified - 1
        ratio = ratio[251:]  # start from identified - 1

        self._data_dict['ratio'] = ratio[1:]  # start from identified
        self._data_dict['above_upper'] = np.ediff1d(np.where(ratio >= ub_mv, 1, 0))  # start from identified
        self._data_dict['above_mean'] = np.ediff1d(np.where(ratio >= mean_mv, 1, 0))
        self._data_dict['below_mean'] = np.ediff1d(np.where(ratio <= mean_mv, 1, 0))
        self._data_dict['below_lower'] = np.ediff1d(np.where(ratio <= lb_mv, 1, 0))
        self._data_dict['in_flag'] = bn.nansum(self.dl['IN_US_1', self.pair][:, endcol1:endcol], axis=0) == 2
        # start from identified

    def find_activate_date_n_direction(self, max_obs_days=None, last_day_to_activate=None):
        if max_obs_days is None:
            max_obs_days = self.MAX_OBS_DAYS

        delist = ~self._data_dict['in_flag']
        will_delist = delist.any()
        short_signal = (self._data_dict['above_upper'] == -1) & (self._data_dict['above_mean'] != -1)
        long_signal = (self._data_dict['below_lower'] == -1) & (self._data_dict['below_mean'] != -1)
        short_idxs = np.argwhere(short_signal)
        long_idxs = np.argwhere(long_signal)
        if not len(short_idxs) and not len(long_idxs):
            return None
        else:
            activate_date_rel_idx = 99999
            direction = 'short'
            if len(short_idxs) and short_idxs[0][0] <= activate_date_rel_idx:
                activate_date_rel_idx = short_idxs[0][0]
            if len(long_idxs) and long_idxs[0][0] <= activate_date_rel_idx:
                activate_date_rel_idx = long_idxs[0][0]
                direction = 'long'
            if will_delist and np.argwhere(delist)[0][0] <= (activate_date_rel_idx + 2):
                # "lookahead" here, but reasonable because we normally would know in advance
                return None
            if last_day_to_activate is None:
                if activate_date_rel_idx > max_obs_days - 1:  # assumption: the identified day is the first observation day
                    return None
                return (self._get_date(activate_date_rel_idx), activate_date_rel_idx,
                        direction)
            else:
                activate_date = self._get_date(activate_date_rel_idx)
                if activate_date > last_day_to_activate:
                    return None
                return activate_date, activate_date_rel_idx, direction

    def _measure_performance(self, long_short_wealth: np.ndarray, holding_returns):
        long_short_wealth[:, -1] = long_short_wealth[:, -1] * np.array(
            [1 - self.TRANSACTION_COST, 1 + self.TRANSACTION_COST])
        # 1 day forward wealth start from activate day
        holding_returns[-1] = (long_short_wealth[0, -1] - long_short_wealth[1, -1])
        self.performance['cumulative_returns'] = holding_returns
        # 1 day forward return start from activate day
        self.performance['holding_period_return'] = holding_returns[-1]
        min_cum_return = holding_returns.min()
        self.performance['max_negative_return'] = min_cum_return if min_cum_return < 0. else np.nan
        self.performance['cumulative_mdd'] = (1 + holding_returns) / np.maximum.accumulate(1 + holding_returns) - 1
        self.performance['mdd'] = min(self.performance['cumulative_mdd'])

    @restore_return_DF
    def _calculate_half_life(self):
        endcol = self.dl.dates_to_indices(self.identified_date)
        pair_prices = foward_fillna_2darray(self.dl['PRCCD', self.pair][:, (endcol - 251):(endcol + 1)], 1)
        pair_wealth = self.dl['CUM_WEALTH', self.pair][:, (endcol - 251):(endcol + 1)]
        pair_prices = pair_prices[:, :1] * pair_wealth / pair_wealth[:, :1]
        if np.any(np.isnan(pair_prices)):
            return None, None
        ratio = np.log(pair_prices[0] / pair_prices[1])
        ols_res = linregress(ratio[:-1], ratio[1:] - ratio[:-1])
        self.half_life_reg_pvalue = ols_res.pvalue
        half_life = -np.log(2) / ols_res.slope
        self.half_life = half_life
        return half_life, self.half_life_reg_pvalue

    def update_PIT(self, date=None):
        pass

    @restore_return_DF
    def update_to_termination(self, max_holding_days=None,
                              max_obs_days=None, last_day_to_activate=None, use_half_life=False):
        """
        Backtest the pair until termination
        :param max_holding_days: maximum holding days before automatic termination
        :param max_obs_days: the number of days after identification date the pair remains effective and elgible for trade;
                             also named tracking window somewhere else in this package
        :param last_day_to_activate:
        :param use_half_life: if True, the half life condition (both the absolute value and the coefficient significance)
                              will apply
        """
        if max_obs_days is None:
            max_obs_days = self.MAX_OBS_DAYS

        if max_holding_days is None:
            max_holding_days = self.MAX_HLD_DAYS

        if use_half_life:
            half_life, hl_pvalue = self._calculate_half_life()
            if half_life is None or half_life <= 0:
                self.is_terminated = True
                return
            if half_life * 2 > max_holding_days:
                self.is_terminated = True
                return
            if hl_pvalue > 0.05:
                self.is_terminated = True
                return

        activate_date_n_direction = self.find_activate_date_n_direction(max_obs_days=max_obs_days,
                                                                        last_day_to_activate=last_day_to_activate)
        if activate_date_n_direction:

            self.is_activated = True
            self.activate_date, activate_date_rel_idx, self.type = activate_date_n_direction
            if self.type == 'long':
                long_leg = self.pair[0]
                short_leg = self.pair[1]
                exit_signal = self._data_dict['below_mean'] == -1
            else:
                long_leg = self.pair[1]
                short_leg = self.pair[0]
                exit_signal = self._data_dict['above_mean'] == -1
            exit_idxs = np.argwhere(exit_signal).flatten()
            exit_date_rel_idxs = exit_idxs[exit_idxs > activate_date_rel_idx]
            # find exit date

            if not len(exit_date_rel_idxs):
                exit_date_rel_idx = activate_date_rel_idx + max_holding_days
                exit_reason = 'max.holding.days'
            else:
                exit_date_rel_idx = exit_date_rel_idxs[0]
                exit_reason = 'reverted'
                if exit_date_rel_idx > activate_date_rel_idx + max_holding_days:  # reversion on the last day is still reversion
                    exit_date_rel_idx = activate_date_rel_idx + max_holding_days
                    exit_reason = 'max.holding.days'

            # get forward return
            # example: day 1 activate; day 2 buy at close price; day 10 exit signal, day 11 sell at close price
            # need the forward return of day 2 to day 10
            sl = slice(self._identified_date_id + activate_date_rel_idx + 1,
                       self._identified_date_id + exit_date_rel_idx + 1)
            forward_returns = self.dl['FRTN1P', [long_leg, short_leg]][:, sl]
            # start from the the day after the activate date
            forward_returns[np.isnan(forward_returns)] = 0
            # wealth after deducting the cost when initiating the position
            wealth = np.cumprod(1 + forward_returns, axis=1) * np.array(
                [[1 / (1 + self.TRANSACTION_COST)], [1 / (1 - self.TRANSACTION_COST)]])
            wealth = np.c_[([1, 1], wealth)]  # start from the the activate day (wealth = 1)
            # one day forward wealth
            holding_returns = (wealth[0] - wealth[1])  # start from the the activate date
            # one day forward pair return

            # check stoploss point
            stop_loss_idxs = np.argwhere(holding_returns <= self.STOP_LOSS)
            # day x,  the 1 day forward return <= stoploss. The stoploss signal is detected on day x + 1
            stop_loss_idx = stop_loss_idxs[0][0] + 1 if len(stop_loss_idxs) else 99999

            # get delisting information and check for delisting
            # delist = ~self._data_dict['in_flag'][activate_date_rel_idx + 2: exit_date_rel_idx + 1]
            delist = ~self._data_dict['in_flag'][activate_date_rel_idx + 2: exit_date_rel_idx + 2]
            # start from the the second day after the activate day
            delist_idxs = np.argwhere(delist)
            delist_idx = delist_idxs[0][0] if len(delist_idxs) else 99999
            # if delist_idx == 0, then delisting happens the second day after the activate day which corresponds to a index
            # value of 2 relative to the activate date. But we need not adjust delist_idx to 2 because we can assume
            # on day 1 we successfully clear the position (at its close price). The 1 day forward return on day 2
            # and day 1 thus should not be counted toward the pair performance

            breakpoint = None  # by default, no breaks caused by either stop loss or delisting.
            if min(delist_idx, stop_loss_idx) != 99999:
                if min(delist_idx, stop_loss_idx) == delist_idx:
                    exit_reason = 'delist'
                    breakpoint = delist_idx
                else:
                    if stop_loss_idx < len(holding_returns):
                        exit_reason = 'stop.loss'
                        breakpoint = stop_loss_idx

            self.exit_reason = exit_reason
            self.open_date = self._get_date(activate_date_rel_idx + 1)
            if breakpoint is not None:
                exit_date_rel_idx = activate_date_rel_idx + breakpoint
            self.exit_date = self._get_date(exit_date_rel_idx)
            self.close_date = self._get_date(exit_date_rel_idx + 1)
            self.holding_days = exit_date_rel_idx - activate_date_rel_idx
            if breakpoint is not None:
                self._measure_performance(wealth[:, :breakpoint + 1], holding_returns[:breakpoint + 1])
            else:
                self._measure_performance(wealth, holding_returns)

        self.is_terminated = True

    @restore_return_DF
    def _set_data_for_visualization(self, days_before_identification=21, days_after_close=10):
        startcol = max(self._identified_date_id - 252 - days_before_identification + 1, 0)
        endcol = min(self._identified_date_id + self.MAX_OBS_DAYS + self.MAX_HLD_DAYS + days_after_close + 1,
                     len(self.dl))
        pair_prices = self.dl['PRCCD', self.pair][:, startcol:endcol]
        pair_wealth = self.dl['CUM_WEALTH', self.pair][:, startcol:endcol]
        pair_prices = pair_prices[:, :1] * pair_wealth / pair_wealth[:, :1]
        pair_prices = foward_fillna_2darray(pair_prices)

        ratio = np.log(pair_prices[0] / pair_prices[1])
        mean_mv = bn.move_mean(ratio, window=252, min_count=200)[251:]
        sd_mv = bn.move_std(ratio, window=252, min_count=200, ddof=1)[251:]
        ub_mv = mean_mv + 2. * sd_mv  # start from identified - days_before_identification
        lb_mv = mean_mv - 2. * sd_mv  # start from identified - days_before_identification
        ratio = ratio[251:]  # start from identified - days_before_identification

        idtf_idx = self._identified_date_id - startcol - 251
        open_idx = self.dl.dates_to_indices(self.open_date) - self._identified_date_id + idtf_idx
        close_idx = self.dl.dates_to_indices(self.close_date) - self._identified_date_id + idtf_idx
        end_idx = close_idx + days_after_close

        return {'ratio': ratio[:end_idx + 1], 'upper': ub_mv[:end_idx + 1], 'lower': lb_mv[:end_idx + 1],
                'mean': mean_mv[:end_idx + 1],
                'open_idx': open_idx, 'close_idx': close_idx, 'idtf_idx': idtf_idx}

    def visualize(self, days_before_identification=21, days_after_close=0):
        """
        Visualize the pairs backtesting result if any trade is triggered
        :param days_before_identification: number of days before the identification date to be displayed in the plot
        :param days_after_close: number of days after the close date to be displayed in the plot
        """
        assert days_before_identification >= 0 and days_after_close >= 0
        if self.is_activated:
            viz_data = self._set_data_for_visualization(days_before_identification, days_after_close)
            idtf_idx = viz_data['idtf_idx']
            open_idx = viz_data['open_idx']
            close_idx = viz_data['close_idx']
            plt.figure(figsize=(10, 7))
            ax1 = plt.subplot2grid((10, 10), (0, 0), 7, 10)
            ax2 = plt.subplot2grid((10, 10), (7, 0), 3, 10, sharex=ax1)
            ax1.plot(viz_data['ratio'], color='black')
            ax1.plot(viz_data['upper'], color='black', linestyle='dotted')
            ax1.plot(viz_data['lower'], color='black', linestyle='dotted')
            ax1.plot(viz_data['mean'], color='black', linestyle='dotted')
            shift = 1 if self.type == 'long' else -1
            ud_range = viz_data['upper'].max() - viz_data['lower'].min()
            # lr_range = len(viz_data['upper'])
            ax1.annotate('identified on\n{}'.format(self.identified_date), xy=(idtf_idx, viz_data['ratio'][idtf_idx]),
                         xytext=(idtf_idx, viz_data['ratio'][idtf_idx] + shift * ud_range * 0.25),
                         arrowprops=dict(arrowstyle='-|>', facecolor='black'), fontsize='x-large',
                         horizontalalignment="center")
            # ax1.plot([open_idx], [viz_data['ratio'][open_idx]], marker='o', markersize=10, color="black")
            ax1.annotate('opened on\n{}'.format(self.open_date),
                         xy=(open_idx, viz_data['ratio'][open_idx] - shift * 0.01 * ud_range),
                         xytext=(open_idx,
                                 viz_data['ratio'][open_idx] - shift * ud_range * 0.2),
                         arrowprops=dict(arrowstyle='-|>', facecolor='black'), fontsize='x-large',
                         horizontalalignment="center")
            # ax1.plot([close_idx], [viz_data['ratio'][close_idx]], marker='X', markersize=10, color="black")
            ax1.annotate('closed on\n{}\n"{}"'.format(self.close_date, self.exit_reason),
                         xy=(close_idx, viz_data['ratio'][close_idx] + shift * 0.01 * ud_range),
                         xytext=(close_idx,
                                 viz_data['ratio'][close_idx] + shift * ud_range * 0.25),
                         arrowprops=dict(arrowstyle='-|>', facecolor='black'), fontsize='x-large',
                         horizontalalignment="center")
            ax1.set_title('{} & {}'.format(*self.pair), fontdict=dict(fontsize='x-large'))
            ax2.bar(range(open_idx, close_idx + 1), self.performance['cumulative_returns'], color='black')
            ax2.set_title('Cumulative Returns')
            plt.show()
        else:
            if not self.is_terminated:
                self.update_to_termination()
                self.visualize()
            else:
                print('No trade is triggered during the first {} days after this pair is identified'.format(
                    self.MAX_OBS_DAYS))
