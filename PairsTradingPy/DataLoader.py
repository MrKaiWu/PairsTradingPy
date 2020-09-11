import re
from collections import defaultdict
from os import path
from typing import Union, Sequence

import bottleneck as bn
import h5py
import numpy as np
import pandas as pd

from .utils import shift_2darray

DATA_DIR = path.join(path.dirname(path.dirname(path.abspath(__file__))), 'data')


class DataLoader:
    # make sure to expand this list if you wanna load additional datasets by default
    VAR_LIST = ['CUM_DIV', 'IN_US_1', 'PRCCD', 'QES_GSECTOR', 'GGROUP', 'GIND', 'GSUBIND']

    def __init__(self):
        self._dict = {}
        self._IDs = None
        self._dates = None
        self._IDs_dict = None
        self._dates_dict = None

        for _, ff in enumerate(self.VAR_LIST):
            with h5py.File(path.join(DATA_DIR, '{}.h5'.format(ff)), 'r') as h:
                if _ == 0:
                    self._IDs = h['ID'][...].astype(str)
                    self._IDs_dict = {ID: i for i, ID in enumerate(self._IDs)}
                    self._dates = h['date'][...].astype(str)
                self._dict[ff] = h['data'][...]

        self._processed_dataset = {}
        self._preprocess()
        self._return_raw = False

    @property
    def return_raw(self):
        # if self._return_raw:
        #     print('return numpy array by default.')
        # else:
        #     print('return DataFrame by default.')
        return self._return_raw

    @return_raw.setter
    def return_raw(self, value: bool):
        """
        usage: DataLoaderObj.return_raw = True
        """
        assert isinstance(value, bool)
        self._return_raw = value
        # if self._return_raw:
        #     print('return numpy array by default.')
        # else:
        #     print('return DataFrame by default.')

    def __getitem__(self, item):
        """
        Ways to retrieve data from a DataLoader object:
         1)  DataLoaderObj[dataset_name]
         2)  DataLoaderObj[dataset_name, a single ID]
         3)  DataLoaderObj[dataset_name, a sequence of IDs]
         4)  DataLoaderObj[dataset_name, a single ID, a single date]
         5)  DataLoaderObj[dataset_name, a single ID, a sequence of dates]
         6)  DataLoaderObj[dataset_name, a single ID, :]
         7)  DataLoaderObj[dataset_name, a single ID, date1:date2] (requires date2 > date1, inclusive)
         8)  DataLoaderObj[dataset_name, a sequence of IDs, a single date]
         9)  DataLoaderObj[dataset_name, a sequence of IDs, a sequence of dates]
         10) DataLoaderObj[dataset_name, a sequence of IDs, :]
         11) DataLoaderObj[dataset_name, a sequence of IDs, date1:date2]
         12) DataLoaderObj[dataset_name, :, a single date]
         13) DataLoaderObj[dataset_name, :, a sequence of dates]
         14) DataLoaderObj[dataset_name, :, date1:date2]
        Whether a DataFrame or a numpy array is returned is dependent on self.return_raw
        Except for 4) which returns a single value
        """
        if isinstance(item, str):
            if self._return_raw:
                return self._dict[item]
            return pd.DataFrame(self._dict[item], index=self._IDs, columns=self._dates)
        elif len(item) == 2:
            dataset, IDs = item
            if isinstance(IDs, str):
                if self._return_raw:
                    return self._dict[dataset][[self._IDs_dict[IDs]]]
                return pd.DataFrame(self._dict[dataset][[self._IDs_dict[IDs]]], index=[IDs, ], columns=self._dates)
            elif isinstance(IDs, Sequence):
                row_slice = [self._IDs_dict[x] for x in IDs]
                if self._return_raw:
                    return self._dict[dataset][row_slice]
                return pd.DataFrame(self._dict[dataset][row_slice], index=IDs, columns=self._dates)
            else:
                raise IndexError('unsupported slice')
        elif len(item) == 3:
            dataset, IDs, dates = item
            if isinstance(IDs, str) and isinstance(dates, str):
                return self._dict[dataset][self._IDs_dict[IDs], self._dates_dict[dates]]
            else:
                is_list_row, is_list_col = False, False
                if isinstance(IDs, str):
                    row_slice = self._IDs_dict[IDs]
                    row_slice = slice(row_slice, row_slice + 1)
                    if not self._return_raw:
                        idx = [IDs, ]
                elif isinstance(IDs, Sequence):
                    row_slice = [self._IDs_dict[x] for x in IDs]
                    is_list_row = True
                    if not self._return_raw:
                        idx = IDs
                elif IDs == slice(None, None, None):
                    row_slice = IDs
                    if not self._return_raw:
                        idx = self._IDs
                else:
                    raise IndexError('unsupported slice')
                if isinstance(dates, str):
                    col_slice = self._dates_dict[dates]
                    col_slice = slice(col_slice, col_slice + 1)
                    if not self._return_raw:
                        cols = [dates, ]
                elif isinstance(dates, Sequence):
                    col_slice = [self._dates_dict[x] for x in dates]
                    is_list_col = True
                    if not self._return_raw:
                        cols = dates
                elif dates == slice(None, None, None):
                    col_slice = dates
                    if not self._return_raw:
                        cols = self._dates
                elif isinstance(dates, slice):
                    start_date = self._dates[0] if dates.start is None else dates.start
                    end_date = self._dates[-1] if dates.stop is None else dates.stop
                    assert re.match(r'\d{4}-\d{2}-\d{2}', start_date) and re.match('\d{4}-\d{2}-\d{2}', end_date), \
                        'wrong date format'
                    start_id = np.searchsorted(self._dates, start_date)
                    stop_id = np.searchsorted(self._dates, end_date, side='right')
                    col_slice = slice(start_id, stop_id, None)
                    if not self._return_raw:
                        cols = self._dates[col_slice]
                else:
                    raise IndexError('unsupported slice')
            if self._return_raw:
                if is_list_col and is_list_row:
                    return self._dict[dataset][row_slice][:, col_slice]
                return self._dict[dataset][row_slice, col_slice]
            else:
                if is_list_col and is_list_row:
                    return pd.DataFrame(self._dict[dataset][row_slice][:, col_slice], columns=cols, index=idx)
                return pd.DataFrame(self._dict[dataset][row_slice, col_slice], columns=cols, index=idx)

    def get_processed_data(self, dataset_name):
        """
        Load a dataset that is not loaded during initialization or comes from additional processing of the loaded ones;
        once loaded, it will be stored and further retrieval will be very fast
        :param dataset_name: {'MKT_RTN_EQUI', 'IND_RTN_EQUI', 'COMP_NAMES'}
        """
        # make sure to modify this line if new data is added
        assert dataset_name in ['MKT_RTN_EQUI', 'IND_RTN_EQUI', 'COMP_NAMES']
        if dataset_name in self._processed_dataset:
            return self._processed_dataset[dataset_name]
        elif dataset_name == 'MKT_RTN_EQUI':
            rtn = self._dict['RTN'].copy()
            rtn[self._dict['IN_US_1'] != 1] = np.nan
            self._processed_dataset['MKT_RTN_EQUI'] = pd.DataFrame(bn.nanmean(rtn, axis=0)[None, :],
                                                                   columns=self._dates)
        elif dataset_name == 'IND_RTN_EQUI':
            ind = self._dict['GIND']
            rtn = self._dict['RTN']
            unique_inds = np.unique(ind[~np.isnan(ind)])
            shape = rtn.shape
            ind_rtn = np.full(shape, fill_value=np.nan)
            for i in unique_inds:
                ind_mark = ind == i
                tmp_ret = np.full(shape, fill_value=np.nan)
                np.copyto(tmp_ret, rtn, where=ind_mark)
                i_ret = bn.nanmean(tmp_ret, axis=0)[None, :]
                np.copyto(ind_rtn, i_ret, where=ind_mark)
            self._processed_dataset['IND_RTN_EQUI'] = pd.DataFrame(ind_rtn, columns=self._dates, index=self._IDs)
        else:  # COMP_NAMES
            self._processed_dataset['COMP_NAMES'] = pd.read_csv(
                path.join(DATA_DIR, 'comp_name.txt'), dtype={'id': np.str})

        return self._processed_dataset[dataset_name]

    def __contains__(self, item):
        if item in self._dict or item in self._processed_dataset:
            return True
        return False

    def __len__(self):
        return len(self._dates)

    def __repr__(self):
        string = ''
        string += 'number of IDs: {}\n'.format(len(self._IDs))
        string += 'number of days: {}, from {} to {}\n'.format(len(self._dates), self._dates[0], self._dates[-1])
        dd = defaultdict(list)
        for k, v in self._dict.items():
            dd[v.shape].append(k)
        for k, v in dd.items():
            string += '{} raw dataset(s) of shape {}:\n'.format(len(v), k)
            string += repr(v) + '\n'
        dd = defaultdict(list)
        for k, v in self._processed_dataset.items():
            dd[v.shape].append(k)
        for k, v in dd.items():
            string += '{} processed dataset(s) of shape {}:\n'.format(len(v), k)
            string += repr(v) + '\n'
        if self._return_raw:
            string += 'return numpy array by default\n'
        else:
            string += 'return DataFrame by default\n'
        return string

    def variables(self):
        return list(self._dict.keys())

    @property
    def IDs(self):
        return self._IDs

    @property
    def dates(self):
        return self._dates

    def dates_to_indices(self, dates: Union[str, Sequence[str], np.array, pd.Index]):
        """
        convert a sequence of date strings to their corresponding indices
        """
        if isinstance(dates, str):
            return self._dates_dict[dates]
        else:
            return [self._dates_dict[d] for d in dates]

    def IDs_to_indices(self, IDs: Union[str, Sequence[str], np.array, pd.Index]):
        """
        convert a sequence of ID strings to their corresponding indices
        """
        if isinstance(IDs, str):
            return self._IDs_dict[IDs]
        else:
            return [self._IDs_dict[i] for i in IDs]

    # how datasets are loaded and preprocessed
    def _preprocess(self):
        # calculate daily return by shifting forward return
        self._dict['RTN'] = (self._dict['PRCCD'][:, 1:] + self._dict['CUM_DIV'][:, 1:] -
                             self._dict['PRCCD'][:, :-1] - self._dict['CUM_DIV'][:, :-1]) / self._dict['PRCCD'][:, :-1]
        # start from date 1

        # find trading days
        rtn = (self._dict['PRCCD'][:, 1:] - self._dict['PRCCD'][:, :-1]) / self._dict['PRCCD'][:, :-1]
        rtn[(self._dict['IN_US_1'] != 1)[:, 1:]] = np.nan
        zero_ratio = (np.round(rtn, 5) == 0).sum(axis=0) / (rtn.shape[0] - np.isnan(rtn).sum(axis=0))
        # start from date 1

        # subsetting data, keep only trading days
        dates_mark = np.argwhere(zero_ratio < 0.9).flatten()
        self._dict['RTN'] = self._dict['RTN'][:, dates_mark]
        for k, v in self._dict.items():
            if k not in ('RTN'):
                self._dict[k] = v[:, dates_mark + 1]  # start from date 1
        self._dict['FRTN1P'] = (shift_2darray(self._dict['PRCCD'] + self._dict['CUM_DIV'], by=-1, axis=1) - self._dict[
            'PRCCD'] - self._dict['CUM_DIV']) / self._dict['PRCCD']
        # start from date 1
        wealth = np.exp(np.nancumsum(np.log(1 + self._dict['RTN']), axis=1))
        wealth[np.isnan(self._dict['RTN'])] = np.nan
        self._dict['CUM_WEALTH'] = wealth

        # # update _dates and _dates_dict
        self._dates = self._dates[1:][zero_ratio < 0.9]
        self._dates_dict = {date: i for i, date in enumerate(self._dates)}
