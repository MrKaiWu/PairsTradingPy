from joblib import Parallel
from tqdm import tqdm


# %%
class ProgressParallel(Parallel):
    """
    A child of the joblib Parallel class that integrates the progress bar from tqdm
    Usage of Parallel: https://joblib.readthedocs.io/en/latest/parallel.html#embarrassingly-parallel-for-loops
    """

    def __init__(self, use_tqdm=True, total=None, *args, **kwargs):
        """
        :param use_tqdm: whether to use tqdm progress bar
        :param total: the total number of tasks to be distributed. input an exact number for the progress bar to behave well
        """
        self._use_tqdm = use_tqdm
        self._total = total
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm(disable=not self._use_tqdm, total=self._total) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()
