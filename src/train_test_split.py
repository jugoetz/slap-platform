from itertools import chain
from typing import Any

import numpy as np
from sklearn.model_selection import ShuffleSplit, LeaveOneGroupOut, GroupKFold
from sklearn.model_selection._split import _validate_shuffle_split
from sklearn.utils import _safe_indexing
from sklearn.utils.validation import check_array, indexable, _num_samples


class GroupShuffleSplit2D(ShuffleSplit):
    """
    Extension of scikit-learn's GroupShuffleSplit to apply to 2-dimensional groups.

    Splits data which is grouped along two dimensions (e.g. data from two-component chemical reactions) so that the
    train and test set groups do not overlap on any dimension. Note that this typically means that some data will not
    be used, neither in the train, nor test set.
    This implementation cannot give exact train-test-ratios as the outcome depends on the data (in particular on the
    number of members for each group).
    """

    def __init__(
        self, n_splits=5, *, test_size=None, train_size=None, random_state=None
    ):
        super().__init__(
            n_splits=n_splits,
            test_size=test_size,
            train_size=train_size,
            random_state=random_state,
        )
        self._default_test_size = 0.05  # we set this lower than the scikit-learn default because the true test size
        # will be bigger than the target

    def _iter_indices(self, X, y, groups):
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None.")
        groups = check_array(groups, ensure_2d=True, dtype=None)
        classes, group_indices = [], []
        n_groups = groups.shape[1]
        for i in range(n_groups):
            classes_i, group_indices_i = np.unique(groups[:, i], return_inverse=True)
            classes.append(classes_i)
            group_indices.append(group_indices_i)
        for ind_train, ind_test in super()._iter_indices(X=X):
            # so far this is a standard ShuffleSplit
            # we identify the groups that landed in the test set
            test_groups = [group_ind[ind_test] for group_ind in group_indices]
            # the train set is equal to all samples that are in none of the test groups
            train = np.flatnonzero(
                ~np.in1d(group_indices[0], test_groups[0])
                & ~np.in1d(group_indices[1], test_groups[1])
            )
            # the test set is equal to all samples that are in both test groups.
            test = np.flatnonzero(
                np.in1d(group_indices[0], test_groups[0])
                & np.in1d(group_indices[1], test_groups[1])
            )
            # print(f"2D splitter yielding a fold with train-test-sizes {len(train)} - {len(test)}")
            if len(train) == 0:
                raise RuntimeError(
                    "No training samples left after removing overlapping samples from the test set. Consider lowering test_size."
                )
            yield train, test

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : array-like of shape (n_samples,), default=None
            The target variable for supervised learning problems.

        groups : array-like of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.

        Notes
        -----
        Randomized CV splitters may return different results for each call of
        split. You can make the results identical by setting `random_state`
        to an integer.
        """
        return super().split(X, y, groups)


class LeaveOneGroupOut2D(LeaveOneGroupOut):
    """Extension of scikit-learn's LeaveOneGroupOut to apply to N-dimensional groups. (where N >= 2)

    If one applied scikit-learn's LeaveOneGroupOut splitter to a dataset with multiple dimensions
    (say chemical reaction data), then the splitter would return a test set with just one group
    (in many cases just on reaction instance). The train set, however, would contain entries that overlap with the test
    set in at least one of those N dimensions.
    In contrast, the splitter implemented here removes all entries that overlap with the test set on any dimension.
    """

    def _iter_test_masks(self, X=None, y=None, groups=None):
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None.")
        # We make a copy of groups to avoid side effects during iteration
        groups = check_array(groups, copy=True, ensure_2d=True, dtype=None)
        # we cannot use np.unique() on groups b/c using the axis keyword includes a sorting operation which can destroy
        # the structure of our array (along axis 1). Instead we use set() (it changes the order along axis 0 but that
        # is not a problem for LOGOCV).
        unique_groups = set([(i, j) for i, j in groups])
        if len(unique_groups) <= 1:
            raise ValueError(
                "The groups parameter contains fewer than 2 unique groups "
                "(%s). LeaveOneGroupOut expects at least 2." % unique_groups
            )
        for i in unique_groups:
            test_mask = groups == i
            yield test_mask

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator

        Parameters
        ----------
        X : object
            Always ignored, exists for compatibility.

        y : object
            Always ignored, exists for compatibility.

        groups : array-like of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set. This 'groups' parameter must always be specified to
            calculate the number of splits, though the other parameters can be
            omitted.

        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None.")
        groups = check_array(groups, ensure_2d=True, dtype=None)
        return len(np.unique(groups, axis=1))

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            The target variable for supervised learning problems.

        groups : array-like of shape (n_samples,), default=None
            Group labels for the samples used while splitting the dataset into
            train/test set.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.
        """
        X, y, groups = indexable(X, y, groups)
        indices = np.arange(_num_samples(X))
        for test_index in self._iter_test_masks(X, y, groups):
            train_index = indices[~np.any(test_index, axis=1)]
            test_index = indices[np.all(test_index, axis=1)]
            if train_index.shape[0] == 0 or test_index.shape[0] == 0:
                raise RuntimeError(
                    "Internal error: Training and/or testing set for this fold is empty"
                )
            yield train_index, test_index


class Leaky2DKFold:
    """
    We start from data which is dependent in two dimensions (e.g. two-component reaction data). And for one of these
    two dimensions (we call this the "active" dimension), we want to test the quality of and estimator depending on
    the number of 1D-dependent samples it has seen, all while keeping the test data independent on the second
    dimension (we call this the "passive" dimension)

    For example, let's look at a reaction between an alcohol and a carboxylic acid to form an ester:
        alcohol (A) + acid (CA) --> ester (E)
    Let's further say we are interested in the question "how many reactions of a specific carboxylic acid CA_1 do I have
    to see, in order to extrapolate to a set of unseen alcohols A_x?"
    According to this question, CA will be our active dimension and A our passive dimension. We first conduct a
    GroupKFold split where the passive dimension is used for group membership. We receive K folds split by alcohol
    identity. In a second step, we remove CA_1 from all training sets, up to a remaining number given by the parameter
    n_leaks. n_leaks thus allows to control the number of CA_1 samples seen during training.
    For decent statistics, a proper implementation should repeat this split for multiple carboxylic acids, that should
    fulfill n_records(CA_x) >> K
    """

    def __init__(
        self,
        n_splits: int,
        n_leaks: int,
        active_dim_idx: int,
        passive_dim_idx: int,
        active_dim_group: Any,
        random_seed: int = None,
    ):
        """
        Args:
            n_splits (int): number of folds, must be >= 2.
            n_leaks (int): number of samples that will be in every training set despite 1D overlap
            active_dim_idx (int): Index for accessing the active dimension in groups array when calling split()
            passive_dim_idx (int): Index for accessing the passive dimension in groups array when calling split()
            active_dim_group (Any): Value of the active group that should be used for the test set and removed from
                the training set up to n_leaky examples
            random_seed (int): Seed for reproducibility. Only random element in this splitter is which of the samples
                will be leaked if n_leaks > 0.
        """
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        self.n_splits = n_splits
        self.n_leaks = n_leaks
        self.active_dim_idx = active_dim_idx
        self.passive_dim_idx = passive_dim_idx
        self.active_dim_group = active_dim_group
        self.random_seed = random_seed

    def split(self, X, y=None, groups=None):
        """

        Args:
            X (np.ndarray): SLAPData to be split. 2D-array of shape (n_samples, n_groups) where n_groups = 2
            y: Irrelevant. Exists for sklearn compatibility.
            groups (np.ndarray): Groups to use for splitting

        Returns:

        """

        if groups is None:
            raise ValueError("groups cannot be None")
        active_groups = groups[:, self.active_dim_idx]
        passive_groups = groups[:, self.passive_dim_idx]

        # first we use sklearn's GroupKFold to split along the passive dimension
        aux_splitter = GroupKFold(n_splits=self.n_splits)
        aux_folds = aux_splitter.split(X, groups=passive_groups)

        # now we clean the aux_folds
        # remove all occurrences of active dimension groups that do not have the value active_dim_group
        active_samples_idx = np.flatnonzero((active_groups == self.active_dim_group))
        if len(active_samples_idx) == 0:
            raise ValueError(
                "Found no samples in active dimension that have the value of argument 'active_dim_group' "
                "(given when instantiating splitter)"
            )
        for aux_train, aux_test in aux_folds:
            # from test set remove all occurrences of active dimension groups that do not have the value
            # active_dim_group
            test = np.intersect1d(aux_test, active_samples_idx)
            if len(test) == 0:
                raise ValueError("Test set is empty.")
            # from train set, remove all occurrences of active dimension groups that have the value active_dim_group,
            # except for a number of random samples specified by n_leaks
            potential_active_leaks = aux_train[np.isin(aux_train, active_samples_idx)]
            rng = np.random.default_rng(self.random_seed)
            rng.shuffle(
                potential_active_leaks
            )  # we shuffle which samples will be leaked into the training set
            if len(potential_active_leaks) < self.n_leaks:
                raise ValueError(
                    f"Number of relevant records ({len(potential_active_leaks)}) available for training is smaller than n_leaks ({self.n_leaks})"
                )
            train = aux_train[
                ~np.isin(aux_train, potential_active_leaks[self.n_leaks :])
            ]

            yield train, test

    def get_n_splits(self):
        """Returns the number of splitting iterations in the cross-validator"""
        return self.n_splits


def group_train_test_split(
    *arrays,
    test_size=None,
    train_size=None,
    random_state=None,
    shuffle=True,
    groups=None,
):
    """Split arrays or matrices into random train and test subsets

    Quick utility that wraps input validation and
    ``next(ShuffleSplit().split(X, y))`` and application to input data
    into a single call for splitting (and optionally subsampling) data in a
    oneliner.

    (Modified from scikit-learn's train_test_split)

    Parameters
    ----------
    *arrays : sequence of indexables with same length / shape[0]
        Allowed inputs are lists, numpy arrays, scipy-sparse
        matrices or pandas dataframes.

    test_size : float or int, default=None
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the test split. If int, represents the
        absolute number of test samples. If None, the value is set to the
        complement of the train size. If ``train_size`` is also None, it will
        be set to 0.25.

    train_size : float or int, default=None
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the train split. If
        int, represents the absolute number of train samples. If None,
        the value is automatically set to the complement of the test size.

    random_state : int, RandomState instance or None, default=None
        Controls the shuffling applied to the data before applying the split.
        Pass an int for reproducible output across multiple function calls.


    shuffle : bool, default=True
        Whether or not to shuffle the data before splitting. If shuffle=False
        then stratify must be None.

    groups : array-like, default=None
        If not None, data is split in a grouped fashion, using this as
        the class labels. This must be of shape (n_samples, x) where x >= 2.
        For x = 1, use scikit-learn's GroupShuffleSplit

    Returns
    -------
    splitting : list, length=2 * len(arrays)
        List containing train-test split of inputs.
        If the input is sparse, the output will be a
        ``scipy.sparse.csr_matrix``. Else, output type is the same as the
        input type.

    """
    n_arrays = len(arrays)
    if n_arrays == 0:
        raise ValueError("At least one array required as input")

    arrays = indexable(*arrays)

    n_samples = _num_samples(arrays[0])
    n_train, n_test = _validate_shuffle_split(
        n_samples, test_size, train_size, default_test_size=0.25
    )

    if shuffle is False:
        raise ValueError("Group train/test split is not implemented for shuffle=False")

    else:
        CVClass = GroupShuffleSplit2D

        cv = CVClass(test_size=n_test, train_size=n_train, random_state=random_state)

        train, test = next(cv.split(X=arrays[0], y=None, groups=groups))

    return list(
        chain.from_iterable(
            (_safe_indexing(a, train), _safe_indexing(a, test)) for a in arrays
        )
    )


# (this is from sklearn:)
# Tell nose that group_train_test_split is not a test.
# (Needed for external libraries that may use nose.)
# Use setattr to avoid mypy errors when monkeypatching.
setattr(group_train_test_split, "__test__", False)
