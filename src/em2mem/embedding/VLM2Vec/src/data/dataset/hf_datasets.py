"""
Based on datasets combine.py and iterable_dataset.py
"""
from itertools import cycle
from typing import List, Optional, TypeVar

from datasets.arrow_dataset import Dataset, _interleave_map_style_datasets
from datasets.dataset_dict import DatasetDict, IterableDatasetDict
from datasets.iterable_dataset import IterableDataset, CyclingMultiSourcesExamplesIterable, \
    RandomlyCyclingMultiSourcesExamplesIterable, _BaseExamplesIterable
from datasets.utils import logging
from copy import deepcopy
from typing import Iterator, List, Optional

import numpy as np
from datasets.arrow_dataset import Dataset, DatasetInfoMixin
from datasets.features import Features
from datasets.features.features import FeatureType, _align_features, _check_if_features_can_be_aligned, cast_to_python_objects
from datasets.info import DatasetInfo
from datasets.splits import NamedSplit
from datasets.utils.py_utils import Literal

from src.utils import print_master

logger = logging.get_logger(__name__)


DatasetType = TypeVar("DatasetType", Dataset, IterableDataset)


class _HasNextIterator(Iterator):
    """Iterator with an hasnext() function. Taken from https://stackoverflow.com/questions/1966591/has-next-in-python-iterators."""

    def __init__(self, it):
        self.it = iter(it)
        self._hasnext = None

    def __iter__(self):
        return self

    def __next__(self):
        if self._hasnext:
            result = self._thenext
        else:
            result = next(self.it)
        self._hasnext = None
        return result

    def hasnext(self):
        if self._hasnext is None:
            try:
                self._thenext = next(self.it)
            except StopIteration:
                self._hasnext = False
            else:
                self._hasnext = True
        return self._hasnext


def interleave_datasets(
    datasets: List[DatasetType],
    probabilities: Optional[List[float]] = None,
    batch_size: Optional[int] = None,
    seed: Optional[int] = None,
    info: Optional[DatasetInfo] = None,
    split: Optional[NamedSplit] = None,
    stopping_strategy: Literal["first_exhausted", "all_exhausted"] = "first_exhausted",
) -> DatasetType:
    """
    Interleave several datasets (sources) into a single dataset.
    The new dataset is constructed by alternating between the sources to get the examples.

    You can use this function on a list of [`Dataset`] objects, or on a list of [`IterableDataset`] objects.

        - If `probabilities` is `None` (default) the new dataset is constructed by cycling between each source to get the examples.
        - If `probabilities` is not `None`, the new dataset is constructed by getting examples from a random source at a time according to the provided probabilities.

    The resulting dataset ends when one of the source datasets runs out of examples except when `oversampling` is `True`,
    in which case, the resulting dataset ends when all datasets have ran out of examples at least one time.

    Note for iterable datasets:

    In a distributed setup or in PyTorch DataLoader workers, the stopping strategy is applied per process.
    Therefore the "first_exhausted" strategy on an sharded iterable dataset can generate less samples in total (up to 1 missing sample per subdataset per worker).

    Args:
        datasets (`List[Dataset]` or `List[IterableDataset]`):
            List of datasets to interleave.
        probabilities (`List[float]`, *optional*, defaults to `None`):
            If specified, the new dataset is constructed by sampling
            examples from one source at a time according to these probabilities.
        batch_size (`int`, *optional*, defaults to `None`):
            If specified, the dataset is interleaved by batches instead of by examples.
        seed (`int`, *optional*, defaults to `None`):
            The random seed used to choose a source for each example.
        info ([`DatasetInfo`], *optional*):
            Dataset information, like description, citation, etc.
            <Added version="2.4.0"/>
        split ([`NamedSplit`], *optional*):
            Name of the dataset split.
            <Added version="2.4.0"/>
        stopping_strategy (`str`, defaults to `first_exhausted`):
            Two strategies are proposed right now, `first_exhausted` and `all_exhausted`.
            By default, `first_exhausted` is an undersampling strategy, i.e the dataset construction is stopped as soon as one dataset has ran out of samples.
            If the strategy is `all_exhausted`,  we use an oversampling strategy, i.e the dataset construction is stopped as soon as every samples of every dataset has been added at least once.
            Note that if the strategy is `all_exhausted`, the interleaved dataset size can get enormous:
            - with no probabilities, the resulting dataset will have `max_length_datasets*nb_dataset` samples.
            - with given probabilities, the resulting dataset will have more samples if some datasets have really low probability of visiting.
    Returns:
        [`Dataset`] or [`IterableDataset`]: Return type depends on the input `datasets`
        parameter. `Dataset` if the input is a list of `Dataset`, `IterableDataset` if the input is a list of
        `IterableDataset`.
    """
    from datasets.arrow_dataset import Dataset
    from datasets.iterable_dataset import IterableDataset

    if not datasets:
        raise ValueError("Unable to interleave an empty list of datasets.")
    for i, dataset in enumerate(datasets):
        if not isinstance(dataset, (Dataset, IterableDataset)):
            if isinstance(dataset, (DatasetDict, IterableDatasetDict)):
                if not dataset:
                    raise ValueError(
                        f"Expected a list of Dataset objects or a list of IterableDataset objects, but element at position {i} "
                        "is an empty dataset dictionary."
                    )
                raise ValueError(
                    f"Dataset at position {i} has at least one split: {list(dataset)}\n"
                    f"Please pick one to interleave with the other datasets, for example: dataset['{next(iter(dataset))}']"
                )
            raise ValueError(
                f"Expected a list of Dataset objects or a list of IterableDataset objects, but element at position {i} is a {type(dataset).__name__}."
            )
        if i == 0:
            dataset_type, other_type = (
                (Dataset, IterableDataset) if isinstance(dataset, Dataset) else (IterableDataset, Dataset)
            )
        elif not isinstance(dataset, dataset_type):
            raise ValueError(
                f"Unable to interleave a {dataset_type.__name__} (at position 0) with a {other_type.__name__} (at position {i}). Expected a list of Dataset objects or a list of IterableDataset objects."
            )
    if stopping_strategy not in ["first_exhausted", "all_exhausted"]:
        raise ValueError(f"{stopping_strategy} is not supported. Please enter a valid stopping_strategy.")
    if dataset_type is Dataset:
        return _interleave_map_style_datasets(
            datasets, probabilities, seed, info=info, split=split, stopping_strategy=stopping_strategy
        )
    else:
        return _interleave_iterable_datasets(
            datasets, probabilities, batch_size, seed, info=info, split=split, stopping_strategy=stopping_strategy
        )


def _interleave_iterable_datasets(
    datasets: List[IterableDataset],
    probabilities: Optional[List[float]] = None,
    batch_size: Optional[int] = None,
    seed: Optional[int] = None,
    info: Optional[DatasetInfo] = None,
    split: Optional[NamedSplit] = None,
    stopping_strategy: Literal["first_exhausted", "all_exhausted"] = "first_exhausted",
) -> IterableDataset:
    """
    Interleave several iterable datasets (sources) into a single iterable dataset.
    The new iterable dataset alternates between the sources to yield examples.
    If `probabilities = None` (default) the iterable dataset will cycles through the sources in order for each next example in the iteration.
    If `probabilities` is not `None, the iterable dataset will sample a random source according to the provided probabilities for each next examples in the iteration.

    <Added version="2.4.0"/>

    Args:
        datasets (`List[IterableDataset]`): list of datasets to interleave
        probabilities (`List[float]`, optional, default None): If specified, the new iterable dataset samples
            examples from one source at a time according to these probabilities.
        batch_size (`int`, optional, default None): If specified, the dataset is interleaved by batches.
        seed (`int`, optional, default None): The random seed used to choose a source for each example.
        stopping_strategy (`str`, defaults to `first_exhausted`):
            Two strategies are proposed right now.
            By default, `first_exhausted` is an undersampling strategy, i.e the dataset construction is stopped as soon as one dataset has ran out of samples.
            If the strategy is `all_exhausted`,  we use an oversampling strategy, i.e the dataset construction is stopped as soon as every samples of every dataset has been added at least once.
            Note that if the strategy is `all_exhausted`, the interleaved dataset size can get enormous:
            - with no probabilities, the resulting dataset will have max_length_datasets*nb_dataset samples.
            - with given probabilities, the resulting dataset will have more samples if some datasets have really low probability of visiting.

    Output:
        `datasets.IterableDataset`
    """
    datasets = [d._resolve_features() for d in datasets]
    print_master("=" * 50)
    print_master(f"Print the features of each dataset, make sure that all datasets have valid features.")
    for idx, d in enumerate(datasets):
        print_master(f"\t\tDataset {idx} features: {[f for f in datasets[0].features]}")
    print_master("=" * 50)

    # Perform checks
    _check_if_features_can_be_aligned([dset.features for dset in datasets])

    # TODO: improve this to account for a mix of ClassLabel and Value for example
    # right now it would keep the type of the first dataset in the list
    features = Features(
        {k: v for features in _align_features([dset.features for dset in datasets]) for k, v in features.items()}
    )

    ex_iterables = [d._ex_iterable for d in datasets]
    # Use cycling or random cycling of sources
    if batch_size is not None and batch_size > 0:
        # (interleaved-batches) if batch_size is specified, we sample data grouped by batch size, i.e. each batch (size=batch_size) is from the same dataset source
        if probabilities is None:
            ex_iterable = CyclingMultiSourcesBatchesIterable(ex_iterables, batch_size=batch_size, stopping_strategy=stopping_strategy)
        else:
            generator = np.random.default_rng(seed)
            ex_iterable = RandomlyCyclingMultiSourcesBatchesIterable(
                ex_iterables, generator=generator, probabilities=probabilities, batch_size=batch_size, stopping_strategy=stopping_strategy
            )
    else:
        #  (interleaved-instances) each example is sampled independently according to the given probabilities
        if probabilities is None:
            ex_iterable = CyclingMultiSourcesExamplesIterable(ex_iterables, stopping_strategy=stopping_strategy)
        else:
            generator = np.random.default_rng(seed)
            ex_iterable = RandomlyCyclingMultiSourcesExamplesIterable(
                ex_iterables, generator=generator, probabilities=probabilities, stopping_strategy=stopping_strategy
            )
    # Set new info - we update the features
    # setting the features also ensures to fill missing columns with None
    if info is None:
        info = DatasetInfo.from_merge([d.info for d in datasets])
    else:
        info = info.copy()
    info.features = features
    # Get all the auth tokens per repository - in case the datasets come from different private repositories
    token_per_repo_id = {
        repo_id: token for dataset in datasets for repo_id, token in dataset._token_per_repo_id.items()
    }
    # Return new daset
    return IterableDataset(ex_iterable=ex_iterable, info=info, split=split, token_per_repo_id=token_per_repo_id)


class CyclingMultiSourcesBatchesIterable(_BaseExamplesIterable):
    def __init__(
        self,
        ex_iterables: List[_BaseExamplesIterable],
        batch_size: int,
        stopping_strategy: Literal["first_exhausted", "all_exhausted"] = "all_exhausted",
    ):
        super().__init__()
        self.ex_iterables = ex_iterables
        self.batch_size =batch_size
        self.stopping_strategy = stopping_strategy

        # if undersampling ("first_exhausted"), we stop as soon as one dataset is exhausted
        # if oversampling ("all_exhausted"), we stop as soons as every dataset is exhausted, i.e as soon as every samples of every dataset has been visited at least once
        self.bool_strategy_func = np.all if (stopping_strategy == "all_exhausted") else np.any
        # TODO(QL): implement iter_arrow

    def _get_indices_iterator(self):
        # this is an infinite iterator to keep track of which iterator we want to pick examples from
        return cycle(range(len(self.ex_iterables)))

    def __iter__(self):
        iterators = [_HasNextIterator(ex_iterable) for ex_iterable in self.ex_iterables]
        indices_iterator = self._get_indices_iterator()
        is_exhausted = np.full(len(self.ex_iterables), False)
        count = 0
        for i in indices_iterator:
            count += 1
            try:  # let's pick one example from the iterator at index i
                yield next(iterators[i])
                # it will resume from the yield at the next call so that we can directly test if the iterable is exhausted and if we need to break out of the loop
                if not iterators[i].hasnext():
                    is_exhausted[i] = True
                    # print(f"dataset-{i} exhausted")
                    # print(is_exhausted)
                    if self.bool_strategy_func(is_exhausted):
                        # if the stopping criteria is met, break the main for loop
                        # print(f"all datasets exhausted")
                        break
                    # otherwise reinitialise the iterator and yield the first example
                    iterators[i] = _HasNextIterator(self.ex_iterables[i])
            except StopIteration:
                # here it means that the i-th iterabledataset is empty, i.e we never have the occasion to yield an element of the i-th dataset.
                # we still check if the stopping criteria is met and if we break out of the loop in case of an oversampling strategy
                is_exhausted[i] = True
                # print(f"StopIteration: dataset-{i} exhausted")
                # print(is_exhausted)
                if self.bool_strategy_func(is_exhausted):
                    # if the stopping criteria is met, break the main for loop
                    # print(f"StopIteration: all datasets exhausted")
                    break

    def shuffle_data_sources(self, generator: np.random.Generator) -> "CyclingMultiSourcesBatchesIterable":
        """Shuffle each underlying examples iterable."""
        ex_iterables = [ex_iterable.shuffle_data_sources(generator) for ex_iterable in self.ex_iterables]
        return CyclingMultiSourcesBatchesIterable(ex_iterables, batch_size=self.batch_size, stopping_strategy=self.stopping_strategy)

    @property
    def n_shards(self) -> int:
        return min(ex_iterable.n_shards for ex_iterable in self.ex_iterables)

    @property
    def num_shards(self) -> int:
        return min(ex_iterable.num_shards for ex_iterable in self.ex_iterables)

    def shard_data_sources(
        self, num_shards: int, index: int, contiguous=True
    ) -> "CyclingMultiSourcesBatchesIterable":
        """Either keep only the requested shard, or propagate the request to the underlying iterable."""
        return CyclingMultiSourcesBatchesIterable(
            [iterable.shard_data_sources(num_shards, index, contiguous=contiguous) for iterable in self.ex_iterables],
            stopping_strategy=self.stopping_strategy,
        )


class RandomlyCyclingMultiSourcesBatchesIterable(CyclingMultiSourcesBatchesIterable):
    def __init__(
        self,
        ex_iterables: List[_BaseExamplesIterable],
        generator: np.random.Generator,
        batch_size: int,
        probabilities: Optional[List[float]] = None,
        stopping_strategy: Literal["first_exhausted", "all_exhausted"] = "first_exhausted",
    ):
        super().__init__(ex_iterables, batch_size=batch_size, stopping_strategy=stopping_strategy)
        self.generator = deepcopy(generator)
        self.probabilities = probabilities
        self.batch_size = int(batch_size)
        # TODO(QL): implement iter_arrow

    @staticmethod
    def _iter_random_indices(
        rng: np.random.Generator,
        num_sources: int,
        batch_size: int,
        p: Optional[List[float]] = None,
    ) -> Iterator[int]:
        """Get an infinite iterator that randomly samples the index of the source to pick examples from."""
        sample_size = 100_000 * 16  # approximately (num_step*num_device). do not use batch_size as variable for computing sample_size (it can be small)
        if p is None:
            # sample uniformly if p is not given
            while True:
                # return [k for i in rng.integers(0, num_sources, size=sample_size) for k in [int(i)] * batch_size]
                yield from (k for i in rng.integers(0, num_sources, size=sample_size) for k in [int(i)] * batch_size)
        else:
            while True:
                yield from (k for i in rng.choice(num_sources, size=sample_size, p=p) for k in [int(i)] * batch_size)
                '''
                data_ids = [k for i in rng.choice(num_sources, size=sample_size, p=p) for k in [int(i)] * batch_size]
                for i in range(sample_size):  # to check the purity of each batch
                    if len(set(data_ids[batch_size * i: batch_size * (i + 1)])) != 1:
                        print("not a pure batch?")
                return data_ids
                '''

    def _get_indices_iterator(self):
        rng = deepcopy(self.generator)
        # this is an infinite iterator that randomly samples the index of the source to pick examples from
        return self._iter_random_indices(rng, len(self.ex_iterables), p=self.probabilities, batch_size=self.batch_size)

    def _init_state_dict(self) -> dict:
        self._state_dict = {
            "ex_iterable_idx": 0,
            "ex_iterables": [ex_iterable._init_state_dict() for ex_iterable in self.ex_iterables],
            "previous_states": [None] * len(self.ex_iterables),
            "is_exhausted": [False] * len(self.ex_iterables),
        }
        return self._state_dict

    def shuffle_data_sources(self, generator: np.random.Generator) -> "RandomlyCyclingMultiSourcesBatchesIterable":
        """Shuffle the data sources of each wrapped examples iterable."""
        ex_iterables = [ex_iterable.shuffle_data_sources(generator) for ex_iterable in self.ex_iterables]
        return RandomlyCyclingMultiSourcesBatchesIterable(
            ex_iterables,
            generator=generator,
            batch_size=self.batch_size,
            probabilities=self.probabilities,
            stopping_strategy=self.stopping_strategy,
        )

    def shard_data_sources(self, num_shards: int, index: int, contiguous=True) -> "RandomlyCyclingMultiSourcesBatchesIterable":
        """Either keep only the requested shard, or propagate the request to the underlying iterable."""
        return RandomlyCyclingMultiSourcesBatchesIterable(
            ex_iterables=[iterable.shard_data_sources(num_shards, index, contiguous=contiguous) for iterable in self.ex_iterables],
            generator=self.generator,
            batch_size=self.batch_size,
            probabilities=self.probabilities,
            stopping_strategy=self.stopping_strategy,
        )
