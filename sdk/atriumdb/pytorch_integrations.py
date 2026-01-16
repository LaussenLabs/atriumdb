import logging
from atriumdb import AtriumSDK, DatasetDefinition

_LOGGER = logging.getLogger(__name__)

try:
    from torch.utils.data import Dataset
except ImportError:
    _LOGGER.error("Error Pytorch not installed. To use pytorch integrations please install pytorch.")


class AtriumDBMapDataset(Dataset):
    """
    A PyTorch-compatible dataset wrapper for AtriumDB that provides index mapped
    windows of time-series data for machine learning workflows.

    This class uses AtriumSDK to load and iterate over data defined by a
    DatasetDefinition. It supports optional in-memory caching for faster access
    and subclasses Pytorch's Dataset class to make it compatible with PyTorch's DataLoader.

    It is recommended you subclass this class and add your preprocessing code for data and labels to the __getitem__()
    function as well as a collator function to control how batching is done (see pytorch docs).


    :param Union[str, PurePath] dataset_location: A file path or a path-like object that points to the directory in which the AtriumDB dataset is located.
    :param filename: (str) Path to the YAML or pickle file containing the dataset definition. If the file extension is `.pkl`,
    the validated dataset definition is loaded (fast). If a YAML file is provided, the contents of the file will be validated (slow for repeat usage).
    :param window_duration (int): Duration of each window in `time_units`.
    :param window_slide (int): Step size for sliding windows in `time_units`.
    :param time_units (str): Units for time-based windowing ["s", "ms", "us", "ns"].
    :param memcache_metadata (bool): Whether to cache dataset metadata in memory for speed (but uses more RAM).
    :param gap_tolerance (int): Maximum allowed gap in data continuity.
    :param num_threads (int): Number of threads for data decompression. WARNING: AtriumDB will use OpenMP to deal with threading.
    It will set the OMP_NUM_THREADS environment variable to the value you specify. Pytorch and related libraries sometimes
    also use this variable which would be overwritten and could cause slower performance in other libraries if set to a low number.


    Example:
        >>> dataset = AtriumDBMapDataset(
        ...     dataset_location="/path/to/data",
        ...     dataset_definition_path="/path/to/definition.yaml",
        ...     window_duration=10,
        ...     window_slide=5,
        ...     time_units="s",
        ...     memcache_metadata=True,
        ...     gap_tolerance=0,
        ...     num_threads=2
        ... )
        >>> # get the number of samples in the dataset
        >>> len(dataset)
        >>> 1000
        >>> # get a window from the dataset
        >>> window = dataset[0]
        >>>
        >>> # how to use the dataset with a pytorch dataloader
        >>> from torch.utils.data import DataLoader
        >>>
        >>> # Wrap the dataset in a DataLoader
        >>> dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=3)
        >>>
        >>> # Iterate over batches
        >>> for batch_idx, batch in enumerate(dataloader):
        >>>    # Each batch will contain 32 windows from AtriumDB
        >>>    print(f"Batch {batch_idx}: {batch}")

    """

    def __init__(self, dataset_location, dataset_definition_path, window_duration, window_slide, memcache_metadata,
                 time_units="s", gap_tolerance=0, num_threads=2):

        super(AtriumDBMapDataset).__init__()

        # define the sdk object you will be using
        self.sdk = AtriumSDK(dataset_location=dataset_location, num_threads=num_threads)

        # get your dataset definition
        self.dataset_def = DatasetDefinition(filename=dataset_definition_path)

        # if you want to cache your dataset in memory for speed do that here
        if memcache_metadata:
            self.sdk.load_definition(self.dataset_def, gap_tolerance=gap_tolerance)

        # get your iterator over the dataset from the atriumdb
        self.iterator_base_instance = self.sdk.get_iterator(self.dataset_def, window_duration=window_duration, window_slide=window_slide,
                                                            time_units=time_units, iterator_type="lightmapped", shuffle=False, gap_tolerance=gap_tolerance)

        _LOGGER.info(f"Number of samples: {self.__len__()}")

    # get the number of samples in the iterator
    def __len__(self):
        return self.iterator_base_instance.__len__()

    # iterate over the dataset in order
    def __iter__(self):
        return self.iterator_base_instance.__iter__()

    # get a window from a specific index
    def __getitem__(self, index):
        return self.iterator_base_instance.__getitem__(index)