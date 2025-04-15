import numpy as np

from atriumdb.windowing.dataset_iterator import DatasetIterator
from atriumdb.windowing.window import Window


class MappedIterator(DatasetIterator):
    """
    Subclass of DatasetIterator that allows random access to windowed segments of a dataset.

    This class extends the capabilities of DatasetIterator by implementing __getitem__,
    enabling direct indexing. It still uses the same sequence optimization as the default iterator
    therefore it is best used when sequentially accessing windowed segments.

    If you need random/shuffled access to windows we recommend using the `LightMappedIterator`.

    :param AtriumSDK sdk: SDK object to fetch data
    :param list validated_measure_list: List of validated measures with information about each measure
    :param dict validated_sources: Dictionary containing sources with associated time ranges
    :param int window_duration_ns: Duration of each window in nanoseconds
    :param int window_slide_ns: Interval in nanoseconds by which the window advances in time
    :param int num_windows_prefetch: Number of windows you want to get from AtriumDB at a time. Setting this value
            higher will make decompression faster but at the expense of using more RAM. (default the number of windows
            that gets you closest to 10 million values).
    :param float label_threshold: Threshold for labeling in classification tasks.
    """

    def __init__(self, sdk, definition, window_duration_ns: int, window_slide_ns: int, num_windows_prefetch: int = None,
                 label_threshold=0.5, shuffle=False, max_cache_duration=None, patient_history_fields: list = None,
                 label_exact_match=False):
        super().__init__(sdk=sdk, definition=definition, window_duration_ns=window_duration_ns,
                         window_slide_ns=window_slide_ns, num_windows_prefetch=num_windows_prefetch,
                         label_threshold=label_threshold, shuffle=shuffle, max_cache_duration=max_cache_duration,
                         patient_history_fields=patient_history_fields, label_exact_match=label_exact_match)

        pass

    def __len__(self) -> int:
        """
        Get the total number of windows in the dataset.

        :return: Total number of windows in the dataset
        :rtype: int
        """
        return self._length

    def __getitem__(self, idx: int) -> Window:
        """
        Fetches a Window object corresponding to the given index, encapsulating multiple signals of varying
        frequencies along with their associated metadata. The Window object returned will have its `signals` attribute
        populated with a dictionary, where each key is a tuple describing the measure tag, the frequency of the
        measure in Hz, and the units of the measure. The value corresponding to each key is another dictionary
        containing the actual data points, expected count, actual count of non-NaN data points, measure ID, and
        the timestamps associated with each data point of the signal.

        :param idx: The index of the desired window. This index must be within the bounds of the available data
            windows. Negative indexing is supported, `-idx = len(iterator) - idx`.
        :type idx: int
        :return: A Window object encapsulating the signals and associated metadata for the specified index. The
            structure of the Window object and the included signals dictionary is described in the Window Format section
            of the documentation.
        :rtype: Window
        :raises IndexError: Raised if the index is out of bounds.

        :Example:

        .. code-block:: python

            window_obj = dataset_iterator[5]
            signals_dict = window_obj.signals

            for measure_info, signal_data in signals_dict.items():
                print(f"Measure Info: {measure_info}")
                print(f"Times: {signal_data['times']}")
                print(f"Values: {signal_data['values']}")
                print(f"Expected Count: {signal_data['expected_count']}")
                print(f"Actual Count: {signal_data['actual_count']}")
                print(f"Measure ID: {signal_data['measure_id']}")

        """
        if idx < 0:
            idx = self._length - idx

        if idx < 0 or self._length <= idx:
            raise IndexError(f"Index {idx} out of bounds for iterator of length {self._length}")

        if idx < self.current_batch_start_index or self.current_batch_end_index <= idx:
            self._load_batch_matrix(idx)

        self.current_index = idx

        return self.window_cache[idx - self.current_batch_start_index]

    def get_array_matrix(self, idx: int) -> np.ndarray:
        """
        Fetch the window data (numpy matrix) for a given index. By nature of being a matrix, the returned array will
        have equal numbers of values for each row (signal), and therefore gaps are filled with numpy.nan values in
        rows with lower sample frequencies.

        :param idx: Index of the desired window
        :type idx: int
        :return: Array of data corresponding to the given index
        :rtype: np.ndarray
        :raises IndexError: If the index is out of bounds
        """
        if idx < 0:
            idx = self._length - idx

        if idx < 0 or self._length <= idx:
            raise IndexError(f"Index {idx} out of bounds for iterator of length {self._length}")

        if idx < self.current_batch_start_index or self.current_batch_end_index <= idx:
            self._load_batch_matrix(idx)

        self.current_index = idx

        return self.matrix_cache[idx - self.current_batch_start_index]
