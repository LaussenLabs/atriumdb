import numpy as np

from atriumdb.windowing.dataset_iterator import DatasetIterator
from atriumdb.windowing.window import Window


class RandomAccessDatasetIterator(DatasetIterator):
    """
    Subclass of DatasetIterator that allows random access to windowed segments of a dataset.

    This class extends the capabilities of DatasetIterator by implementing __len__ and __getitem__,
    enabling direct indexing and length querying. It's particularly useful in scenarios where random
    access to the data is required, such as in machine learning models where shuffling or specific sampling
    strategies are applied.

    Inherits from DatasetIterator and maintains all its functionalities, including iterative access to
    sliding windows of data from different sources. Adds the ability to access specific windows directly
    by their index and to determine the total number of windows available in the dataset.

    :param sdk: SDK object to fetch data
    :type sdk: AtriumSDK
    :param validated_measure_list: List of validated measures with information about each measure
    :type validated_measure_list: list
    :param validated_sources: Dictionary containing sources with associated time ranges
    :type validated_sources: dict
    :param window_duration_ns: Duration of each window in nanoseconds
    :type window_duration_ns: int
    :param window_slide_ns: Interval in nanoseconds by which the window advances in time
    :type window_slide_ns: int
    :param num_windows_prefetch: Number of windows you want to get from AtriumDB at a time. Setting this value
            higher will make decompression faster but at the expense of using more RAM. (default the number of windows
            that gets you closest to 10 million values).
    :type num_windows_prefetch: int, optional
    :param label_threshold: Threshold for labeling in classification tasks.
    :type label_threshold: float
    :param time_units: If you would like the window_duration and window_slide to be specified in units other than
                            nanoseconds you can choose from one of ["s", "ms", "us", "ns"].
    :type time_units: str
    """

    def __init__(self, sdk, validated_measure_list, validated_label_set_list, validated_sources,
                 window_duration_ns: int, window_slide_ns: int, num_windows_prefetch: int = None,
                 label_threshold=0.5, time_units=None, shuffle=False, max_cache_duration=None,
                 patient_history_fields: list = None):
        super().__init__(sdk, validated_measure_list, validated_label_set_list, validated_sources,
                         window_duration_ns, window_slide_ns, num_windows_prefetch, label_threshold, time_units,
                         shuffle, max_cache_duration, patient_history_fields)

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
