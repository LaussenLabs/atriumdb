# AtriumDB is a timeseries database software designed to best handle the unique features and
# challenges that arise from clinical waveform data.
#     Copyright (C) 2023  The Hospital for Sick Children
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.

import yaml
import os
import pickle
import warnings
import json
import numpy as np

from atriumdb.adb_functions import time_unit_options
from atriumdb.windowing.definition_builder import build_source_intervals
from atriumdb.windowing.verify_definition import verify_definition
from atriumdb.windowing.windowing_functions import get_signal_dictionary, get_label_dictionary, get_window_list, \
    _load_patient_cache


class DatasetDefinition:
    """
    Encapsulates the Dataset Definition in YAML or binary format.

    The ``DatasetDefinition`` class represents the definition of dataset sources
    (devices or patients), their associated time regions, and the available signals
    over those time regions for the respective sources, along with any labels
    associated with these sources. It also supports validation and filtering of the dataset
    definition, which can then be saved in a binary format using pickle so you don't have to repeat the validation
    and/or filtering steps for repeated runs.

    :ivar data_dict: A dictionary containing the structured data from the YAML file
                     or passed parameters. The keys include 'measures', 'patient_ids',
                     'mrns', 'device_ids', 'device_tags', and 'labels'.
    :ivar validated_data_dict: A dictionary containing validated data after a successful validation process.
                               Populated only after calling the `validate` method.
    :ivar is_validated: A boolean flag indicating whether the dataset has been successfully validated.

    :param filename: (str, optional) Path to the YAML or pickle file containing the dataset definition. If
                     provided, the contents of the file will populate `data_dict`. If the file extension is
                     `.pkl`, the validated dataset definition is loaded instead.
    :param measures: (list, optional) List of requested measures to be considered. Measures can be:
                     1. Just a tag (string) representing the measure.
                     2. A dictionary specifying the measure tag, its frequency in Hertz, and its units.
                     Example: [{"tag": "ECG", "freq_hz": 300, "units": "mV"}]
    :param patient_ids: (dict, optional) Dictionary containing patient identifiers with their associated
                        time specifications. The specifications can be interval-based, event-based,
                        or the keyword 'all' to indicate all available time data for the given patient.
                        Example: {1234567: "all", 7654321: [{'start': 123456000 * 10**9, 'end': 123457000 * 10**9}]}
    :param mrns: (dict, optional) Dictionary containing medical record numbers. Each MRN can have associated
                 time specifications (interval-based, event-based) or 'all' indicating all available
                 time data for the given MRN.
                 Example: {1234567: "all", 7654321: [{'start': 123456000 * 10**9, 'end': 123457000 * 10**9}]}
    :param device_ids: (dict, optional) Dictionary containing device identifiers, each associated with its time
                       specifications. The specifications can be interval, event, or 'all' for all
                       available time data.
                       Example: {1: "all", 14: [{'time0': 123456000 * 10**9, 'pre': 1000 * 10**9, 'post': 3000 * 10**9}]}
    :param device_tags: (dict, optional) Dictionary containing device tags. Each tag is mapped to its respective
                        time specifications, which can be interval-based, event-based, or simply 'all'.
                        Example: {"dev_1": "all", "dev_2b": [{'start': 123456000 * 10**9}]}
    :param labels: (list, optional) List of strings, each representing a unique label associated with the dataset elements
                   (such as patient conditions or device states). These labels can be used for classification,
                   segmentation, or other forms of analysis.
                   Example: ["Sinus rhythm", "Junctional ectopic tachycardia (JET)", "arrhythmia"]

    .. note:: For more details on the format expectations, see :ref:`definition_file_format`.

    **Examples**:

    **Reading from an existing YAML file**:

    >>> dataset_definition = DatasetDefinition(filename="/path/to/my_definition.yaml")

    **Reading from an existing validated pickle file**:

    >>> dataset_definition = DatasetDefinition(filename="/path/to/my_validated_definition.pkl")

    **Creating an empty definition**:

    >>> dataset_definition = DatasetDefinition()

    **Creating a definition with measures and labels**:

    >>> measures = ["measure_tag_1", {"tag": "measure_tag_2", "freq_hz": 62.5, "units": "measure_units_2"}]
    >>> labels = ["label1", "label2"]
    >>> dataset_definition = DatasetDefinition(measures=measures, labels=labels)

    **Adding a label to the definition**:

    >>> dataset_definition.add_label("new_label")

    **Validating a definition**:

    >>> dataset_definition.validate(sdk=my_sdk)
    >>> print(dataset_definition.is_validated)
    ... True

    **Saving the definition to a YAML file**:

    >>> dataset_definition.save(filepath="path/to/saved/definition.yaml")

    **Saving a validated definition to a pickle file**:

    >>> dataset_definition.save(filepath="path/to/saved/definition.pkl")

    **Filtering a validation definition**:

    >>> def my_filter_fn(window):
    ...     # Require more than 5 values from example_measure
    ...     return window.signals[("example_measure", 1.0, "units")]['actual_count'] > 5
    >>> dataset_definition.filter(sdk=my_sdk, filter_fn=my_filter_fn, window_duration=1_000_000_000, window_slide=1_000_000_000)

    **Saving a filtered definition**:

    >>> dataset_definition.save(filepath="path/to/saved/filtered_definition.pkl")
    """

    def __init__(self, filename=None, measures=None, patient_ids=None, mrns=None, device_ids=None,
                 device_tags=None, labels=None):
        self.data_dict = {
            'measures': measures if measures is not None else [],
            'patient_ids': patient_ids if patient_ids is not None else {},
            'mrns': mrns if mrns is not None else {},
            'device_ids': device_ids if device_ids is not None else {},
            'device_tags': device_tags if device_tags is not None else {},
            'labels': labels if labels is not None else [],
        }
        # Check format and convert data as before
        self._check_format_and_convert_data()

        self.validated_data_dict = {}
        self.is_validated = False
        self.validated_gap_tolerance = None
        self.filtered_window_size = None
        self.filtered_window_slide = None

        if filename:
            _, ext = os.path.splitext(filename)
            if ext == '.pkl':
                # Load from pickle
                self._load_validated_pickle(filename)
            elif ext in ['.yaml', '.yml']:
                # Assume YAML
                self.read_yaml(filename)
                self._check_format_and_convert_data()
            else:
                raise ValueError("Unsupported file extension '{}', must be .pkl, .yaml or .yml".format(ext))



    @classmethod
    def build_from_intervals(cls, sdk, build_from_signal_type, measures=None, labels=None, build_labels=None,
                             patient_id_list=None, mrn_list=None, device_id_list=None, device_tag_list=None,
                             start_time=None, end_time=None, gap_tolerance=None, merge_strategy=None):
        """
        Class method that builds a DatasetDefinition object using signal-based intervals.

        :param sdk: Data SDK used to interact with the database or data service
        :param build_from_signal_type: Signal type to build from, either "measures" or "labels"
        :param measures: List of measures to build from, if applicable
        :param labels: List of labels for the definition (used to build by labels, if build_labels is not provided).
        :param build_labels: Optional alternative list of labels to use for building intervals when build_from_signal_type is "labels".
        :param patient_id_list: List of patient IDs
        :param mrn_list: List of medical record numbers
        :param device_id_list: List of device IDs
        :param device_tag_list: List of device tags
        :param start_time: Start timestamp for filtering
        :param end_time: End timestamp for filtering
        :param int gap_tolerance: The maximum allowable gap size in the data such that the output considers a
            region continuous. Put another way, the minimum gap size, such that the output of this method will add
            a new row.
        :param merge_strategy: Strategy to merge intervals. 'union' (default) for returning all intervals with at
            least one specified measure or label, 'intersection' for returning all intervals with every specified
            measure or label.
        :return: DatasetDefinition object
        """
        # Validate build_from_signal_type
        if build_from_signal_type not in ["measures", "labels"]:
            raise ValueError("build_from_signal_type must be either 'measures' or 'labels'")

        # Determine which signals to build from based on build_from_signal_type
        if build_from_signal_type == "measures":
            assert measures, "If you are building on measures, you must provide measures"
            build_measures = measures
            build_labels_value = None
        else:  # "labels"
            # Prefer build_labels if provided; otherwise, fall back to labels
            if build_labels is not None:
                build_labels_value = build_labels
            else:
                assert labels, "If you are building on labels, you must provide labels"
                build_labels_value = labels
            build_measures = None

        # Build the source intervals using the build_source_intervals function
        source_intervals = build_source_intervals(
            sdk,
            measures=build_measures,
            labels=build_labels_value,
            patient_id_list=patient_id_list,
            mrn_list=mrn_list,
            device_id_list=device_id_list,
            device_tag_list=device_tag_list,
            start_time=start_time,
            end_time=end_time,
            gap_tolerance=gap_tolerance,
            merge_strategy=merge_strategy
        )

        # Create a DatasetDefinition instance
        kwargs = {
            'measures': measures,
            'labels': labels,
            'patient_ids': source_intervals.get('patient_ids'),
            'mrns': source_intervals.get('mrns'),
            'device_ids': source_intervals.get('device_ids'),
            'device_tags': source_intervals.get('device_tags'),
        }
        dataset_def = cls(**kwargs)
        return dataset_def

    def validate(self, sdk, gap_tolerance=None, measure_tag_match_rule="best", start_time=None, end_time=None,
                 time_units: str = "ns"):
        """
        Verifies and validates a dataset definition against the given SDK, ensuring the data specified actually exists.

        :param sdk: SDK object to validate the definition against.
        :param gap_tolerance: (int, optional) Minimum allowed gap size in nanoseconds for continuous time ranges.
        :param measure_tag_match_rule: (str, optional) "best" or "all" to determine matching strategy for measure tags.
        :param start_time: (int, optional) Global start time in the specified `time_units`.
        :param end_time: (int, optional) Global end time in the specified `time_units`.
        :param time_units: (str, optional) Units for interpreting time parameters. Defaults to "ns".
        :raises ValueError: If validation fails or input parameters are invalid.

        **Examples**:

        >>> my_sdk = AtriumSDK(dataset_location)
        >>> dataset_definition.validate(sdk=my_sdk, gap_tolerance=100, start_time=1735845426, end_time=1737236445, time_units="s")
        """
        start_time_n = None if start_time is None else int(start_time * time_unit_options[time_units])
        end_time_n = None if end_time is None else int(end_time * time_unit_options[time_units])
        gap_tolerance_n = None if gap_tolerance is None else int(gap_tolerance * time_unit_options[time_units])

        validated_measure_list, validated_label_set_list, mapped_sources = verify_definition(
            self, sdk=sdk, gap_tolerance=gap_tolerance_n, measure_tag_match_rule=measure_tag_match_rule,
            start_time_n=start_time_n, end_time_n=end_time_n)

        self.validated_data_dict = {
            "measures": validated_measure_list,
            "labels": validated_label_set_list,
            "sources": mapped_sources,
        }

        self.is_validated = True
        self.validated_gap_tolerance = gap_tolerance_n

    def filter(self, sdk, filter_fn, window_duration=None, window_slide=None, time_units='ns',
               allow_partial_windows=True, label_threshold=0.5, patient_history_fields=None):
        """
        Filters the dataset definition using a custom filter function.

        Your custom filter function must take a window object like those passed by AtriumSDK.get_iterator
        It should return True to accept the Window, and False to reject the Window.

        :param sdk: SDK object to retrieve and process data for filtering.
        :param filter_fn: Callable to filter dataset windows. Should return True for accepted windows.
        :param window_duration: (int, optional) Duration of each window in specified `time_units`.
        :param window_slide: (int, optional) Sliding interval for the windows in specified `time_units`.
        :param time_units: (str, optional) Units for window size and slide. One of "ns", "us", "ms", or "s".
        :param allow_partial_windows: (bool, optional) Whether to include partially filled windows. Defaults to True.
        :param label_threshold: (float, optional) Minimum label coverage threshold for inclusion. Defaults to 0.5.
        :param patient_history_fields: (list, optional) Additional fields from patient history to include in the window object.
        :raises ValueError: If the definition is not validated or parameters are invalid.

        **Examples**:

        >>> my_sdk = AtriumSDK(dataset_location)
        >>> def my_filter_fn(window):
        ...     # Require more than 5 values from example_measure
        ...     return window.signals[("example_measure", 1.0, "units")]['actual_count'] > 5
        >>> dataset_definition.filter(sdk=my_sdk, filter_fn=my_filter_fn, window_duration=1_000_000_000, window_slide=1_000_000_000)
        """
        if not self.is_validated:
            raise ValueError("Definition must be validated before filtering, run DatasetDefinition.validate()")
        time_units = "ns" if time_units is None else time_units
        if time_units not in time_unit_options.keys():
            raise ValueError("Invalid time units. Expected one of: %s" % time_unit_options)

        if window_duration is None or window_slide is None:
            raise ValueError("Both window duration and window slide must be specified.")

        # convert to nanoseconds
        window_duration = int(window_duration * time_unit_options[time_units])
        window_slide = int(window_slide * time_unit_options[time_units])

        if self.filtered_window_size is not None and self.filtered_window_size != window_duration:
            warnings.warn(f"Definition has already been filtered with different window duration "
                          f"{self.filtered_window_size} ns. Refiltering will alter the window positions.")

        # Warn if the new window slide differs from a previously set one
        if self.filtered_window_slide is not None and self.filtered_window_slide != window_slide:
            warnings.warn(
                f"Definition has already been filtered with a different window slide "
                f"({self.filtered_window_slide} ns). Refiltering will alter the window positions."
            )

        highest_freq_nhz = max([measure['freq_nhz'] for measure in self.validated_data_dict['measures']])
        row_size = int((highest_freq_nhz * window_duration) // (10 ** 18))
        slide_size = int((highest_freq_nhz * window_slide) // (10 ** 18))
        row_period_ns = int((10 ** 18) // highest_freq_nhz)

        filtered_sources = {}
        patient_info_cache = {}
        patient_history_cache = {}
        for source_type, sources in self.validated_data_dict['sources'].items():
            if source_type == "patient_ids":
                # Can't filter without device information
                filtered_sources[source_type] = sources
                continue

            filtered_sources[source_type] = {}
            for source_id, time_ranges in sources.items():
                if source_type == "device_ids":
                    device_id = source_id
                    patient_id = None
                elif source_type == "device_patient_tuples":
                    device_id, patient_id = source_id
                else:
                    raise ValueError(
                        f"Source type must be either device_ids or device_patient_tuples, not {source_type}")
                filtered_sources[source_type][source_id] = []
                if patient_id is not None:
                    _load_patient_cache(patient_id, patient_info_cache, patient_history_cache, sdk, patient_history_fields)

                for start_time, end_time in time_ranges:
                    duration = end_time - start_time
                    if duration <= 0 or (not allow_partial_windows and duration < window_duration):
                        continue

                    # Calculate number of windows
                    num_windows = int((duration - window_duration) // window_slide) + 1
                    if allow_partial_windows and ((duration - window_duration) % window_slide > 0):
                        num_windows += 1

                    if num_windows <= 0:
                        continue

                    # Get data for each measure
                    data_dictionary = get_signal_dictionary(
                        sdk, device_id, None, window_duration, window_slide,
                        self.validated_data_dict['measures'], start_time, end_time, num_windows, start_time, end_time)

                    sliced_labels, threshold_labels = get_label_dictionary(
                        sdk, device_id, None, start_time, end_time,
                        self.validated_data_dict['labels'],
                        label_threshold, num_windows, row_period_ns, row_size, slide_size)

                    batch_window_list = get_window_list(device_id, patient_id, self.validated_data_dict['measures'], data_dictionary,
                                                        start_time, num_windows, window_duration,
                                                        threshold_labels, sliced_labels, patient_history_cache,
                                                        patient_history_fields, patient_info_cache)

                    accepted_intervals = []
                    current_interval_start = None
                    current_interval_end = None

                    for window in batch_window_list:
                        window_start_time = window.start_time
                        window_end_time = window_start_time + window_duration
                        if window_end_time > end_time:
                            window_end_time = end_time

                        if filter_fn(window):
                            # If we have no active interval, start one
                            if current_interval_start is None:
                                current_interval_start = window_start_time
                                current_interval_end = window_end_time
                            else:
                                # Check if contiguous or overlapping with the last accepted interval
                                if window_start_time <= current_interval_end:
                                    # Merge intervals if overlapping or contiguous
                                    if window_end_time > current_interval_end:
                                        current_interval_end = window_end_time
                                else:
                                    # Non-contiguous, finalize the previous interval
                                    accepted_intervals.append([current_interval_start, current_interval_end])
                                    # Start a new interval
                                    current_interval_start = window_start_time
                                    current_interval_end = window_end_time
                        else:
                            # Rejected window breaks any ongoing accepted interval
                            if current_interval_start is not None:
                                accepted_intervals.append([current_interval_start, current_interval_end])
                                current_interval_start = None
                                current_interval_end = None

                    # If we ended with an active interval, finalize it
                    if current_interval_start is not None:
                        accepted_intervals.append([current_interval_start, current_interval_end])

                    # accepted_intervals now holds all sub-intervals from [start_time, end_time]
                    # that passed the filter. We add these intervals to filtered_sources.
                    filtered_sources[source_type][source_id].extend(accepted_intervals)

        # Update the filtered_window_size and filtered_window_slide
        self.validated_data_dict['sources'] = filtered_sources
        self.filtered_window_size = window_duration
        self.filtered_window_slide = window_slide

    def read_yaml(self, filename):
        try:
            with open(filename, 'r') as file:
                loaded_data = yaml.load(file, Loader=yaml.FullLoader)
                for key, value in loaded_data.items():
                    if key == "measures":
                        # Convert measure lists to tuples
                        value = [
                            tuple(item) if isinstance(item, list) else item
                            for item in value
                        ]
                    if key in self.data_dict:
                        self.data_dict[key] = value
                    else:
                        raise ValueError(f"Unexpected key: {key}")
        except FileNotFoundError:
            raise FileNotFoundError("The specified file does not exist.")
        except yaml.YAMLError as err:
            raise ValueError("An error occurred while reading the YAML file.") from err

    def _check_format_and_convert_data(self):
        # Convert any numpy types to native Python
        self.data_dict = convert_numpy_types(self.data_dict)

        # Validate measures
        seen = set()
        for measure in self.data_dict['measures']:
            if isinstance(measure, tuple):
                # Ensure tuple has the format (str, float|int, str)
                if len(measure) != 3 or not isinstance(measure[0], str) or \
                   not isinstance(measure[1], (float, int)) or not isinstance(measure[2], str):
                    raise ValueError(
                        f"Invalid measure tuple: {measure}. Must be (str, float|int, str)"
                    )
            elif isinstance(measure, dict):
                # Validate measure dictionary
                if 'tag' not in measure or ('freq_hz' not in measure and 'freq_nhz' not in measure) or 'units' not in measure:
                    raise ValueError(
                        "Measure dictionary must contain 'tag', 'freq_hz' (or 'freq_nhz'), and 'units' keys"
                    )
                # Convert dict to string to make it hashable
                measure_str = json.dumps(measure, sort_keys=True)
                if measure_str in seen:
                    raise ValueError(f"Duplicate measure found: {measure}")
                seen.add(measure_str)
            elif isinstance(measure, str):
                # Handle measure as string
                if measure in seen:
                    raise ValueError(f"Duplicate measure found: {measure}")
                seen.add(measure)
            else:
                raise ValueError("Measure must be a string, dictionary, or tuple (str, float|int, str)")

        # Validate labels
        if not isinstance(self.data_dict['labels'], list):
            raise ValueError("labels must be a list or None.")
        for label_name in self.data_dict['labels']:
            if not isinstance(label_name, str):
                raise ValueError(f"Label {label_name} must be a string")

        # Validate and convert patient_ids
        for patient_id, times in self.data_dict['patient_ids'].items():
            if not isinstance(patient_id, int):
                raise ValueError(f"Patient ID {patient_id} must be an integer")
            self._check_times_and_warn(times, source_type="patient_id", source_id=patient_id)

        # Convert mrns to integers
        try:
            self.data_dict['mrns'] = {int(key): value for key, value in self.data_dict['mrns'].items()}
        except ValueError:
            raise ValueError("MRN must be convertible to an integer")
        for mrn, times in self.data_dict['mrns'].items():
            self._check_times_and_warn(times, source_type="mrn", source_id=mrn)

        # Validate and convert device_ids
        for device_id, times in self.data_dict['device_ids'].items():
            if not isinstance(device_id, int):
                raise ValueError(f"Device ID {device_id} must be an integer")
            self._check_times_and_warn(times, source_type="device_id", source_id=device_id)

        # Convert device_tags to strings
        self.data_dict['device_tags'] = {str(key): value for key, value in self.data_dict['device_tags'].items()}
        for device_tag, times in self.data_dict['device_tags'].items():
            self._check_times_and_warn(times, source_type="device_tag", source_id=device_tag)

    def _check_times_and_warn(self, times, source_type, source_id):
        if times == "all":
            return

        allowed_keys = ['start', 'end', 'time0', 'pre', 'post', 'files']

        for time_dict in times:
            if 'start' in time_dict and 'end' in time_dict and time_dict['start'] >= time_dict['end']:
                raise ValueError(f"{source_type} {source_id}: start time {time_dict['start']} must be less than "
                                 f"end time {time_dict['end']}")

            for key, value in time_dict.items():
                if key == 'files':
                    continue
                if key not in allowed_keys:
                    raise ValueError(f"{source_type} {source_id}: Invalid time key: {key}. "
                                     f"Allowed keys are: {', '.join(allowed_keys)}")

                if key in ['pre', 'post'] and value < 0:
                    raise ValueError(f"{source_type} {source_id}: {key} cannot be negative")

                if value < 1e9 or (value < 1e16 and key in ['start', 'end', 'time0']):
                    warnings.warn(f"{source_type} {source_id}: The epoch for {key}: {value} looks like it's "
                                  f"formatted in seconds. However {key} will be interpreted as nanosecond data.")

            if ('pre' in time_dict or 'post' in time_dict) and 'time0' not in time_dict:
                raise ValueError(f"{source_type} {source_id}: 'pre' and 'post' cannot be provided without 'time0'")

    def add_measure(self, measure_tag, freq=None, units=None):
        """
        Adds a new measure to the definition.

        :param measure_tag: Tag (string) representing the measure.
        :type measure_tag: str
        :param freq: Frequency in Hertz for the measure.
                     Only required if units are provided.
        :type freq: float, optional
        :param units: Units for the measure.
                      Only required if freq is provided.
        :type units: str, optional
        :raises ValueError: Raised when the measure tag is already present or when only one of freq and units is provided.

        **Examples**:

        >>> dataset_definition.add_measure(measure_tag="ART_BLD_PRESS", freq=250, units="mmHG")
        >>> dataset_definition.add_measure(measure_tag="PULSE")
        """
        # Check if exact same measure specification is already present
        for measure in self.data_dict['measures']:
            if (isinstance(measure, str) and measure == measure_tag) or \
                    (isinstance(measure, dict) and measure.get('tag') == measure_tag and
                     measure.get('freq_hz') == freq and measure.get('units') == units):
                warnings.warn(f"The exact same measure {measure} is already present, skipping duplicate")
                return

        # Add a measure to the measures list
        if freq and units:
            self.data_dict['measures'].append({'tag': measure_tag, 'freq_hz': freq, 'units': units})
        else:
            self.data_dict['measures'].append(measure_tag)

        return self

    def add_label(self, label_name):
        """
        Adds a new label to the definition.

        In the context of creating an iterator, labels specified will be included in the Window
        information if present.

        A label can be considered as a categorization or classification applied to a data point
        or a set of data points in the dataset. It might represent some meaningful information
        like 'abnormal', 'healthy', 'artifact', etc. for data sections or points.

        :param label_name: Name of the label to be added.
        :type label_name: str

        **Examples**:

        >>> dataset_definition.add_label(label_name="abnormal")
        >>> dataset_definition.add_label(label_name="artifact")

        :raises ValueError: If the label is already present in the definition.
        """
        if label_name in self.data_dict['labels']:
            raise ValueError(f"The label '{label_name}' is already present in the definition.")
        else:
            self.data_dict['labels'].append(label_name)

        return self

    def add_region(self, patient_id=None, mrn=None, device_id=None, device_tag=None, start=None, end=None, time0=None,
                   pre=None, post=None):

        """
        Adds a new region to the definition.

        :param patient_id: Identifier for a patient.
        :type patient_id: int, optional
        :param mrn: Medical record number.
        :type mrn: int, optional
        :param device_id: Identifier for a device.
        :type device_id: int, optional
        :param device_tag: Tag for a device.
        :type device_tag: str, optional
        :param start: Start time for interval-based specification.
        :type start: int, optional
        :param end: End time for interval-based specification.
        :type end: int, optional
        :param time0: Reference time for event-based specification.
        :type time0: int, optional
        :param pre: Time before the `time0` for event-based specification.
        :type pre: int, optional
        :param post: Time after the `time0` for event-based specification.
        :type post: int, optional
        :raises ValueError: Raised on invalid input combinations.

        **Examples**:

        >>> dataset_definition.add_region(patient_id=12345, start=1693364515_000_000_000, end=1693464515_000_000_000)
        >>> dataset_definition.add_region(device_tag="tag_1", time0=1693364515_000_000_000, pre=5000_000_000_000, post=6000_000_000_000)
        """
        # Check if both interval-based and event-based times are specified
        if (start or end) and (time0 or pre or post):
            raise ValueError("Cannot specify both interval-based and event-based times.")

        # Check if all necessary times are specified
        if (start or end) and (time0 or pre or post):
            raise ValueError("Mixing interval and event based time formats.")
        if (time0 or pre or post) and not (time0 and pre and post):
            raise ValueError("time0, pre, and post must all be specified for an event-based region.")

        # Prepare the region data
        region_data = {}
        if start and end:
            region_data = {'start': start, 'end': end}
        if time0 and pre and post:
            region_data = {'time0': time0, 'pre': pre, 'post': post}

        # Check if more than one identifier is specified
        identifiers = [patient_id, mrn, device_id, device_tag]
        if sum(x is not None for x in identifiers) > 1:
            raise ValueError("Only one of patient_id, mrn, device_id, device_tag should be specified.")

        # Identify the target dictionary and key
        target_dict, key = None, None
        if patient_id:
            target_dict, key = self.data_dict['patient_ids'], patient_id
        elif mrn:
            target_dict, key = self.data_dict['mrns'], mrn
        elif device_id:
            target_dict, key = self.data_dict['device_ids'], device_id
        elif device_tag:
            target_dict, key = self.data_dict['device_tags'], device_tag

        # Update the target dictionary
        if target_dict is not None and key is not None:
            if key in target_dict:
                target_dict[key].append(region_data)
            else:
                target_dict[key] = [region_data] if region_data else 'all'
        else:
            raise ValueError("One of patient_id, mrn, device_id, device_tag must be specified.")

        return self

    def save(self, filepath, force=False):
        """
        Saves the definition to a file. If the extension is `.yaml` or `.yml`,
        saves the original data_dict as YAML. If the extension is `.pkl`,
        saves the validated dataset definition using pickle.

        :param filepath: Path where the YAML file should be saved.
        :type filepath: str
        :param force: If set to True, overwrites the file if it already exists.
                      Default is False.
        :type force: bool, optional
        :raises OSError: Raised when the file already exists and `force` is not set to True.
        :raises ValueError: Raised when file extension is not .yaml.

        **Examples**:

        >>> dataset_definition.save(filepath="path/to/saved/definition.yaml")
        >>> dataset_definition.validate(sdk=my_sdk)
        >>> # Save validated binary dataset definition.
        >>> dataset_definition.save(filepath="path/to/definition.pkl", force=True)  # force=True overwrites file.
        """
        # Check if the file already exists
        if os.path.exists(filepath) and not force:
            raise OSError("File already exists, to overwrite set force=True")

        _, ext = os.path.splitext(filepath)
        if ext in ['.yaml', '.yml']:
            # YAML save
            converted_data = convert_numpy_types(self.data_dict)
            with open(filepath, 'w') as file:
                yaml.dump(converted_data, file, sort_keys=False)
        elif ext == '.pkl':
            # Pickle save
            if not self.is_validated:
                raise ValueError("Dataset can't be saved as a validated dataset because it has not been validated "
                                 "yet, call DatasetDefinition.validate() in order to validate it.")

            data_to_save = {
                "data_dict": self.data_dict,
                "validated_data_dict": self.validated_data_dict,
                "is_validated": self.is_validated,
                "validated_gap_tolerance": self.validated_gap_tolerance,
                "filtered_window_size": self.filtered_window_size,
                "filtered_window_slide": self.filtered_window_slide,
            }
            with open(filepath, 'wb') as f:
                pickle.dump(data_to_save, f)
        else:
            raise ValueError("File extension must be yaml/yml or pkl.")

    def _load_validated_pickle(self, filename):
        """
        Internal method to load a validated dataset definition from a pickle file.
        """
        if not os.path.exists(filename):
            raise FileNotFoundError("The specified file does not exist.")

        with open(filename, 'rb') as f:
            loaded_data = pickle.load(f)

        required_keys = ["data_dict", "validated_data_dict", "is_validated"]
        for key in required_keys:
            if key not in loaded_data:
                raise ValueError(f"Loaded pickle does not contain required key '{key}'")

        self.data_dict = loaded_data["data_dict"]
        self.validated_data_dict = loaded_data["validated_data_dict"]
        self.is_validated = loaded_data["is_validated"]
        self.validated_gap_tolerance = loaded_data["validated_gap_tolerance"]
        self.filtered_window_size = loaded_data["filtered_window_size"]
        self.filtered_window_slide = loaded_data["filtered_window_slide"]

def convert_numpy_types(data):
    if isinstance(data, dict):
        return {convert_numpy_types(key): convert_numpy_types(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_numpy_types(element) for element in data]
    elif isinstance(data, np.generic):
        return data.item()
    else:
        return data
