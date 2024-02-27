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
import warnings
import json

from atriumdb.windowing.definition_builder import build_source_intervals


class DatasetDefinition:
    """
    Encapsulates the Dataset Definition in a YAML format.

    The ``DefinitionYAML`` class represents the definition of dataset sources
    (devices or patients), their associated time regions, and the available signals
    over those time regions for the respective sources, along with any labels
    associated with these sources.

    :ivar data_dict: A dictionary containing the structured data from the YAML file
                     or passed parameters. The keys include 'measures', 'patient_ids',
                     'mrns', 'device_ids', 'device_tags', and 'labels'.

    :param filename: Path to the YAML file containing the dataset definition. If
                     provided, the contents of the file will populate `data_dict`. The
                     YAML format is described in detail at :ref:`definition_file_format`.
    :type filename: str, optional

    :param measures: List of requested measures to be considered. Measures can be:
                     1. Just a tag (string) representing the measure.
                     2. A dictionary specifying the measure tag, its frequency in Hertz, and its units.
                     Example: [{"tag": "ECG", "freq_hz": 300, "units": "mV"}]
    :type measures: list, optional

    :param patient_ids: Dictionary containing patient identifiers with their associated
                        time specifications. The specifications can be interval-based, event-based,
                        or the keyword 'all' to indicate all available time data for the given patient.
                        Example: {1234567: "all", 7654321: [{'start': 123456000 * 10**9, 'end': 123457000 * 10**9}]}
    :type patient_ids: dict, optional

    :param mrns: Dictionary containing medical record numbers. Each MRN can have associated
                 time specifications (interval-based, event-based) or 'all' indicating all available
                 time data for the given MRN.
                 Example: {1234567: "all", 7654321: [{'start': 123456000 * 10**9, 'end': 123457000 * 10**9}]}
    :type mrns: dict, optional

    :param device_ids: Dictionary containing device identifiers, each associated with its time
                       specifications. The specifications can be interval, event, or 'all' for all
                       available time data.
                       Example: {1: "all", 14: [{'time0': 123456000 * 10**9, 'pre': 1000 * 10**9, 'post': 3000 * 10**9}]}
    :type device_ids: dict, optional

    :param device_tags: Dictionary containing device tags. Each tag is mapped to its respective
                        time specifications, which can be interval-based, event-based, or simply 'all'.
                        Example: {"dev_1": "all", "dev_2b": [{'start': 123456000 * 10**9}]}
    :type device_tags: dict, optional

    :param labels: List of strings, each representing a unique label associated with the dataset elements
                   (such as patient conditions or device states). These labels can be used for classification,
                   segmentation, or other forms of analysis.
                   Example: ["Sinus rhythm", "Junctional ectopic tachycardia (JET)", "arrhythmia"]
    :type labels: list, optional

    .. note:: For more details on the format expectations, see :ref:`definition_file_format`.

    **Examples**:

    **Reading from an existing YAML file**:

    >>> dataset_definition = DatasetDefinition(filename="/path/to/my_definition.yaml")

    **Creating an empty definition**:

    >>> dataset_definition = DatasetDefinition()

    **Creating a definition with measures and labels**:

    >>> measures = ["measure_tag_1", {"tag": "measure_tag_2", "freq_hz": 62.5, "units": "measure_units_2"}]
    >>> labels = ["label1", "label2"]
    >>> dataset_definition = DatasetDefinition(measures=measures, labels=labels)

    **Adding a label to the definition**:

    >>> dataset_definition.add_label("new_label")

    **Saving the definition to a YAML file**:

    >>> dataset_definition.save(filepath="path/to/saved/definition.yaml")
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

        if filename:
            self.read_yaml(filename)

        # Validate and convert the data
        self._validate_and_convert_data()

    @classmethod
    def build_from_intervals(cls, sdk, build_from_signal_type, measures=None, labels=None, patient_id_list=None,
                             mrn_list=None, device_id_list=None, device_tag_list=None, start_time=None, end_time=None,
                             gap_tolerance=None, merge_strategy=None):
        """
        Class method that builds a DatasetDefinition object using signal-based intervals.

        :param sdk: Data SDK used to interact with the database or data service
        :param build_from_signal_type: Signal type to build from, either "measures" or "labels"
        :param measures: List of measures to build from, if applicable
        :param labels: List of labels to build from, if applicable
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

        # Build the source intervals using the build_source_intervals function
        if build_from_signal_type == "measures":
            assert measures, "If you are building on measures, you must provide measures"
            build_measures = measures
            build_labels = None
        else:
            # "labels"
            assert labels, "If you are building on labels, you must provide labels"
            build_measures = None
            build_labels = labels

        source_intervals = build_source_intervals(sdk, measures=build_measures, labels=build_labels,
                                                  patient_id_list=patient_id_list,
                                                  mrn_list=mrn_list, device_id_list=device_id_list,
                                                  device_tag_list=device_tag_list, start_time=start_time,
                                                  end_time=end_time, gap_tolerance=gap_tolerance,
                                                  merge_strategy=merge_strategy)

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

    def read_yaml(self, filename):
        try:
            with open(filename, 'r') as file:
                loaded_data = yaml.load(file, Loader=yaml.FullLoader)
                for key, value in loaded_data.items():
                    if key in self.data_dict:
                        self.data_dict[key] = value
                    else:
                        raise ValueError(f"Unexpected key: {key}")
        except FileNotFoundError:
            raise FileNotFoundError("The specified file does not exist.")
        except yaml.YAMLError as err:
            raise ValueError("An error occurred while reading the YAML file.") from err

    def _validate_and_convert_data(self):
        # Validate measures
        seen = set()
        for measure in self.data_dict['measures']:
            # convert dict to string to make it hashable
            measure_str = json.dumps(measure, sort_keys=True) if isinstance(measure, dict) else measure
            if measure_str in seen:
                raise ValueError(f"Duplicate measure found: {measure}")
            seen.add(measure_str)
            if not isinstance(measure, (str, dict)):
                raise ValueError("Measure must be a string or a dictionary")
            if isinstance(measure, dict):
                if 'tag' not in measure or ('freq_hz' not in measure and 'freq_nhz' not in measure) or 'units' not in measure:
                    raise ValueError("Measure dictionary must contain 'tag', 'freq_hz' (or 'freq_nhz'), and 'units' keys")

        # Validate labels
        if not isinstance(self.data_dict['labels'], list):
            raise ValueError("labels must be a list or None.")
        for label_name in self.data_dict['labels']:
            if not isinstance(self.data_dict['labels'], list):
                raise ValueError("labels must be a list or None.")

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

    def save(self, filepath, force=False):
        """
        Saves the definition to a YAML file.

        :param filepath: Path where the YAML file should be saved.
        :type filepath: str
        :param force: If set to True, overwrites the file if it already exists.
                      Default is False.
        :type force: bool, optional
        :raises OSError: Raised when the file already exists and `force` is not set to True.
        :raises ValueError: Raised when file extension is not .yaml.

        **Examples**:

        >>> dataset_definition.save(filepath="path/to/saved/definition.yaml")
        """
        # Check if the file already exists
        if os.path.exists(filepath) and not force:
            raise OSError("File already exists, to overwrite set force=True")

        # Check file extension
        _, ext = os.path.splitext(filepath)
        if ext not in ['.yaml', '.yml']:
            raise ValueError("File extension must be yaml/yml.")

        # Save the dataset definition to a YAML file
        with open(filepath, 'w') as file:
            yaml.dump(self.data_dict, file, sort_keys=False)
