from atriumdb.windowing.verify_definition import verify_definition
from atriumdb.windowing.definition import DatasetDefinition
import numpy as np
import random
import copy


def partition_dataset(definition, sdk, partition_ratios, priority_stratification_labels=None, random_state=None,
                      verbose=False):
    """
    Partition a dataset into training, testing, and optionally validation sets, with an option for stratified splitting based on priority labels.

    This function processes a dataset, validating its components and splitting it according to specified criteria. It is designed to work with health-related data, focusing on ensuring balanced representation of important categories in each split dataset.

    :param definition: An instance of the DatasetDefinition class from the atriumdb library. This object contains the specifications and parameters of the dataset to be partitioned.
    :param sdk: An instance of the AtriumSDK used to interact with the dataset and perform necessary conversions or validations.
    :param partition_ratios: A list of N integers representing the distribution of stratified variables over N partitions.
    :param priority_stratification_labels: Optional. A list of labels (either as integers or strings) given priority in the stratification process. If integers, they are treated as label set IDs. If strings, they are converted to IDs using the SDK.
    :param random_state: Optional. An integer that seeds the random number generator for reproducible splits.
    :param verbose: Optional. A boolean flag that, when set to True, enables the function to print detailed information about the splitting process.

    :return: A tuple of DatasetDefinition instances for the training, testing, and optionally validation sets. The tuple will contain two or three elements depending on whether a validation set is created.

    Example:
    --------
    >>> from atriumdb import DatasetDefinition, AtriumSDK
    >>> definition = DatasetDefinition(...)
    >>> sdk = AtriumSDK(...)
    >>> train_def, test_def, val_def = partition_dataset(
            definition,
            sdk,
            partition_ratios=[60, 20, 20],
            priority_stratification_labels=['label1', 'label2'],
            random_state=42,
            verbose=False
        )
    """

    # Validate the dataset definition and gather necessary components.
    validated_measure_list, validated_label_set_list, validated_sources = verify_definition(
        definition, sdk, gap_tolerance=0)

    # Convert priority stratification labels to label set IDs, using the SDK if necessary.
    priority_stratification_label_set_ids = get_priority_stratification_label_set_ids(
        priority_stratification_labels, validated_label_set_list, sdk)

    # Calculate the duration for each label in each data source.
    label_duration_list = get_label_duration_list(validated_sources, priority_stratification_label_set_ids, sdk)

    # Perform stratified partitioning of the dataset based on the labels.
    partitioned_source_list, partitioned_durations = stratified_partition_by_labels(
        label_duration_list, partition_ratios, random_state=random_state)

    # Convert the partitioned source lists into DatasetDefinition objects.
    partitioned_definition_objects = convert_source_lists_to_definitions(partitioned_source_list, definition)

    # Gather information about the distribution of durations across different partitions.
    duration_info = get_duration_info(partitioned_durations, priority_stratification_labels)

    # Optionally output duration information for each partition if verbose mode is enabled.
    if verbose:
        print(duration_info)
        return partitioned_definition_objects, duration_info

    return partitioned_definition_objects


def get_priority_stratification_label_set_ids(priority_labels, validated_label_set_list, sdk):
    if not priority_labels:
        return []

    if all(isinstance(label, int) for label in priority_labels):
        label_set_ids = priority_labels
    elif all(isinstance(label, str) for label in priority_labels):
        label_set_ids = [sdk.get_label_set_id(label) for label in priority_labels]
    else:
        raise ValueError("Priority stratification labels must be all integers or all strings.")

    if not all(label_id in validated_label_set_list for label_id in label_set_ids):
        raise ValueError("One or more priority stratification labels are not in the validated label set list.")

    return label_set_ids


def get_label_duration_list(validated_sources, priority_stratification_label_set_ids, sdk):
    label_duration_list = []
    label_to_index_dict = {label: label_i for label_i, label in enumerate(priority_stratification_label_set_ids)}

    for source_type, source_data in validated_sources.items():
        for source_key, time_ranges in source_data.items():
            # Initialize durations for this source, one for each label, plus one for waveform hours
            durations = [0] * (len(priority_stratification_label_set_ids) + 1)

            # (device_id, patient_id)
            if source_type == "device_patient_tuples":
                device_list = [source_key[0]]
                patient_id_list = None
            elif source_type == "device_ids":
                device_list = [source_key]
                patient_id_list = None
            elif source_type == "patient_ids":
                device_list = None
                patient_id_list = [source_key]
            else:
                raise ValueError(
                    f"source type must be device_patient_tuples, device_ids, patient_ids, not {source_type}")

            # Loop through each time range for this source
            for start_time, end_time in time_ranges:
                # Fetch labels within this time range
                labels = sdk.get_labels(
                    label_set_id_list=priority_stratification_label_set_ids,
                    device_list=device_list,
                    patient_id_list=patient_id_list,
                    start_time=start_time,
                    end_time=end_time,
                )
                durations[0] += end_time - start_time

                # Calculate the duration for each priority label within this time range
                for label in labels:
                    if label['label_name_id'] in label_to_index_dict:
                        label_index = label_to_index_dict[label['label_name_id']]
                        label_duration = label['end_time_n'] - label['start_time_n']
                        durations[label_index + 1] += label_duration

            # Add the computed durations to the list
            label_duration_list.append([source_type, source_key, time_ranges] + durations)

    return label_duration_list


def stratified_partition_by_labels(data_list, partition_ratios, random_state=None):
    data_list = copy.deepcopy(data_list)
    if any(ratio == 0 for ratio in partition_ratios):
        raise ValueError("Cannot have 0 in partition ratio.")

    if len(data_list) == 0:
        return [[] for _ in range(len(partition_ratios))], [[] for _ in range(len(partition_ratios))]
    # Set random state for reproducibility
    random_gen = random.Random(random_state) if random_state is not None else random.Random()

    # Calculate total sums for each label
    total_sums = np.array([sum(item[i] for item in data_list) for i in range(3, len(data_list[0]))])

    # Normalize the partition ratios
    ratios = np.array(partition_ratios) / sum(partition_ratios)
    target_sums = np.outer(total_sums, ratios).T

    # Initialize partitions and their total sums
    partitions = [[] for _ in range(len(ratios))]
    partition_sums = [np.zeros(len(total_sums), dtype=np.int64) for _ in range(len(ratios))]

    # Function to find the best partition for the current item
    def find_best_partition(item):
        item_sums = np.array(item[3:], dtype=np.int64)
        deficits = target_sums - partition_sums
        deficits[deficits < 0] = np.inf
        scores = np.sum(deficits - item_sums, axis=1)
        return np.argmin(scores)

    # Randomly shuffle the data list for more randomness in partitioning
    random_gen.shuffle(data_list)

    # Distribute items into partitions
    for item in data_list:
        best_partition = find_best_partition(item)
        partitions[best_partition].append(item)
        partition_sums[best_partition] += item[3:]

    return partitions, partition_sums


def convert_source_lists_to_definitions(partitioned_source_list, original_definition):
    # Get measures and labels from the original definition
    measures = original_definition.data_dict['measures']
    labels = original_definition.data_dict['labels']

    partitioned_definitions = []

    for partition in partitioned_source_list:
        # Initialize empty dictionaries for each source type
        patient_ids = {}
        device_ids = {}

        for entry in partition:
            source_type, source_key, time_ranges = entry[:3]
            # Reconstruct start, end dictionaries for the ranges
            time_specifications = [{'start': start_time, 'end': end_time}
                                   for start_time, end_time in time_ranges]

            # Depending on the source type, update the correct dictionary
            if source_type == 'device_patient_tuples':
                # Using patient_id and ignoring device_id for simplicity
                patient_id = source_key[1]
                patient_ids[patient_id] = time_specifications
            elif source_type == 'patient_ids':
                patient_id = source_key
                patient_ids[patient_id] = time_specifications
            elif source_type == 'device_ids':
                device_id = source_key
                device_ids[device_id] = time_specifications
            else:
                raise ValueError("Invalid source_type encountered")

        # Create a new definition for this partition
        partition_def = DatasetDefinition(measures=measures, patient_ids=patient_ids,
                                          device_ids=device_ids, labels=labels)
        partitioned_definitions.append(partition_def)

    return partitioned_definitions


def get_duration_info(partitioned_durations, priority_stratification_labels, convert_to_hours=True):
    duration_info_list = []
    time_units = "hours" if convert_to_hours else "nanoseconds"

    for partition_index, durations in enumerate(partitioned_durations):
        total_waveform_duration = durations[0] / (3600 * 1e9) if convert_to_hours else durations[0]
        duration_info = {"partition": partition_index, f"total waveform {time_units}": total_waveform_duration}
        for label_index, label in enumerate(priority_stratification_labels, start=1):
            label_key = f"{label} {time_units}"
            duration_info[label_key] = durations[label_index] / (3600 * 1e9) if convert_to_hours else durations[label_index]

        duration_info_list.append(duration_info)

    return duration_info_list
