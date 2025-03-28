from atriumdb.windowing.verify_definition import verify_definition
from atriumdb.windowing.definition import DatasetDefinition
import numpy as np
import random
import bisect
import copy
from tqdm import tqdm


def partition_dataset(definition, sdk, partition_ratios, priority_stratification_labels=None, random_state=None,
                      verbose=False, n_trials=None, num_show_best_trials=None, gap_tolerance=60_000_000_000):
    """
    Partition a dataset into training, testing, and optionally validation sets, with an option for stratified splitting based on priority labels.

    This function processes a dataset, validating its components and splitting it according to specified criteria. It is designed to work with health-related data, focusing on ensuring balanced representation of important categories in each split dataset.

    :param definition: An instance of the DatasetDefinition class from the atriumdb library. This object contains the specifications and parameters of the dataset to be partitioned.
    :param sdk: An instance of the AtriumSDK used to interact with the dataset and perform necessary conversions or validations.
    :param partition_ratios: A list of N integers representing the distribution of stratified variables over N partitions.
    :param priority_stratification_labels: Optional. A list of labels (either as integers or strings) given priority in the stratification process. If integers, they are treated as label set IDs. If strings, they are converted to IDs using the SDK.
    :param random_state: Optional. An integer that seeds the random number generator for reproducible splits.
    :param verbose: Optional. A boolean flag that, when set to True, enables the function to print detailed information about the splitting process.
    :param n_trials: Optional. An integer specifying the number of trials to run with different random states in order to find the most balanced partitioning. If None, the function runs only once using the provided or default random state.
    :param num_show_best_trials: Optional. An integer specifying the number of best trials to display if verbose is True and multiple trials (n_trials > 1) are run. If None or 0, no trials are displayed.
    :param gap_tolerance: Optional. An integer specifying the minimum allowed gap (in nanoseconds) in any time ranges generated from a dataset defintition source id being set to "all". (Default 1 minute)

    :return: A tuple of DatasetDefinition instances for the training, testing, and optionally validation sets. The tuple will contain two or three elements depending on whether a validation set is created.

    Example:
    --------
    >>> from atriumdb import DatasetDefinition, AtriumSDK, partition_dataset
    >>> definition = DatasetDefinition(...)
    >>> sdk = AtriumSDK(...)
    >>> train_def, test_def, val_def = partition_dataset(
            definition,
            sdk,
            partition_ratios=[60, 20, 20],
            priority_stratification_labels=['label1', 'label2'],
            random_state=42,
            verbose=False,
            gap_tolerance=10**9  # 1 second
        )
    """
    if len(definition.data_dict['measures']) == 0:
        raise ValueError("Supplied dataset has no measures and therefore cannot stratify based on measure availability")

    # Validate the dataset definition and gather necessary components.
    validated_measure_list, validated_label_set_list, validated_sources = verify_definition(
        definition, sdk, gap_tolerance=gap_tolerance)

    # Convert priority stratification labels to label set IDs, using the SDK if necessary.
    priority_stratification_label_set_ids = get_priority_stratification_label_set_ids(
        priority_stratification_labels, validated_label_set_list, sdk)

    # Calculate the duration for each label in each data source.
    label_duration_list = get_label_duration_list(validated_sources, priority_stratification_label_set_ids, sdk)

    trials_results = []
    # Generate list of random states if n_trials is given
    random_states = get_random_states(n_trials, random_state) if n_trials else [random_state]

    for trial_random_state in tqdm(random_states, desc="Running partition distribution trials", unit="trial"):
        # Perform stratified partitioning of the dataset based on the labels.
        partitioned_source_list, partitioned_durations, partition_source_counts = stratified_partition_by_labels(
            label_duration_list, partition_ratios, random_state=trial_random_state)

        # Gather information about the distribution of durations across different partitions.
        duration_info = get_duration_info(partitioned_durations, priority_stratification_labels, partition_source_counts)
        ratio_obedience_metric = evaluate_ratio_obedience_metric(duration_info, partition_ratios)
        trials_results.append((ratio_obedience_metric, trial_random_state, duration_info))

        if n_trials is None:
            # If we're not using trials, return immediately.
            # Convert the partitioned source lists into DatasetDefinition objects.
            partitioned_definition_objects = convert_source_lists_to_definitions(partitioned_source_list, definition)
            if verbose:
                return partitioned_definition_objects, duration_info
            return partitioned_definition_objects

    # Sort the trials to find the best one.
    best_trials = sorted(trials_results, key=lambda x: x[0])

    # Display the specified number of best trials if verbose is True.
    if verbose and num_show_best_trials is not None:
        print(f"Displaying top {num_show_best_trials} trials:")
        for i, trial in enumerate(best_trials[:num_show_best_trials]):
            trial_random_state, trial_ratio_obedience_metric, trial_duration_info = trial[1], trial[0], trial[2]
            print(f"\nTrial {i + 1} - Random State: {trial_random_state}, RMSE: {trial_ratio_obedience_metric}")
            for partition_info in trial_duration_info:
                print(partition_info)
            print("-" * 50)

    # Select the best trial based on the metric.
    best_trial_random_state = best_trials[0][1] if best_trials else random_state

    # Rerun the partitioning using the best trial's random state.
    partitioned_source_list, partitioned_durations, partition_source_counts = stratified_partition_by_labels(
        label_duration_list, partition_ratios, random_state=best_trial_random_state)

    # Convert the partitioned source lists into DatasetDefinition objects.
    final_partitioned_definition_objects = convert_source_lists_to_definitions(partitioned_source_list, definition)

    # Gather final information about the distribution of durations across different partitions.
    final_duration_info = get_duration_info(partitioned_durations, priority_stratification_labels,
                                            partition_source_counts)

    if verbose:
        print("Final partitioning using best trial's random state:")
        print(f"Random State: {best_trial_random_state}")
        print('-' * 50)
        for partition in final_duration_info:
            print(partition)
        print("-" * 50 + "\n")
        return final_partitioned_definition_objects, final_duration_info

    return final_partitioned_definition_objects


def get_priority_stratification_label_set_ids(priority_labels, validated_label_set_list, sdk):
    if not priority_labels:
        return []

    if all(isinstance(label, int) for label in priority_labels):
        label_set_ids = priority_labels
    elif all(isinstance(label, str) for label in priority_labels):
        label_set_ids = [sdk.get_label_name_id(label) for label in priority_labels]
    else:
        raise ValueError("Priority stratification labels must be all integers or all strings.")

    if not all(label_id in validated_label_set_list for label_id in label_set_ids):
        raise ValueError("One or more priority stratification labels are not in the validated label set list.")

    return label_set_ids


def merge_intervals(intervals):
    if not intervals:
        return []

    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    merged = [list(sorted_intervals[0])]
    for current in sorted_intervals[1:]:
        prev = merged[-1]
        if current[0] <= prev[1]:
            prev[1] = max(prev[1], current[1])
        else:
            merged.append(list(current))

    return [tuple(interval) for interval in merged]


def get_label_duration_list(validated_sources, priority_stratification_label_set_ids, sdk):
    if "device_patient_tuples" not in validated_sources or not validated_sources["device_patient_tuples"]:
        raise ValueError(
            "There is no patient-mapped data in the dataset and so it is impossible to stratify by patient."
        )

    label_duration_list = []
    label_to_index_dict = {label: label_i for label_i, label in enumerate(priority_stratification_label_set_ids)}

    # Get all the label results
    label_result = sdk.sql_handler.select_labels(
        label_set_id_list=priority_stratification_label_set_ids,
    )

    # Sort them by device
    device_labels = {}
    for (label_entry_id, label_set_id, device_id, measure_id, label_source_id,
         start_time_n, end_time_n) in label_result:
        if device_id not in device_labels:
            device_labels[device_id] = []
        formatted_label = {
            'label_entry_id': label_entry_id,
            'label_name_id': label_set_id,
            'device_id': device_id,
            'start_time_n': start_time_n,
            'end_time_n': end_time_n,
            'label_source_id': label_source_id,
            'measure_id': measure_id
        }
        device_labels[device_id].append(formatted_label)

    # Sort the device results by time
    label_starts = {}
    label_ends = {}
    for device_id in device_labels.keys():
        device_labels[device_id] = sorted(device_labels[device_id], key=lambda x: (x['start_time_n'], x['end_time_n']))
        label_starts[device_id] = [entry['start_time_n'] for entry in device_labels[device_id]]
        label_ends[device_id] = [entry['end_time_n'] for entry in device_labels[device_id]]

    patient_results = {}

    for source_key, time_ranges in list(validated_sources["device_patient_tuples"].items()):
        device_id, patient_id = source_key
        if patient_id not in patient_results:
            patient_results[patient_id] = {
                "time_ranges": [],
                # durations[0] will be waveform duration; each subsequent index corresponds to a label
                "durations": [0] * (len(priority_stratification_label_set_ids) + 1)
            }
        for start_time, end_time in time_ranges:
            # Fetch labels within this time range
            patient_results[patient_id]["time_ranges"].append((start_time, end_time))

            # Add waveform duration.
            patient_results[patient_id]["durations"][0] += (end_time - start_time)

            # Fetch labels within this time range
            if device_id not in device_labels:
                range_labels = []
            else:
                range_labels = find_labels(
                    device_labels[device_id],
                    label_starts[device_id],
                    label_ends[device_id],
                    start_time,
                    end_time
                )
            # Calculate the duration for each priority label within this time range
            for label in range_labels:
                if label['label_name_id'] in label_to_index_dict:
                    label_index = label_to_index_dict[label['label_name_id']]
                    label_duration = label['end_time_n'] - label['start_time_n']
                    patient_results[patient_id]["durations"][label_index + 1] += label_duration

    # Merge and sort the time ranges per patient.
    for patient_id, result in patient_results.items():
        result["time_ranges"] = merge_intervals(result["time_ranges"])

    for patient_id, result in patient_results.items():
        label_duration_list.append(["patient_ids", patient_id, result["time_ranges"]] + result["durations"])

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
    partition_source_counts = [0 for _ in range(len(ratios))]

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
        partition_source_counts[best_partition] += 1

    return partitions, partition_sums, partition_source_counts


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
                # patient_id = source_key[1]
                # patient_ids[patient_id] = time_specifications

                device_id = source_key[0]
                if device_id not in device_ids:
                    device_ids[device_id] = []

                device_ids[device_id].extend(time_specifications)

            elif source_type == 'patient_ids':
                patient_id = source_key
                if patient_id not in patient_ids:
                    patient_ids[patient_id] = []
                patient_ids[patient_id].extend(time_specifications)
            elif source_type == 'device_ids':
                device_id = source_key
                if device_id not in device_ids:
                    device_ids[device_id] = []

                device_ids[device_id].extend(time_specifications)
            else:
                raise ValueError("Invalid source_type encountered")

        # Merge time ranges for patient_ids and device_ids
        patient_ids = {pid: merge_time_ranges(ranges) for pid, ranges in patient_ids.items()}
        device_ids = {did: merge_time_ranges(ranges) for did, ranges in device_ids.items()}

        # Create a new definition for this partition
        partition_def = DatasetDefinition(measures=measures, patient_ids=patient_ids,
                                          device_ids=device_ids, labels=labels)
        partitioned_definitions.append(partition_def)

    return partitioned_definitions


def merge_time_ranges(time_ranges):
    if not time_ranges:
        return []

    # Sort by the 'start' time
    time_ranges.sort(key=lambda x: x['start'])

    merged_ranges = []
    current_range = time_ranges[0]

    for next_range in time_ranges[1:]:
        if current_range['end'] == next_range['start']:
            # Merge the ranges
            current_range['end'] = next_range['end']
        else:
            # No merging needed, push the current range and move to the next
            merged_ranges.append(current_range)
            current_range = next_range

    # Add the last range
    merged_ranges.append(current_range)

    return merged_ranges


def get_duration_info(partitioned_durations, priority_stratification_labels, partition_source_counts,
                      convert_to_hours=True):
    duration_info_list = []
    time_units = "hours" if convert_to_hours else "nanoseconds"

    for partition_index, (durations, source_count) in enumerate(zip(partitioned_durations, partition_source_counts)):
        total_waveform_duration = round(durations[0] / (3600 * 1e9), 3) if convert_to_hours else durations[0]
        duration_info = {
            "partition": partition_index,
            f"total waveform {time_units}": total_waveform_duration,
            "num_sources": source_count
        }
        for label_index, label in enumerate(priority_stratification_labels, start=1):
            label_key = f"{label} {time_units}"
            duration_info[label_key] = round(durations[label_index] / (3600 * 1e9), 3) if convert_to_hours else durations[label_index]

        duration_info_list.append(duration_info)

    return duration_info_list


def evaluate_ratio_obedience_metric(duration_info, target_ratios):
    """
    Evaluate how well the partitions divided the total waveform time and each label according to the target ratios.

    :param duration_info: List of dictionaries containing information about partition durations.
    :param target_ratios: The target ratios used for splitting, a list of integers like [1, 2, 3].
    :return: The computed ratio_obedience_metric.
    """
    # Convert target ratios to proportions
    total_target_ratio = sum(target_ratios)
    target_proportions = [r / total_target_ratio for r in target_ratios]

    # Initialize variables to store sums of squared errors for each category
    error_sums = {key: 0 for key in duration_info[0] if key not in ['partition', 'num_sources']}
    num_partitions = len(duration_info)

    # Calculate sums of squared errors
    for key in error_sums:
        actual_durations = [partition[key] for partition in duration_info]
        total_duration = sum(actual_durations)
        expected_durations = [total_duration * prop for prop in target_proportions]

        for actual, expected in zip(actual_durations, expected_durations):
            error_sums[key] += (actual - expected) ** 2

    # Calculate RMSE for each category and combine them
    rmses = [np.sqrt(error_sum / num_partitions) for error_sum in error_sums.values()]
    combined_metric = np.mean(rmses)  # or use np.sum(rmses) for a cumulative error

    return combined_metric


def get_random_states(n_trials, random_state=None):
    """
    Generate a list of random states.

    :param n_trials: Number of random states to generate.
    :param random_state: Seed for the random number generator.
    :return: A list of random states.
    """
    rng = np.random.default_rng(random_state)
    return rng.choice(np.iinfo(np.int32).max, size=n_trials, replace=False).tolist()


def find_labels(sorted_labels, starts, ends, start_time, end_time):
    start_idx = bisect.bisect_left(ends, start_time)
    end_idx = bisect.bisect_left(ends, end_time)

    if start_idx == end_idx:
        if start_idx >= len(starts):
            return []
        if (not (starts[start_idx] <= start_time <= ends[start_idx])
                and not (starts[end_idx] <= end_time <= ends[end_idx])):
            return []

    if end_idx < len(starts) and end_time < starts[end_idx]:
        end_idx = max(0, end_idx - 1)

    return sorted_labels[start_idx:end_idx + 1]