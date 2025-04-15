from atriumdb.windowing.verify_definition import verify_definition
from atriumdb.windowing.definition import DatasetDefinition
import numpy as np
import random
import bisect
import copy
from tqdm import tqdm


def partition_dataset(definition, sdk, partition_ratios, priority_stratification_labels=None, additional_labels=None,
                      random_state=None, verbose=False, n_trials=None, num_show_best_trials=None,
                      gap_tolerance=60_000_000_000):
    """
    Partition a dataset into training, testing, validation sets or any number of N splits depending on how many `partition_ratios` you provide.

    :param definition: An instance of the DatasetDefinition class from the atriumdb library.
    :param sdk: An instance of the AtriumSDK used to interact with the dataset.
    :param partition_ratios: A list of N integers representing the distribution over N partitions.
    :param priority_stratification_labels: Optional. A list of label ids (int) or label names (str) used for stratification.
    :param additional_labels: Optional. A list of label ids (int) or label names (str) that are not used for stratification but whose
          durations will be tallied per patient and added to the duration_info which is returned
          and printed when `verbose=True`.
    :param random_state: Optional. An integer seed for reproducibility.
    :param verbose: Optional. If True, prints and returns detailed partitioning information as second argument.
    :param n_trials: Optional. Number of random trials to find the most balanced partitioning out of all trials,
        measured by how well the durations of the resulting priority_stratification_labels fit the requested partition_ratios.
    :param num_show_best_trials: Optional. Number of best trials to display if verbose is True.
    :param gap_tolerance: Optional. An integer specifying the minimum allowed gap in nanoseconds for time ranges.
    :return: A tuple of DatasetDefinition objects (one per partition). If verbose and n_trials is None, also returns
             the duration_info (a list of dicts containing duration and count info, including unique patients and
             additional label tallies).

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
    >>> # Or if verbose=True
    >>> (train_def, test_def, val_def), partition_duration_info = partition_dataset(
            definition,
            sdk,
            partition_ratios=[60, 20, 20],
            priority_stratification_labels=['label1', 'label2'],
            random_state=42,
            verbose=True,
            gap_tolerance=10**9  # 1 second
        )
    """
    if len(definition.data_dict['measures']) == 0:
        raise ValueError("Supplied dataset has no measures and therefore cannot stratify based on measure availability")

    if not definition.is_validated:
        definition.validate(sdk=sdk)

    # Extract validated data from the definition
    validated_data = definition.validated_data_dict
    validated_measure_list = validated_data['measures']
    validated_label_set_list = validated_data['labels']
    validated_sources = validated_data['sources']

    # Convert priority stratification labels to label set IDs, using the SDK if necessary.
    priority_stratification_label_set_ids = get_priority_stratification_label_set_ids(
        priority_stratification_labels, validated_label_set_list, sdk)

    # Convert additional labels (for reporting only) similarly.
    additional_label_set_ids = get_priority_stratification_label_set_ids(
        additional_labels, validated_label_set_list, sdk)

    # Calculate the duration for each patient for both priority and additional labels.
    label_duration_list, patient_additional_results = get_label_duration_list(
        validated_sources,
        priority_stratification_label_set_ids,
        sdk,
        additional_label_set_ids=additional_label_set_ids
    )

    trials_results = []
    # Generate list of random states if n_trials is given
    if n_trials in [0, 1]:
        n_trials = None
    random_states = get_random_states(n_trials, random_state) if n_trials else [random_state]

    for trial_random_state in tqdm(random_states, desc="Running partition distribution trials", unit="trial"):
        # Perform stratified partitioning using only waveform and priority label durations.
        partitioned_source_list, partitioned_durations, partition_source_counts = stratified_partition_by_labels(
            label_duration_list, partition_ratios, random_state=trial_random_state)

        # Compute additional label totals for each partition (they are not used for partitioning).
        additional_totals = None
        if additional_label_set_ids:
            additional_totals = []
            for partition in partitioned_source_list:
                total = np.zeros(len(additional_label_set_ids), dtype=np.int64)
                for item in partition:
                    patient_id = item[1]
                    if patient_id in patient_additional_results:
                        total += np.array(patient_additional_results[patient_id], dtype=np.int64)
                additional_totals.append(total)

        # Gather information about the distribution of durations across partitions.
        duration_info = get_duration_info(
            partitioned_durations,
            priority_stratification_labels,
            partition_source_counts,
            partitioned_source_list=partitioned_source_list,
            additional_durations=additional_totals,
            additional_labels=additional_labels,
            patient_additional_results=patient_additional_results  # NEW parameter for unique patient counts
        )
        ratio_obedience_metric = evaluate_ratio_obedience_metric(duration_info, partition_ratios)
        trials_results.append((ratio_obedience_metric, trial_random_state, duration_info))

        if n_trials is None:
            # If we're not using multiple trials, return immediately.
            partitioned_definition_objects = convert_source_lists_to_definitions(partitioned_source_list, definition)
            if verbose:
                print(f"Random State: {random_state}")
                pretty_print_duration_info(duration_info, header="Partition Duration Info",
                                           priority_labels=priority_stratification_labels,
                                           additional_labels=additional_labels)
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
            pretty_print_duration_info(trial_duration_info, header=f"Trial {i + 1} Partition Duration Info",
                                       priority_labels=priority_stratification_labels,
                                       additional_labels=additional_labels)

    # Select the best trial based on the metric.
    best_trial_random_state = best_trials[0][1] if best_trials else random_state

    # Rerun the partitioning using the best trial's random state.
    partitioned_source_list, partitioned_durations, partition_source_counts = stratified_partition_by_labels(
        label_duration_list, partition_ratios, random_state=best_trial_random_state)

    if additional_label_set_ids:
        additional_totals = []
        for partition in partitioned_source_list:
            total = np.zeros(len(additional_label_set_ids), dtype=np.int64)
            for item in partition:
                patient_id = item[1]
                if patient_id in patient_additional_results:
                    total += np.array(patient_additional_results[patient_id], dtype=np.int64)
            additional_totals.append(total)
    else:
        additional_totals = None

    # Convert the partitioned source lists into DatasetDefinition objects.
    final_partitioned_definition_objects = convert_source_lists_to_definitions(partitioned_source_list, definition)

    # Gather final information about the distribution of durations across partitions.
    final_duration_info = get_duration_info(
        partitioned_durations,
        priority_stratification_labels,
        partition_source_counts,
        partitioned_source_list=partitioned_source_list,
        additional_durations=additional_totals,
        additional_labels=additional_labels,
        patient_additional_results=patient_additional_results  # NEW parameter for unique patient counts
    )

    if verbose:
        print("Final partitioning using best trial's random state:")
        print(f"Random State: {best_trial_random_state}")
        pretty_print_duration_info(final_duration_info, header="Final Partition Duration Info",
                                   priority_labels=priority_stratification_labels,
                                   additional_labels=additional_labels)
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
        raise ValueError("Labels must be all integers or all strings.")

    if not all(label_id in validated_label_set_list for label_id in label_set_ids):
        raise ValueError("One or more labels are not in the validated label set list.")

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


def get_label_duration_list(validated_sources, priority_stratification_label_set_ids, sdk,
                            additional_label_set_ids=None):
    if "device_patient_tuples" not in validated_sources or not validated_sources["device_patient_tuples"]:
        raise ValueError(
            "There is no patient-mapped data in the dataset and so it is impossible to stratify by patient."
        )

    label_duration_list = []
    label_to_index_dict = {label: label_i for label_i, label in enumerate(priority_stratification_label_set_ids)}

    # Get all the priority label results
    label_result = sdk.sql_handler.select_labels(
        label_set_id_list=priority_stratification_label_set_ids,
    )

    # Group priority labels by device.
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

    # Sort the priority labels by time per device.
    label_starts = {}
    label_ends = {}
    for device_id in device_labels.keys():
        device_labels[device_id] = sorted(device_labels[device_id], key=lambda x: (x['start_time_n'], x['end_time_n']))
        label_starts[device_id] = [entry['start_time_n'] for entry in device_labels[device_id]]
        label_ends[device_id] = [entry['end_time_n'] for entry in device_labels[device_id]]

    # Process additional labels if provided.
    additional_device_labels = {}
    additional_label_starts = {}
    additional_label_ends = {}
    additional_label_to_index_dict = {}
    if additional_label_set_ids:
        additional_label_to_index_dict = {label: idx for idx, label in enumerate(additional_label_set_ids)}
        additional_label_result = sdk.sql_handler.select_labels(
            label_set_id_list=additional_label_set_ids,
        )
        for (label_entry_id, label_set_id, device_id, measure_id, label_source_id,
             start_time_n, end_time_n) in additional_label_result:
            if device_id not in additional_device_labels:
                additional_device_labels[device_id] = []
            formatted_label = {
                'label_entry_id': label_entry_id,
                'label_name_id': label_set_id,
                'device_id': device_id,
                'start_time_n': start_time_n,
                'end_time_n': end_time_n,
                'label_source_id': label_source_id,
                'measure_id': measure_id
            }
            additional_device_labels[device_id].append(formatted_label)
        for device_id in additional_device_labels.keys():
            additional_device_labels[device_id] = sorted(additional_device_labels[device_id],
                                                         key=lambda x: (x['start_time_n'], x['end_time_n']))
            additional_label_starts[device_id] = [entry['start_time_n'] for entry in
                                                  additional_device_labels[device_id]]
            additional_label_ends[device_id] = [entry['end_time_n'] for entry in additional_device_labels[device_id]]

    patient_results = {}
    # This will hold additional label durations per patient.
    patient_additional_results = {}

    for source_key, time_ranges in list(validated_sources["device_patient_tuples"].items()):
        device_id, patient_id = source_key
        if patient_id not in patient_results:
            patient_results[patient_id] = {
                "time_ranges": [],
                # durations[0] will be waveform duration; subsequent indices correspond to priority labels.
                "durations": [0] * (len(priority_stratification_label_set_ids) + 1)
            }
        # Initialize additional durations for this patient.
        if additional_label_set_ids and patient_id not in patient_additional_results:
            patient_additional_results[patient_id] = [0] * len(additional_label_set_ids)

        for start_time, end_time in time_ranges:
            # Append the time range.
            patient_results[patient_id]["time_ranges"].append((start_time, end_time))
            # Add waveform duration.
            patient_results[patient_id]["durations"][0] += (end_time - start_time)

            # Fetch priority labels within this time range.
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
            # Tally priority label durations.
            for label in range_labels:
                if label['label_name_id'] in label_to_index_dict:
                    label_index = label_to_index_dict[label['label_name_id']]
                    label_duration = label['end_time_n'] - label['start_time_n']
                    patient_results[patient_id]["durations"][label_index + 1] += label_duration

            # Fetch and tally additional labels if applicable.
            if additional_label_set_ids:
                if device_id not in additional_device_labels:
                    range_additional_labels = []
                else:
                    range_additional_labels = find_labels(
                        additional_device_labels[device_id],
                        additional_label_starts[device_id],
                        additional_label_ends[device_id],
                        start_time,
                        end_time
                    )
                for label in range_additional_labels:
                    if label['label_name_id'] in additional_label_to_index_dict:
                        label_index = additional_label_to_index_dict[label['label_name_id']]
                        label_duration = label['end_time_n'] - label['start_time_n']
                        patient_additional_results[patient_id][label_index] += label_duration

    # Merge and sort the time ranges per patient.
    for patient_id, result in patient_results.items():
        result["time_ranges"] = merge_intervals(result["time_ranges"])

    for patient_id, result in patient_results.items():
        label_duration_list.append(["patient_ids", patient_id, result["time_ranges"]] + result["durations"])

    return label_duration_list, patient_additional_results


def stratified_partition_by_labels(data_list, partition_ratios, random_state=None):
    data_list = copy.deepcopy(data_list)
    if any(ratio == 0 for ratio in partition_ratios):
        raise ValueError("Cannot have 0 in partition ratio.")

    if len(data_list) == 0:
        return [[] for _ in range(len(partition_ratios))], [[] for _ in range(len(partition_ratios))]
    # Set random state for reproducibility
    random_gen = random.Random(random_state) if random_state is not None else random.Random()

    # Calculate total sums for each duration (waveform + priority labels only)
    total_sums = np.array([sum(item[i] for item in data_list) for i in range(3, len(data_list[0]))])

    # Normalize the partition ratios
    ratios = np.array(partition_ratios) / sum(partition_ratios)
    target_sums = np.outer(total_sums, ratios).T

    # Initialize partitions and their total sums
    partitions = [[] for _ in range(len(ratios))]
    partition_sums = [np.zeros(len(total_sums), dtype=np.int64) for _ in range(len(ratios))]
    partition_source_counts = [0 for _ in range(len(ratios))]

    # Function to find the best partition for the current item.
    def find_best_partition(item):
        item_sums = np.array(item[3:], dtype=np.int64)
        deficits = target_sums - partition_sums
        deficits[deficits < 0] = np.inf
        scores = np.sum(deficits - item_sums, axis=1)
        return np.argmin(scores)

    # Randomly shuffle the data list.
    random_gen.shuffle(data_list)

    # Distribute items into partitions.
    for item in data_list:
        best_partition = find_best_partition(item)
        partitions[best_partition].append(item)
        partition_sums[best_partition] += item[3:]
        partition_source_counts[best_partition] += 1

    return partitions, partition_sums, partition_source_counts


def convert_source_lists_to_definitions(partitioned_source_list, original_definition):
    # Get measures and labels from the original definition.
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
                      partitioned_source_list=None, additional_durations=None, additional_labels=None,
                      patient_additional_results=None,
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
        # If partitioned_source_list is provided, compute unique patients.
        if partitioned_source_list is not None:
            unique_patients = len(set(item[1] for item in partitioned_source_list[partition_index]))
            duration_info["unique_patients"] = unique_patients

        # Add priority label durations.
        for label_index, label in enumerate(priority_stratification_labels, start=1):
            label_key = f"{label} {time_units}"
            duration_info[label_key] = round(durations[label_index] / (3600 * 1e9), 3) if convert_to_hours else \
            durations[label_index]

        # Add additional label durations if provided.
        if additional_durations is not None and additional_labels:
            add_dur = additional_durations[partition_index]
            for idx, label in enumerate(additional_labels):
                label_key = f"{label} (additional) {time_units}"
                val = round(add_dur[idx] / (3600 * 1e9), 3) if convert_to_hours else add_dur[idx]
                duration_info[label_key] = val

        # Add unique patient counts per priority label.
        if partitioned_source_list is not None:
            for label_index, label in enumerate(priority_stratification_labels, start=1):
                unique_patients_label = len(
                    {item[1] for item in partitioned_source_list[partition_index] if item[3 + label_index] > 0})
                duration_info[f"unique patients {label}"] = unique_patients_label

            # Add unique patient counts per additional label if available.
            if additional_labels and patient_additional_results is not None:
                for idx, label in enumerate(additional_labels):
                    unique_patients_additional = len({item[1] for item in partitioned_source_list[partition_index]
                                                      if item[1] in patient_additional_results and
                                                      patient_additional_results[item[1]][idx] > 0})
                    duration_info[f"unique patients {label} (additional)"] = unique_patients_additional

        duration_info_list.append(duration_info)

    return duration_info_list


def evaluate_ratio_obedience_metric(duration_info, target_ratios):
    total_target_ratio = sum(target_ratios)
    target_proportions = [r / total_target_ratio for r in target_ratios]
    error_sums = {key: 0 for key in duration_info[0] if key not in ['partition', 'num_sources', 'unique_patients']}

    num_partitions = len(duration_info)
    for key in error_sums:
        actual_durations = [partition[key] for partition in duration_info]
        total_duration = sum(actual_durations)
        expected_durations = [total_duration * prop for prop in target_proportions]
        for actual, expected in zip(actual_durations, expected_durations):
            error_sums[key] += (actual - expected) ** 2

    rmses = [np.sqrt(error_sum / num_partitions) for error_sum in error_sums.values()]
    combined_metric = np.mean(rmses)
    return combined_metric


def get_random_states(n_trials, random_state=None):
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


def pretty_print_duration_info(duration_info, header=None, priority_labels=None, additional_labels=None,
                               convert_to_hours=True):
    # Define fixed column widths.
    label_width = 50
    duration_width = 15
    unique_patients_width = 20
    # Compute overall line length for separators.
    line_length = label_width + duration_width + unique_patients_width

    # Determine the time unit string.
    time_unit = "hours" if convert_to_hours else "nanoseconds"

    if header:
        print("=" * line_length)
        print(header.center(line_length))
        print("=" * line_length)

    for partition in duration_info:
        partition_id = partition.get("partition", "N/A")
        print(f"\nPartition {partition_id}".center(line_length))
        # Print the header row for the table.
        header_label = "Label"
        header_duration = f"Duration ({time_unit})"
        header_unique = "Unique Patients"
        print(f"{header_label:<{label_width}}{header_duration:>{duration_width}}{header_unique:>{unique_patients_width}}")
        print("-" * line_length)

        total_priority_duration = 0
        # Print priority label durations and unique patient counts.
        if priority_labels:
            for label in priority_labels:
                duration_key = f"{label} {time_unit}"
                unique_patients_key = f"unique patients {label}"
                duration_val = partition.get(duration_key, 0)
                unique_patients_val = partition.get(unique_patients_key, 0)
                # Round duration values to 3 decimal places
                duration_val = round(duration_val, 3)
                print(f"{label:<{label_width}}{duration_val:>{duration_width}}{unique_patients_val:>{unique_patients_width}}")
                total_priority_duration += duration_val
            overall_priority_unique = partition.get("unique_patients", 0)
            print(f"{'Total Labelled Data':<{label_width}}{round(total_priority_duration, 3):>{duration_width}}{overall_priority_unique:>{unique_patients_width}}")

        # Print additional label durations and unique patient counts if provided.
        if additional_labels:
            print("\nAdditional Labels:")
            # Optionally print a header for additional labels too.
            print(f"{header_label:<{label_width}}{header_duration:>{duration_width}}{header_unique:>{unique_patients_width}}")
            print("-" * line_length)
            total_additional_duration = 0
            for label in additional_labels:
                duration_key = f"{label} (additional) {time_unit}"
                unique_patients_key = f"unique patients {label} (additional)"
                duration_val = partition.get(duration_key, 0)
                unique_patients_val = partition.get(unique_patients_key, 0)
                # Round duration values to 3 decimal places
                duration_val = round(duration_val, 3)
                print(f"{label:<{label_width}}{duration_val:>{duration_width}}{unique_patients_val:>{unique_patients_width}}")
                total_additional_duration += duration_val
            # Here, we print only the total duration for additional labels.
            print(f"{'Total Additional Labelled Data':<{label_width}}{round(total_additional_duration, 3):>{duration_width}}{'':>{unique_patients_width}}")

        # Print total waveform (raw) data info.
        waveform_key = f"total waveform {time_unit}"
        total_waveform = partition.get(waveform_key, 0)
        num_sources = partition.get("num_sources", 0)
        # Round total waveform duration to 3 decimal places
        total_waveform = round(total_waveform, 3)
        print("\n" + "-" * line_length)
        print(f"{'Total Data':<{label_width}}{total_waveform:>{duration_width}}{num_sources:>{unique_patients_width}}")
        print("=" * line_length)
