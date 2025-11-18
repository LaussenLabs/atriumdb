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

import random
import time
from collections import defaultdict
from pathlib import Path
from typing import List

import numpy as np
from tqdm import tqdm

from atriumdb.block_wrapper import BlockMetadata
from atriumdb.intervals.union import intervals_union_list
from atriumdb.transfer.adb.csv import _write_csv
from atriumdb.transfer.adb.datestring_conversion import nanoseconds_to_date_string_with_tz
from atriumdb.transfer.adb.definition import create_dataset_definition_from_verified_data
from atriumdb.transfer.adb.labels import transfer_label_sets
from atriumdb.transfer.adb.numpy import _write_numpy
from atriumdb.transfer.adb.parquet import _write_parquet
from atriumdb.transfer.adb.patients import transfer_patient_info
from atriumdb.transfer.adb.tsc import _ingest_data_tsc
from atriumdb.transfer.adb.wfdb import _ingest_data_wfdb
from atriumdb.windowing.definition import DatasetDefinition
from atriumdb.atrium_sdk import AtriumSDK
from atriumdb.adb_functions import convert_value_to_nanoseconds, condense_byte_read_list, get_block_and_interval_data, \
    group_sorted_block_list, reconstruct_messages_multi, sort_message_time_values, truncate_messages, \
    create_gap_arr_from_variable_messages
from atriumdb.transfer.adb.devices import transfer_devices
from atriumdb.transfer.adb.measures import transfer_measures
from atriumdb.windowing.verify_definition import verify_definition

MIN_TRANSFER_TIME = -(2 ** 63)
MAX_TRANSFER_TIME = (2 ** 63) - 1
DEFAULT_GAP_TOLERANCE = 5 * 60 * 1_000_000_000  # 5 minutes in nanoseconds
time_unit_options = {"ns": 1, "s": 10 ** 9, "ms": 10 ** 6, "us": 10 ** 3}


def transfer_data(src_sdk: AtriumSDK, dest_sdk: AtriumSDK, definition: DatasetDefinition, export_format='tsc',
                  start_time=None, end_time=None, gap_tolerance=None, deidentify=False, patient_info_to_transfer=None,
                  include_labels=True, measure_tag_match_rule=None, deidentification_functions=None, time_shift=None,
                  time_units=None, export_time_format=None, parquet_engine=None, timezone_str=None,
                  reencode_waveforms=False, **kwargs):
    """
    Transfers data from a source AtriumSDK instance to a destination AtriumSDK instance based on a specified dataset definition.
    This includes transferring measures, devices, patient information, and labels with options for data de-identification,
    time shifting, and custom gap tolerance.

    :param AtriumSDK src_sdk: The source SDK instance from which data will be transferred.
    :param AtriumSDK dest_sdk: The destination SDK instance to which data will be transferred.
    :param DatasetDefinition definition: Specifies the structure and contents of the dataset to be transferred.
    :param str export_format: The format used for exporting data ('tsc' by default). Supported formats include 'tsc', 'csv', 'npz', 'parquet', and 'wfdb'.
    :param start_time: A global start time for the transfer, units specified in `time_units`.
    :param end_time: A global end time for the transfer, units specified in `time_units`.
    :param Optional[int] gap_tolerance: A tolerance period for gaps in data, units specified in `time_units` (defaults to 5 minutes if not specified).
        Helps to optimize the waveform transfer by transferring large chunks at a time.
    :param bool deidentify: If True or a filename, scrambles patient_ids during the transfer. patient IDs are replaced with randomly generated IDs or according to provided de-identification csv
        with source ids as column 1 and destination ids as column 2. If the file doesn't exist, then a new one is created with randomly assigned ids. Exported datasets will remain identified with full patient info by default.
    :param Optional[list] patient_info_to_transfer: Specific patient information fields to transfer. If not provided or set to None, all available information will be considered.
    :param bool include_labels: Specifies whether labels should be included in the transfer process.
    :param Optional[str] measure_tag_match_rule: Determines how to match the measures by tags. One of ['all', 'best'].
    :param Optional[dict] deidentification_functions: Custom functions for de-identifying specific patient information fields.
        A dictionary where keys are the patient_info or patient_history field to be altered and values are the functions that alter them.
        Example: `{'height': lambda x: x + random.uniform(-1.5, 1.5)}`
    :param Optional[int] time_shift: An amount of time by which to shift all timestamps in the transferred data, specified in `time_units`.
    :param Optional[str] time_units: Units for `gap_tolerance` and `time_shift`. Supported units are 'ns' (nanoseconds), 's' (seconds), 'ms' (milliseconds), and 'us' (microseconds). Defaults to 'ns'.
    :param Optional[str] export_time_format: The format for timestamps in the exported data. Supports 'ns', 's', 'ms', 'us', and 'date'. Defaults to 'ns'.
    :param Optional[str] parquet_engine: Specifies the engine to use for writing Parquet files. Can be 'fastparquet' or 'pyarrow'.
        'fastparquet' - uses fastparquet to write DataFrame directly.
        'pyarrow' - uses pyarrow to create a Table from data and write it to a Parquet file.
        If None, the default engine installed will be used. The specific engine affects how the Parquet files are handled and can be influenced by additional kwargs.
    :param Optional[str] timezone_str: The timezone to use for the conversion. Default is 'Etc/GMT'.
        Valid values are any timezone strings recognized by the `zoneinfo` module. Examples include 'America/New_York',
        'Asia/Tokyo', 'Europe/London', etc. For a complete list of valid timezones, refer to the IANA time zone database.
    :param Optional[bool] reencode_waveforms: Specifies whether to reencode data into newly encoded blocks.
        (Default False) Setting to False will reuse existing blocks where possible and significantly speed up transfer.
        Setting to True allows you to change the block size and in general will reorder the blocks by time, which can
        speed up datasets that were originally ingested in small or unordered chunks.

    Examples:
    ---------
    Create a dataset definition:

    >>> measures = [{"tag": "ECG", "freq_hz": 300, "units": "mV"}]
    >>> labels = ["Sinus rhythm", "Atrial fibrillation"]
    >>> device_ids = {1: "all"}
    >>> my_definition = DatasetDefinition(measures=measures, labels=labels, device_ids=device_ids)

    Transfer all available data with default parameters, de-identifying patient information:

    >>> transfer_data(src_sdk=my_src_sdk,dest_sdk=my_dest_sdk,definition=my_definition,deidentify=True)

    Transfer data with a specific gap tolerance of one day and without including labels:

    >>> transfer_data(src_sdk=my_src_sdk,dest_sdk=my_dest_sdk,definition=my_definition,include_labels=False)

    Transfer data with a two-hour time shift applied to the entire dataset, and use custom de-identification functions:

    >>> time_shift = 2*60*60  # 2 hours in seconds
    >>> my_deid_funcs = {'height': lambda x: x + random.uniform(-1.5, 1.5)}
    >>> transfer_data(src_sdk=my_src_sdk,dest_sdk=my_dest_sdk,definition=my_definition,deidentify=True,deidentification_functions=my_deid_funcs,time_shift=time_shift,time_units='s')

    """

    # Set defaults for time units and export format
    time_units = "ns" if time_units is None else time_units
    export_time_format = "ns" if export_time_format is None else export_time_format

    # Determine if analog values and duplicates are allowed based on export format
    analog_values = export_format != 'tsc'
    allow_duplicates = export_format == 'tsc'

    # Validate provided time units
    if time_units not in time_unit_options.keys():
        raise ValueError("Invalid time units. Expected one of: %s" % time_unit_options)

    # convert time values to nanoseconds
    start_time_n, end_time_n = start_time, end_time
    if start_time_n is not None:
        start_time_n = int(start_time_n * time_unit_options[time_units])

    if end_time_n is not None:
        end_time_n = int(end_time_n * time_unit_options[time_units])

    if gap_tolerance is not None:
        gap_tolerance = int(gap_tolerance * time_unit_options[time_units])

    if time_shift is not None:
        time_shift = int(time_shift * time_unit_options[time_units])

    # Gap tolerance is used to construct the time ranges when the validator is asked to construct it itself.
    # Using a large value helps optimize the waveform transfer by transferring large chunks at a time.
    gap_tolerance = DEFAULT_GAP_TOLERANCE if gap_tolerance is None else gap_tolerance
    measure_tag_match_rule = "all" if measure_tag_match_rule is None else measure_tag_match_rule

    # Validate the dataset definition if not already done
    if not definition.is_validated:
        definition.validate(sdk=src_sdk, gap_tolerance=gap_tolerance,
                            measure_tag_match_rule=measure_tag_match_rule, start_time=start_time_n,
                            end_time=end_time_n)

    # Extract validated info
    validated_measure_list = definition.validated_data_dict['measures']
    validated_label_set_list = definition.validated_data_dict['labels']
    validated_sources = definition.validated_data_dict['sources']

    src_measure_id_list = [measure_info["id"] for measure_info in validated_measure_list]
    src_device_id_list, src_patient_id_list = extract_src_device_and_patient_id_list(validated_sources)

    # Transfer measure, devices, patients between SDKs
    measure_id_map = transfer_measures(src_sdk, dest_sdk, measure_id_list=src_measure_id_list,
                                       measure_tag_match_rule=measure_tag_match_rule)
    device_id_map = transfer_devices(src_sdk, dest_sdk, device_id_list=src_device_id_list)
    patient_id_map = transfer_patient_info(
        src_sdk, dest_sdk, patient_id_list=src_patient_id_list, deidentify=deidentify,
        patient_info_to_transfer=patient_info_to_transfer, start_time_nano=start_time_n, end_time_nano=end_time_n,
        deidentification_functions=deidentification_functions, time_shift_nano=time_shift)

    if include_labels:
        label_set_id_map = transfer_label_sets(src_sdk, dest_sdk, label_set_id_list=validated_label_set_list)
    else:
        label_set_id_map = {}

    file_path_dicts = {}

    for source_type, sources in validated_sources.items():
        for source_id, time_ranges in tqdm(list(sources.items())):
            dest_device_id, src_device_id = extract_device_ids(source_id, source_type, device_id_map)

            if src_device_id is None:
                continue

            if export_format == "tsc":
                for src_measure_id, dest_measure_id in measure_id_map.items():
                    seen_block_ids = set()
                    blocks_to_transfer = []
                    time_range_index_to_transfer = []
                    re_encode_block_bool_list = []
                    measure_device_intervals = []

                    for time_range_i, (start_time_nano, end_time_nano) in enumerate(time_ranges):
                        # Finding all blocks that match the given time ranges
                        if src_sdk.mode == "api":
                            params = {
                                "start_time": int(start_time_nano),
                                "end_time": int(end_time_nano),
                                "measure_id": int(src_measure_id),
                                "device_id": int(src_device_id),
                            }
                            block_list = src_sdk._request("GET", "sdk/blocks", params=params)
                            block_list.sort(key=lambda b: b["start_time_n"])
                        else:
                            block_list = src_sdk.sql_handler.select_blocks(
                                int(src_measure_id), int(start_time_nano), int(end_time_nano), int(src_device_id), None)
                            block_list.sort(key=lambda x: x[6])

                        # Filter out duplicates
                        filtered_blocks = []
                        for block in block_list:
                            block_id = block["id"] if src_sdk.mode == "api" else block[0]
                            if block_id not in seen_block_ids:
                                seen_block_ids.add(block_id)
                                filtered_blocks.append(block)

                        # Extend the main list with unique blocks
                        blocks_to_transfer.extend(filtered_blocks)

                        # Decide which blocks need to be reencoded.
                        if not reencode_waveforms:
                            for block in block_list:
                                block_s = block["start_time_n"] if src_sdk.mode == "api" else block[6]
                                block_e = block["end_time_n"] if src_sdk.mode == "api" else block[7]
                                if start_time_nano <= block_s and block_e <= end_time_nano:
                                    time_range_index_to_transfer.append(-1)
                                    re_encode_block_bool_list.append(False)
                                else:
                                    re_encode_block_bool_list.append(True)
                        else:
                            re_encode_block_bool_list.extend([True] * len(block_list))

                        time_range_index_to_transfer.extend([time_range_i] * len(block_list))

                        # Grab the interval data for transferring into the new sdk.
                        write_intervals = src_sdk.get_interval_array(
                            measure_id=src_measure_id, device_id=src_device_id,
                            start=int(start_time_nano), end=int(end_time_nano))

                        measure_device_intervals.append(write_intervals)

                    if not blocks_to_transfer:
                        continue

                    block_start_index = 0
                    # Break up  the list of total blocks to transfer into manageable block groups.
                    for block_group in group_sorted_block_list(
                            blocks_to_transfer, num_values_per_group=dest_sdk.block.block_size * 2048,
                            src_sdk_mode=src_sdk.mode):

                        group_re_encode_flags = re_encode_block_bool_list[block_start_index:block_start_index + len(block_group)]
                        block_time_range_indices = time_range_index_to_transfer[
                            block_start_index:block_start_index + len(block_group)]
                        block_start_index += len(block_group)
                        if len(block_group) == 0:
                            continue

                        # Read the bytes for the block group.
                        if src_sdk.mode == "api":
                            encoded_bytes = src_sdk._block_websocket_request(block_group)
                        else:
                            file_id_list = list({row[3] for row in block_group})
                            filename_dict = src_sdk.get_filename_dict(file_id_list)
                            read_list = condense_byte_read_list(block_group)
                            encoded_bytes = src_sdk.file_api.read_file_list(read_list, filename_dict)

                        groups_to_reencode = {}
                        blocks_to_hold = []

                        start_byte = 0
                        # Read the block headers and organize block information.
                        for block_i, (time_range_i, block, re_encode_flag) in enumerate(
                                zip(block_time_range_indices, block_group, group_re_encode_flags)):
                            block_num_bytes = block["num_bytes"] if src_sdk.mode == "api" else block[5]
                            block_bytes = encoded_bytes[start_byte:start_byte + block_num_bytes]
                            header = BlockMetadata.from_buffer(block_bytes, 0)

                            scale_m = header.scale_m
                            scale_b = header.scale_b
                            freq_nhz_or_period = int(
                                header.freq_nhz)  # This could be freq_nhz or period_ns depending on version
                            t_encoded_type = int(header.t_encoded_type)

                            # Calculate version as 10 * num + ext
                            version = 10 * header.tsc_version_num + header.tsc_version_ext

                            # Determine if header.freq_nhz represents frequency or period based on version
                            if version >= 24:  # Version 2.4 and above
                                group_freq_nhz = None
                                group_period_ns = freq_nhz_or_period
                            else:
                                group_freq_nhz = freq_nhz_or_period
                                group_period_ns = None

                            key = (
                            time_range_i, scale_m, scale_b, group_freq_nhz, group_period_ns, t_encoded_type, version)

                            if not re_encode_flag and time_shift is not None and t_encoded_type == 1:
                                re_encode_flag = True

                            if not re_encode_flag:
                                # Pass encoded blocks directly to be written.
                                if time_shift is not None:
                                    header.start_n = header.start_n + time_shift
                                    header.end_n = header.end_n + time_shift
                                blocks_to_hold.append({'block_bytes': block_bytes, 'header': header})
                            else:
                                # Pass encoded blocks to be re-encoded before written.
                                if key not in groups_to_reencode:
                                    groups_to_reencode[key] = {
                                        'block_bytes_list': [], 'block_num_bytes_list': [],
                                        'blocks': [], 'headers': []}
                                groups_to_reencode[key]['block_bytes_list'].append(block_bytes)
                                groups_to_reencode[key]['block_num_bytes_list'].append(block_num_bytes)
                                groups_to_reencode[key]['blocks'].append(block)
                                groups_to_reencode[key]['headers'].append(header)

                            start_byte += block_num_bytes

                        # Process each group for reencoding
                        segments = []
                        for key, group_data in groups_to_reencode.items():
                            time_range_i, scale_m, scale_b, group_freq_nhz, group_period_ns, t_encoded_type, version = key
                            encode_group_lower_bound, encode_group_upper_bound = time_ranges[time_range_i]

                            block_bytes_list = group_data['block_bytes_list']
                            block_num_bytes_list = group_data['block_num_bytes_list']
                            group_headers = group_data['headers']

                            # Concatenate all block bytes into one array
                            encode_group_bytes = np.concatenate(block_bytes_list)

                            # Decode the blocks to get times and values
                            encode_group_times, encode_group_values, _ = src_sdk.block.decode_blocks(
                                encode_group_bytes, block_num_bytes_list, analog=False, time_type='encoded')

                            if np.issubdtype(encode_group_values.dtype, np.integer):
                                raw_value_type = 1
                                encoded_value_type = 3
                            else:
                                raw_value_type = 2
                                encoded_value_type = 2

                            group_time_type = t_encoded_type
                            if group_time_type == 1:
                                # Time type is 1
                                encode_group_times, sorted_time_indices = np.unique(encode_group_times,
                                                                                    return_index=True)
                                encode_group_values = encode_group_values[sorted_time_indices]

                                # Truncate data
                                left = np.searchsorted(encode_group_times, encode_group_lower_bound, side='left')
                                right = np.searchsorted(encode_group_times, encode_group_upper_bound, side='left')
                                encode_group_times = encode_group_times[left:right]
                                encode_group_values = encode_group_values[left:right]
                                if encode_group_times.size == 0:
                                    continue

                                if time_shift is not None:
                                    encode_group_times += time_shift

                                encode_group_start_time = int(encode_group_times[0])
                            elif group_time_type == 2:
                                # Time type is 2
                                message_starts, message_sizes = reconstruct_messages_multi(
                                    group_headers, encode_group_times, freq_nhz=group_freq_nhz,
                                    period_ns=group_period_ns)
                                sort_message_time_values(message_starts, message_sizes, encode_group_values)

                                # Truncate the message data
                                encode_group_values, message_starts, message_sizes = truncate_messages(
                                    encode_group_values, message_starts, message_sizes, freq_nhz=group_freq_nhz,
                                    period_ns=group_period_ns, trunc_start_nano=encode_group_lower_bound,
                                    trunc_end_nano=encode_group_upper_bound)

                                if len(message_starts) == 0:
                                    continue

                                # Revert back to gap array
                                gap_data = create_gap_arr_from_variable_messages(
                                    message_starts, message_sizes, freq_nhz=group_freq_nhz, period_ns=group_period_ns)
                                encode_group_times = gap_data

                                encode_group_start_time = int(message_starts[0])

                                if time_shift is not None:
                                    encode_group_start_time += time_shift
                            else:
                                raise ValueError("time_type must be 1 or 2")

                            segment = {
                                'times': encode_group_times,
                                'values': encode_group_values,
                                'freq_nhz': group_freq_nhz,
                                'period_ns': group_period_ns,
                                'start_ns': encode_group_start_time,
                                'raw_time_type': t_encoded_type,
                                'encoded_time_type': t_encoded_type,
                                'raw_value_type': raw_value_type,
                                'encoded_value_type': encoded_value_type,
                                'scale_m': scale_m,
                                'scale_b': scale_b
                            }
                            segments.append(segment)

                        # Combine reencoded blocks and held blocks
                        all_blocks_bytes = []
                        all_blocks_headers = []

                        if segments:
                            # Re-encode the blocks that needed re-encoding.
                            reencoded_bytes, reencoded_headers, reencoded_byte_start_array = dest_sdk.block.encode_blocks_from_multiple_segments(segments)

                            # Add re-encoded blocks to total blocks to be written.
                            for i in range(len(reencoded_headers)):
                                header = reencoded_headers[i]
                                start_byte = reencoded_byte_start_array[i]
                                end_byte = reencoded_byte_start_array[i + 1] if i < len(reencoded_headers) - 1 else len(reencoded_bytes)

                                block_bytes = reencoded_bytes[start_byte:end_byte]

                                all_blocks_bytes.append(block_bytes)
                                all_blocks_headers.append(header)

                        # Add blocks that didn't need re-encoding.
                        for block_info in blocks_to_hold:
                            block_bytes = block_info['block_bytes']
                            header = block_info['header']
                            all_blocks_bytes.append(block_bytes)
                            all_blocks_headers.append(header)

                        if not all_blocks_bytes:
                            continue

                        paired = sorted(zip(all_blocks_headers, all_blocks_bytes), key=lambda hb: hb[0].start_n)
                        total_recombined_headers, sorted_block_bytes = zip(*paired)
                        total_recombined_bytes = np.concatenate(sorted_block_bytes, dtype=np.uint8)

                        filename = dest_sdk.file_api.write_bytes(dest_measure_id, dest_device_id, total_recombined_bytes)

                        sql_total_block_data = []
                        start_byte = 0
                        for header in total_recombined_headers:
                            sql_total_block_data.append({
                                "measure_id": dest_measure_id,
                                "device_id": dest_device_id,
                                "start_byte": start_byte,
                                "num_bytes": int(header.meta_num_bytes + header.t_num_bytes + header.v_num_bytes),
                                "start_time_n": header.start_n,
                                "end_time_n": header.end_n,
                                "num_values": header.num_vals,
                            })
                            start_byte += int(header.meta_num_bytes + header.t_num_bytes + header.v_num_bytes)

                        # Insert block + file data into sql table.
                        dest_sdk.sql_handler.insert_tsc_file_blocks(filename, sql_total_block_data)

                    # Recombine data intervals
                    total_recombined_data_intervals = intervals_union_list(measure_device_intervals, gap_tolerance_nano=0)

                    if time_shift is not None:
                        total_recombined_data_intervals += time_shift

                    # Get interval sql data
                    sql_total_interval_data = []
                    for interval in total_recombined_data_intervals:
                        sql_total_interval_data.append({
                            "measure_id": dest_measure_id,
                            "device_id": dest_device_id,
                            "start_time_n": int(interval[0]),
                            "end_time_n": int(interval[1]),
                        })
                    # Insert Interval data into sql table
                    dest_sdk.sql_handler.insert_intervals(sql_total_interval_data)

            # Simple for now. Optimized by a large gap_tolerance.
            # Might want to aggregate reads and writes in the future.
            for start_time_nano, end_time_nano in time_ranges:
                file_path_dicts[(source_type, source_id, start_time_nano, end_time_nano)] = {}
                for src_measure_id, dest_measure_id in measure_id_map.items():
                    # Insert Waveforms
                    if export_format == "tsc":
                        break
                    else:
                        headers, times, values = src_sdk.get_data(
                            src_measure_id, start_time_nano, end_time_nano, device_id=src_device_id, time_type=1,
                            analog=analog_values, sort=True, allow_duplicates=allow_duplicates)

                        for h_i in range(len(headers)):
                            headers[h_i].t_raw_type = 1

                        if values.size == 0:
                            continue

                        if time_shift is not None:
                            times += time_shift

                        file_path = ingest_data(dest_sdk, dest_measure_id, dest_device_id, headers, times, values,
                                                export_format=export_format, export_time_format=export_time_format,
                                                parquet_engine=parquet_engine, timezone_str=timezone_str, **kwargs)

                        if file_path is not None:
                            file_path_dicts[(source_type, source_id, start_time_nano, end_time_nano)][
                                dest_measure_id] = file_path

                if not include_labels:
                    continue

                # Insert labels
                if len(label_set_id_map) == 0:
                    continue
                labels = src_sdk.get_labels(
                    label_name_id_list=list(label_set_id_map.keys()), device_list=[src_device_id],
                    start_time=start_time_nano, end_time=end_time_nano)

                if len(labels) == 0:
                    continue

                if time_shift is not None:
                    for label_dict in labels:
                        label_dict['start_time_n'] += time_shift
                        label_dict['end_time_n'] += time_shift

                # Make the list of label tuples to insert to the other dataset
                dest_labels = [(label['label_name'], device_id_map[label['device_id']],
                                measure_id_map[label['measure_id']], label['start_time_n'], label['end_time_n'],
                                label['label_source_id']) for label in labels]

                dest_sdk.insert_labels(dest_labels, source_type="device_id")

    # Create new definition file
    export_definition = create_dataset_definition_from_verified_data(
        src_sdk, validated_measure_list, validated_sources, validated_label_set_list=validated_label_set_list,
        prefer_patient=True, patient_id_map=patient_id_map, file_path_dicts=file_path_dicts, time_shift_nano=time_shift)

    definition_path = Path(dest_sdk.dataset_location) / "meta" / "definition.yaml"
    definition_path.parent.mkdir(parents=True, exist_ok=True)
    export_definition.save(definition_path, force=True)

def extract_device_ids(source_id, source_type, device_id_map):
    if source_type == "device_ids":
        src_device_id = source_id
    elif source_type == "patient_ids":
        # Mapping failed
        src_device_id = None
    elif source_type == "device_patient_tuples":
        src_device_id, _ = source_id
    else:
        raise ValueError(f"Source type must be either device_ids, device_patient_tuples or "
                         f"patient_ids, not {source_type}")
    dest_device_id = device_id_map.get(src_device_id)
    return dest_device_id, src_device_id


def extract_src_device_and_patient_id_list(validated_sources):
    src_device_id_list = []
    src_patient_id_list = []
    for source_type, sources in validated_sources.items():
        # For each source identifier of that type
        for source_id, time_ranges in sources.items():
            if source_type == "device_ids":
                device_id = source_id
                patient_id = None
            elif source_type == "patient_ids":
                device_id = None
                patient_id = source_id
            elif source_type == "device_patient_tuples":
                device_id, patient_id = source_id
            else:
                raise ValueError(f"Source type must be either device_ids or patient_ids, not {source_type}")
            if device_id is not None:
                src_device_id_list.append(device_id)
            if patient_id is not None:
                src_patient_id_list.append(patient_id)

    src_device_id_list = list(set(src_device_id_list))
    src_patient_id_list = list(set(src_patient_id_list))
    return src_device_id_list, src_patient_id_list


def ingest_data(to_sdk, measure_id, device_id, headers, times, values, export_format='tsc', export_time_format=None,
                parquet_engine=None, timezone_str=None, **kwargs):
    # Determine the file path based on the format
    measure_info = to_sdk.get_measure_info(measure_id)
    measure_tag = measure_info['tag']
    freq_hz = measure_info['freq_nhz'] / (10 ** 9)
    measure_units = measure_info['unit']
    measure_folder_name = f"{measure_tag}^{freq_hz}Hz^{measure_units}".replace(".", "_")

    device_info = to_sdk.get_device_info(device_id)
    device_tag = device_info['tag']
    device_folder_name = str(device_tag)

    base_path = Path(to_sdk.dataset_location) / export_format / device_folder_name / measure_folder_name
    file_name = None
    if export_format != 'tsc':
        base_path.mkdir(parents=True, exist_ok=True)
        file_name = str(nanoseconds_to_date_string_with_tz(int(times[0]), timezone_str=timezone_str)).replace(".", "f").replace(":", "-")
    file_path = None

    if export_format == 'tsc':
        _ingest_data_tsc(to_sdk, measure_id, device_id, headers, times, values)
    elif export_format == 'csv':
        file_path = base_path / f"{file_name}.csv"
        _write_csv(file_path, times, values, measure_tag, export_time_format=export_time_format, timezone_str=timezone_str)
    elif export_format == 'npz':
        file_path = base_path / f"{file_name}.npz"
        _write_numpy(file_path, times, values, measure_tag)
    elif export_format == 'wfdb':
        file_path = base_path / file_name  # WFDB format uses multiple files with the same base name
        _ingest_data_wfdb(headers, times, values, file_path, measure_tag, freq_hz, measure_units)
    elif export_format == 'parquet':
        file_path = base_path / f"{file_name}.parquet"
        _write_parquet(file_path, times, values, measure_tag, engine=parquet_engine, **kwargs)
    else:
        raise ValueError(f"Unsupported format {export_format}")

    # Return the relative path to to_sdk.dataset_location
    relative_path = str(file_path.relative_to(to_sdk.dataset_location)) if file_path is not None else None
    return relative_path
