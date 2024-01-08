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
from typing import List

import numpy as np
from tqdm import tqdm

from atriumdb.transfer.adb.labels import transfer_label_sets
from atriumdb.transfer.adb.patients import transfer_patient_info
from atriumdb.windowing.definition import DatasetDefinition
from atriumdb.atrium_sdk import AtriumSDK
from atriumdb.adb_functions import convert_value_to_nanoseconds
from atriumdb.transfer.adb.devices import transfer_devices
from atriumdb.transfer.adb.measures import transfer_measures
from atriumdb.windowing.verify_definition import verify_definition

MIN_TRANSFER_TIME = -(2**63)
MAX_TRANSFER_TIME = (2**63) - 1
DEFAULT_GAP_TOLERANCE = 30 * 24 * 60 * 60 * 1_000_000_000  # 1 month in nanoseconds.


def transfer_data(src_sdk: AtriumSDK, dest_sdk: AtriumSDK, definition: DatasetDefinition, gap_tolerance=None,
                  deidentify=True, patient_info_to_transfer=None):
    # Gap tolerance is used to construct the time ranges when the validator is asked to construct it itself.
    # Using a large value helps optimize the waveform transfer by transferring large chunks at a time.
    gap_tolerance = DEFAULT_GAP_TOLERANCE if gap_tolerance is None else gap_tolerance

    validated_measure_list, validated_label_set_list, validated_sources = verify_definition(
        definition, src_sdk, gap_tolerance=gap_tolerance)

    src_measure_id_list = [measure_info["id"] for measure_info in validated_measure_list]
    src_device_id_list, src_patient_id_list = extract_src_device_and_patient_id_list(validated_sources)

    measure_id_map = transfer_measures(src_sdk, dest_sdk, measure_id_list=src_measure_id_list)
    device_id_map = transfer_devices(src_sdk, dest_sdk, device_id_list=src_device_id_list)
    patient_id_map = transfer_patient_info(
        src_sdk, dest_sdk, patient_id_list=src_patient_id_list, deidentify=deidentify,
        patient_info_to_transfer=patient_info_to_transfer, start_time_nano=None, end_time_nano=None)
    label_set_id_map = transfer_label_sets(src_sdk, dest_sdk, label_set_id_list=validated_label_set_list)

    # Transfer Waveforms and Labels
    for source_type, sources in validated_sources.items():
        # For each source identifier of that type
        for source_id, time_ranges in tqdm(list(sources.items())):
            dest_device_id, src_device_id = extract_device_ids(source_id, source_type, device_id_map)

            if src_device_id is None:
                continue

            # Simple for now. Optimized by a large gap_tolerance.
            # Might want to aggregate reads and writes in the future.
            for start_time_nano, end_time_nano in time_ranges:
                for src_measure_id, dest_measure_id in measure_id_map.items():
                    # Insert Waveforms
                    headers, times, values = src_sdk.get_data(
                        src_measure_id, start_time_nano, end_time_nano, device_id=src_device_id, time_type=1,
                        analog=False, sort=False, allow_duplicates=True)

                    if values.size == 0:
                        continue

                    ingest_data(dest_sdk, dest_measure_id, dest_device_id, headers, times, values)

                # Insert labels
                if len(label_set_id_map) == 0:
                    continue
                labels = src_sdk.get_labels(
                    label_set_id_list=list(label_set_id_map.keys()), device_list=[src_device_id],
                    start_time=start_time_nano, end_time=end_time_nano)

                insert_labels(dest_sdk, labels, device_id_map)


def insert_labels(dest_sdk, labels, device_id_map):
    if len(labels) == 0:
        return

    dest_labels = []
    for label in labels:
        label_dest_device_id = device_id_map[label['device_id']]
        label_name = label['label_name']
        label_start_time_n = label['start_time_n']
        label_end_time_n = label['end_time_n']

        dest_labels.append([label_name, label_dest_device_id, label_start_time_n, label_end_time_n])
    dest_sdk.insert_labels(dest_labels, source_type="device_id")


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


def old_transfer_data(from_sdk: AtriumSDK, to_sdk: AtriumSDK, measure_id_list: List[int] = None,
                      device_id_list: List[int] = None, patient_id_list: List[int] = None, mrn_list: List[int] = None,
                      start: int = None, end: int = None, time_units: str = None, batch_size=None,
                      include_patient_context=False, deidentify=None, time_shift=None):
    """
    Transfers data from one dataset to another. If measure_id_list, device_id_list, patient_id_list, mrn_list, start,
    end are all None, then all data is transferred, otherwise these parameters serve to limit the data transferred.
    """

    batch_size = 40 if batch_size is None else batch_size
    time_units = 'ns' if time_units is None else time_units

    if start is not None:
        start = convert_value_to_nanoseconds(start, time_units)

    if end is not None:
        end = convert_value_to_nanoseconds(end, time_units)

    # Transfer measures
    transfer_measures(from_sdk, to_sdk, measure_id_list=measure_id_list)

    if include_patient_context:
        device_patient_list = from_sdk.get_device_patient_data(
            device_id_list=device_id_list, patient_id_list=patient_id_list, mrn_list=mrn_list,
            start_time=start, end_time=end)

        device_id_list = list(set([row[0] for row in device_patient_list]))

        # Transfer devices
        transfer_devices(from_sdk, to_sdk, device_id_list=device_id_list)

        # Transfer patients
        patient_id_map = transfer_patients(
            from_sdk, to_sdk, patient_id_list=patient_id_list, mrn_list=mrn_list, deidentify=deidentify)

        patient_id_idx = 1
        if deidentify:
            for i in range(len(device_patient_list)):
                device_patient_list[i] = list(device_patient_list[i])
                device_patient_list[i][patient_id_idx] = patient_id_map[device_patient_list[i][patient_id_idx]]

        start_time_idx = 2
        end_time_idx = 3
        if isinstance(time_shift, int):
            for i in range(len(device_patient_list)):
                device_patient_list[i] = list(device_patient_list[i])
                device_patient_list[i][start_time_idx] -= time_shift
                device_patient_list[i][end_time_idx] -= time_shift

        # Transfer device_patients
        to_sdk.insert_device_patient_data(device_patient_data=device_patient_list)

        for measure_id, measure_info in tqdm(from_sdk.get_all_measures().items()):
            to_measure_id = to_sdk.get_measure_id(measure_tag=measure_info['tag'],
                                                  freq=measure_info['freq_nhz'],
                                                  units=measure_info['unit'], )
            if measure_id_list is not None and measure_id not in measure_id_list:
                continue

            for from_device_id, patient_id, start_time, end_time in tqdm(device_patient_list, leave=False):
                if end_time is None:
                    continue
                # un-time-shift start, end
                if isinstance(time_shift, int):
                    start_time += time_shift
                    end_time += time_shift
                device_info = from_sdk.get_device_info(from_device_id)
                to_device_id = to_sdk.get_device_id(device_tag=device_info['tag'])
                try:
                    headers, times, values = from_sdk.get_data(measure_id, start_time, end_time,
                                                               device_id=from_device_id, analog=False)
                except Exception as e:
                    print(e)
                    continue

                if len(headers) == 0:
                    continue

                if isinstance(time_shift, int):
                    shift_times(times, time_shift)

                ingest_data(to_sdk, to_measure_id, to_device_id, headers, times, values)

    else:
        # Transfer devices
        transfer_devices(from_sdk, to_sdk, device_id_list=device_id_list)
        for measure_id, measure_info in tqdm(from_sdk.get_all_measures().items()):
            to_measure_id = to_sdk.get_measure_id(measure_tag=measure_info['tag'],
                                                  freq=measure_info['freq_nhz'],
                                                  units=measure_info['unit'],)

            if measure_id_list is not None and measure_id not in measure_id_list:
                print("continue")
                continue

            for from_device_id, device_info in tqdm(from_sdk.get_all_devices().items(), leave=False):
                to_device_id = to_sdk.get_device_id(device_tag=device_info['tag'])
                if device_id_list is not None and from_device_id not in device_id_list:
                    continue

                block_list = from_sdk.get_block_id_list(int(measure_id), start_time_n=start,
                                                        end_time_n=end, device_id=from_device_id)

                if len(block_list) == 0:
                    continue

                file_id_list = list(set([row[3] for row in block_list]))
                filename_dict = from_sdk.get_filename_dict(file_id_list)

                start_block = 0
                while start_block < len(block_list):
                    block_batch = block_list[start_block:start_block+batch_size]
                    try:
                        headers, times, values = from_sdk.get_data_from_blocks(block_batch, filename_dict, measure_id,
                                                                               MIN_TRANSFER_TIME, MAX_TRANSFER_TIME,
                                                                               analog=False)
                    except Exception as e:
                        print(e)
                        start_block += batch_size
                        raise e

                    if isinstance(time_shift, int):
                        shift_times(times, time_shift)

                    ingest_data(to_sdk, to_measure_id, to_device_id, headers, times, values)
                    start_block += batch_size


def ingest_data(to_sdk, measure_id, device_id, headers, times, values):
    try:
        if all([h.scale_m == headers[0].scale_m and
                h.scale_b == headers[0].scale_b and
                h.freq_nhz == headers[0].freq_nhz for h in headers]):

            to_sdk.write_data(measure_id, device_id, times, values, headers[0].freq_nhz, int(times[0]),
                              raw_time_type=headers[0].t_raw_type, raw_value_type=headers[0].v_raw_type,
                              encoded_time_type=headers[0].t_encoded_type, encoded_value_type=headers[0].v_encoded_type,
                              scale_m=headers[0].scale_m, scale_b=headers[0].scale_b, interval_index_mode="fast")
        else:
            print("Differently configured headers detected.")
            val_index = 0
            for h in headers:
                try:
                    to_sdk.write_data(measure_id, device_id, times, values, h.freq_nhz, int(times[0]),
                                      raw_time_type=h.t_raw_type, raw_value_type=h.v_raw_type,
                                      encoded_time_type=h.t_encoded_type,
                                      encoded_value_type=h.v_encoded_type,
                                      scale_m=h.scale_m, scale_b=h.scale_b,
                                      interval_index_mode="fast")
                except IndexError:
                    continue
                val_index += h.num_vals
    except Exception as e:
        print(e)
        raise e


def transfer_patients(from_sdk, to_sdk, patient_id_list=None, mrn_list=None, deidentify=None):
    deidentify = False if deidentify is None else deidentify

    if mrn_list is not None:
        patient_id_list = [] if patient_id_list is None else patient_id_list
        mrn_to_patient_id_map = from_sdk.get_mrn_to_patient_id_map(mrn_list)
        patient_id_list.extend([mrn_to_patient_id_map[mrn] for mrn in mrn_list if mrn in mrn_to_patient_id_map])

    from_patients = from_sdk.get_all_patients()

    from_patients_items = list(from_patients.items())
    if deidentify:
        random.shuffle(from_patients_items)

    from_to_patient_id_dict = {}
    for from_patient_id, patient_info in from_patients_items:
        if patient_id_list is not None and from_patient_id not in patient_id_list:
            continue

        if deidentify:
            to_patient_id = to_sdk.sql_handler.insert_patient()
        else:
            to_patient_id = to_sdk.sql_handler.insert_patient(patient_id=patient_info['id'],
                                                              mrn=patient_info['mrn'],
                                                              gender=patient_info['gender'],
                                                              dob=patient_info['dob'],
                                                              first_name=patient_info['first_name'],
                                                              middle_name=patient_info['middle_name'],
                                                              last_name=patient_info['last_name'],
                                                              first_seen=patient_info['first_seen'],
                                                              last_updated=patient_info['last_updated'],
                                                              source_id=patient_info['source_id'],
                                                              weight=patient_info['weight'],
                                                              height=patient_info['height'])

        from_to_patient_id_dict[from_patient_id] = to_patient_id

    return from_to_patient_id_dict


def shift_times(times: np.ndarray, shift_amount: int):
    times -= shift_amount


    def parse_device_ids(source_id, source_type, device_id_map):
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