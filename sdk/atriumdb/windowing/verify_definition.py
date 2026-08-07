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

import warnings
from typing import List, Tuple, Dict

import yaml
import numpy as np

import pickle
import hashlib
import json

from atriumdb.adb_functions import get_measure_id_from_generic_measure
from atriumdb.intervals.union import intervals_union_list
from atriumdb.intervals.intersection import list_intersection
from atriumdb.windowing.map_definition_sources import map_validated_sources
from atriumdb.measure_kinds import is_string_value_type


def verify_definition(definition, sdk, gap_tolerance=None, measure_tag_match_rule="best", start_time_n=None,
                      end_time_n=None):
    """
    Verifies and validates a dataset definition against the given AtriumSDK, including measures, label sets, and sources.

    :param DatasetDefinition definition: The dataset definition to be verified. This can be either a filename (str) pointing to a dataset definition YAML file or a DatasetDefinition object.
    :param AtriumSDK sdk: An AtriumSDK object pointing at the dataset to validate the requested definition against.
    :param Optional[int] gap_tolerance: The minimum allowed gap (in nanoseconds) in any generated time ranges (time ranges are explained below).
    :param str measure_tag_match_rule: "best" or "all" as a strategy for dealing with measure tags where there may be multiple measures with the given tag.
    :param start_time_n: Global start time in nanoseconds.
    :param end_time_n: Global end time in nanoseconds.
    :return: A tuple containing three elements:
        1. validated_measure_list (list of dicts): A list of dictionaries, each representing a validated measure. Each dictionary includes:
           - 'id': The unique identifier of the measure.
           - 'tag': A string tag associated with the measure.
           - 'freq_nhz': The frequency of the measure in nanohertz (optional).
           - 'units': The units of the measure (optional).
        2. validated_label_set_list (list of int): A list of label set IDs that have been validated against the AtriumSDK. Each element is an integer representing the unique identifier of a label set.
        3. mapped_sources (dict): A dictionary representing the validated and mapped sources with the following keys:
           - 'device_patient_tuples': A dictionary where each key is a tuple (device_id, patient_id) and each value is a list of time ranges (start_time, end_time) for which the device-patient tuple has data. Both start_time and end_time are integers, representing nanosecond precision timestamps since the Unix epoch.
           - 'patient_ids' (optional): A dictionary present only if there are unmatched patient IDs. Each key is a patient ID, and each value is a list of time ranges where data could not be matched to devices.
           - 'device_ids' (optional): A dictionary present only if there are unmatched device IDs. Each key is a device ID, and each value is a list of time ranges where data could not be matched to patients.

    """
    gap_tolerance = 60_000_000_000 if gap_tolerance is None else gap_tolerance  # 1 minute nano default

    # Validate measures
    validated_measure_list = _validate_measures(definition, sdk, measure_tag_match_rule=measure_tag_match_rule)

    # Validate label sets
    validated_label_set_list = _validate_label_sets(definition, sdk)

    # Validate sources
    validated_sources = _validate_sources(definition, sdk, validated_measure_list, gap_tolerance=gap_tolerance,
                                          start_time_n=start_time_n, end_time_n=end_time_n)

    mapped_sources = map_validated_sources(validated_sources, sdk)

    result = (validated_measure_list, validated_label_set_list, mapped_sources)

    return result


def _validate_measures(definition, sdk, measure_tag_match_rule="best"):
    assert "measures" in definition.data_dict, "definition must have some specified measures"

    measures = definition.data_dict["measures"]

    validated_measure_list = []

    for measure_spec in measures:
        measure_ids = get_measure_id_from_generic_measure(sdk, measure_spec, measure_tag_match_rule=measure_tag_match_rule)

        for measure_id in measure_ids:
            if measure_id is None:
                raise ValueError(f"Measure {measure_spec} not found in AtriumSDK. Must use AtriumSDK.insert_measure "
                                 f"to insert the measure")

            measure_info = sdk.get_measure_info(measure_id)
            validated_measure_info = {
                'id': measure_id,
                'tag': measure_info['tag'],
                'freq_nhz': measure_info.get('freq_nhz'),
                'units': measure_info.get('unit'),
                'period_ns': measure_info.get('period_ns'),
                # Phase 2 metadata carried through so the iterator can pick the
                # Phase 3 nominal period + per-kind fill rule (read-time defaults
                # already applied by get_measure_info: waveform / numeric).
                'signal_kind': measure_info.get('signal_kind'),
                'value_type': measure_info.get('value_type'),
            }
            validated_measure_list.append(validated_measure_info)

    return validated_measure_list


def _validate_label_sets(definition, sdk):
    # If there aren't any labels, there's nothing to do.
    if "labels" not in definition.data_dict or len(definition.data_dict['labels']) == 0:
        return []

    labels = definition.data_dict["labels"]

    all_sdk_label_sets = sdk.get_all_label_names()
    all_sdk_label_set_name_to_id_dict = {
        label_info['name']: label_info['id'] for label_info in all_sdk_label_sets.values()}

    validated_label_set_list = []

    for label in labels:
        if label not in all_sdk_label_set_name_to_id_dict:
            raise ValueError(
                f"Label set {label} not found in AtriumSDK. Must use AtriumSDK.insert_label with a valid label if you "
                f"want use the label set {label}. If you don't have any valid labels, but still want"
                f"to include the label set use: AtriumSDK.sql_handler.insert_label_set(name) to introduce a new "
                f"label set.")
        validated_label_set_list.append(all_sdk_label_set_name_to_id_dict[label])

    return validated_label_set_list


def _validate_sources(definition, sdk, validated_measure_list, gap_tolerance=None, start_time_n=None,
                      end_time_n=None):
    data_dict = definition.data_dict

    validated_sources_dict = dict()
    for source_type, source_data in data_dict.items():
        if len(source_data) == 0:
            continue
        current_validated_source_dict = dict()
        if source_type == 'mrns':
            source_type = "patient_ids"
            mrn_to_patient_id_map = sdk.get_mrn_to_patient_id_map(list(source_data.keys()))
            for mrn, time_specs in source_data.items():
                if mrn in mrn_to_patient_id_map:
                    patient_id = mrn_to_patient_id_map[mrn]
                    validated_entries = _get_validated_entries(time_specs, validated_measure_list, sdk,
                                                               patient_id=patient_id, gap_tolerance=gap_tolerance, start_time_n=start_time_n, end_time_n=end_time_n)
                    if validated_entries is not None:  # Only add if it's not None
                        current_validated_source_dict[patient_id] = validated_entries
                else:
                    warnings.warn(f"MRN {mrn} not found in database, omitting from cohort data")

        elif source_type == 'patient_ids':
            all_patients = sdk.get_all_patients()
            for patient_id, time_specs in source_data.items():
                if patient_id in all_patients:
                    validated_entries = _get_validated_entries(time_specs, validated_measure_list, sdk,
                                                               patient_id=patient_id, gap_tolerance=gap_tolerance, start_time_n=start_time_n, end_time_n=end_time_n)
                    if validated_entries is not None:  # Only add if it's not None
                        current_validated_source_dict[patient_id] = validated_entries
                else:
                    warnings.warn(f"Patient ID {patient_id} not found in database, omitting from cohort data")

        elif source_type == 'device_tags':
            source_type = 'device_ids'
            all_devices = sdk.get_all_devices()
            tag_to_dev_id_map = {str(device_info['tag']): int(dev_id) for dev_id, device_info in all_devices.items()}
            for device_tag, time_specs in source_data.items():
                if str(device_tag) in tag_to_dev_id_map:
                    device_id = tag_to_dev_id_map[str(device_tag)]
                    validated_entries = _get_validated_entries(time_specs, validated_measure_list, sdk,
                                                               device_id=device_id, gap_tolerance=gap_tolerance, start_time_n=start_time_n, end_time_n=end_time_n)
                    if validated_entries is not None:  # Only add if it's not None
                        current_validated_source_dict[device_id] = validated_entries
                else:
                    warnings.warn(f"Device tag {device_tag} not found in database, omitting from cohort data")

        elif source_type == 'device_ids':
            all_device_ids = list(sdk.get_all_devices().keys())
            for device_id, time_specs in source_data.items():
                if device_id in all_device_ids:
                    validated_entries = _get_validated_entries(time_specs, validated_measure_list, sdk,
                                                               device_id=device_id, gap_tolerance=gap_tolerance, start_time_n=start_time_n, end_time_n=end_time_n)
                    if validated_entries is not None:  # Only add if it's not None
                        current_validated_source_dict[device_id] = validated_entries
                else:
                    warnings.warn(f"Device ID {device_id} not found in database, omitting from cohort data")

        elif source_type in ['measures', 'labels']:
            # Not a source type
            continue
        else:
            raise ValueError(f"Invalid source type {source_type}. Allowed types are: "
                             f"'mrns', 'patient_ids', 'device_tags', 'device_ids'.")

        # Update or create new source dictionary
        if source_type in validated_sources_dict:
            validated_sources_dict[source_type].update(current_validated_source_dict)
        else:
            validated_sources_dict[source_type] = current_validated_source_dict

    return validated_sources_dict


def _get_validated_entries(time_specs, validated_measures, sdk, device_id=None, patient_id=None, gap_tolerance=None,
                           start_time_n=None, end_time_n=None):
    gap_tolerance = 60 * 60 * 1_000_000_000 if gap_tolerance is None else gap_tolerance

    measure_interval_arrays = [
        (measure_info,
         sdk.get_interval_array(
             measure_info['id'], device_id=device_id, patient_id=patient_id,
             gap_tolerance_nano=gap_tolerance, start=start_time_n, end=end_time_n))
        for measure_info in validated_measures
    ]

    union_intervals = intervals_union_list([arr for _measure, arr in measure_interval_arrays])

    merged_union_intervals = []
    for start, end in union_intervals:
        if len(merged_union_intervals) > 0 and start - merged_union_intervals[-1][1] <= gap_tolerance:
            merged_union_intervals[-1][1] = end
        else:
            merged_union_intervals.append([start, end])

    union_intervals = np.array(merged_union_intervals, dtype=np.int64)

    if union_intervals.size == 0:
        source_type, source_id = ("device_id", device_id) if device_id is not None else ("patient_id", patient_id)
        warnings.warn(f"{source_type}: {source_id} could not be found over the requested "
                      f"time regions for the specified measures. Skipping")
        return None

    # Apply global bounds to ALL cases, including "all"
    if time_specs == "all":
        # Apply global start/end time constraints to union_intervals
        constrained_intervals = []
        for start, end in union_intervals:
            # Constrain each interval to global bounds
            if start_time_n is not None:
                start = max(start, start_time_n)
            if end_time_n is not None:
                end = min(end, end_time_n)

            # Only include if the interval is still valid after constraining
            if start < end:
                constrained_intervals.append([start, end])

        return constrained_intervals

    interval_list = []
    for region_data in time_specs:
        # Phase 5 event-anchored regions (design section 23): 'anchor' (X pre / Y post
        # around every occurrence of an event value) or 'from'/'to' (between an opening
        # event and its next closing event, via P4's get_event_intervals). These resolve
        # to a list of (start, end) windows -- already clipped to global bounds and
        # intersected with the source's data union -- exactly the shape the classic
        # branches below produce, so they flow through map_validated_sources/the iterator
        # unchanged. Branch on the event keys so the classic branches are untouched.
        if 'anchor' in region_data or 'from' in region_data:
            interval_list.extend(
                _resolve_event_region(region_data, sdk, device_id, patient_id,
                                      start_time_n, end_time_n, union_intervals,
                                      measure_interval_arrays))
            continue

        if 'time0' in region_data:
            start, end = region_data['time0'] - region_data['pre'], region_data['time0'] + region_data['post']
        elif 'start' in region_data and 'end' in region_data:
            start, end = region_data['start'], region_data['end']
        elif 'start' in region_data:
            start, end = region_data['start'], union_intervals[-1][1]
        elif 'end' in region_data:
            start, end = union_intervals[0][0], region_data['end']
        else:
            raise ValueError("time0, pre, post or start and/or end must be in the region specification.")

        if start_time_n is not None:
            start = max(start, start_time_n)
        if end_time_n is not None:
            end = min(end, end_time_n)

        if start < end:
            interval_list.append([start, end])

    return interval_list


def _merge_windows(windows):
    """Sort + merge overlapping/touching ``[start, end]`` windows into a disjoint,
    ascending list (the shape ``list_intersection`` requires)."""
    if not windows:
        return []
    ordered = sorted([int(s), int(e)] for s, e in windows)
    merged = [ordered[0]]
    for s, e in ordered[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return merged


def _observed_data_end(sdk, measure_interval_arrays, device_id, patient_id, indexed_end):
    """The last instant the definition's measures actually observed something.

    ``indexed_end`` is the end of the measures' interval-array union, and for
    every kind except ``event`` it is exactly right. An aperiodic block's index
    entry ends at ``last_sample_time + period``, which for a `sample` or `state`
    measure is meaningful (the reading stays in effect after its timestamp) but
    for an ``event`` measure is pure padding: an event is instantaneous, and the
    span between the final event and that padded end contains, by construction,
    no event at all.

    Where an event measure is what pushes the union to its end, that padding
    becomes a fabricated container boundary -- and ``on_censored='clip'`` clips a
    never-closed interval to it, producing a tail of windows past the end of the
    recording in which every channel is empty. (Observed: a 7200 s recording
    whose event stream's auto-detected period was 1800 s yielded 20 all-NaN ECG
    windows out to 8400 s.) So substitute each event measure's last real event
    timestamp for its padded index end; every other kind keeps its indexed end.

    Returns ``indexed_end`` unchanged when no event measure participates, when
    the event stream cannot be read, or when the trim would not move the bound.
    """
    from atriumdb.measure_kinds import SIGNAL_KIND_EVENT

    observed_end = None
    for measure_info, interval_array in measure_interval_arrays:
        arr = np.asarray(interval_array)
        if arr.size == 0:
            continue
        m_start, m_end = int(arr[0][0]), int(arr[-1][1])

        kind = sdk.get_measure_kind(measure_info['id'])
        if kind is not None and kind[0] == SIGNAL_KIND_EVENT:
            try:
                _headers, times, _values = sdk.get_data(
                    measure_id=int(measure_info['id']), start_time_n=m_start, end_time_n=m_end,
                    device_id=device_id, patient_id=patient_id, analog=False, time_units="ns")
            except Exception:
                # Never let the trim itself break validation -- fall back to the
                # indexed end, i.e. exactly the previous behaviour.
                times = None
            times = np.asarray(times, dtype=np.int64).reshape(-1) if times is not None else np.empty(0, np.int64)
            if times.size == 0:
                continue
            # +1 ns so the bound is an exclusive end that still includes the
            # final event itself.
            m_end = int(times.max()) + 1

        observed_end = m_end if observed_end is None else max(observed_end, m_end)

    if observed_end is None:
        return int(indexed_end)
    return min(int(indexed_end), int(observed_end))


def _resolve_event_region(region_data, sdk, device_id, patient_id, start_time_n, end_time_n, union_intervals,
                          measure_interval_arrays=None):
    """Resolve a Phase 5 event-anchored region (design section 23) to a list of
    ``[start, end]`` ns windows for one source.

    Two forms:

    * ``anchor``: for each occurrence of the ``anchor`` value in the event ``measure``
      for THIS source over the global bounds, emit ``[t - pre, t + post]`` (section
      23.1). The region's optional ``within`` scopes the emitted windows by
      intersecting them with the resolved container (section 23.2 #4); unscoped when no
      ``within`` is given.
    * ``from``/``to``: call P4's :meth:`AtriumSDK.get_event_intervals` (which applies the
      ``within`` cascade itself), then pad each interval by ``pre``/``post``, cap its
      length at ``max_duration`` if given, and handle censoring per ``on_censored``
      (default ``clip`` + warn; ``drop`` omits either-censored intervals; ``keep`` keeps
      them unchanged) -- section 23.2 #3.

    In both forms the windows are then clipped to the global bounds and intersected with
    the source's data union (``union_intervals``), exactly like the classic region
    branches. The event measure is NOT added to the definition's measures
    (anchor-only, section 23.2 #2).

    Validation (section 23.3), all raised at ``validate()`` time: an unknown event
    ``measure`` tag/id, a non-string measure, or an ``anchor``/``from``/``to`` value not
    in the measure's vocabulary. A source with no such events warns and contributes no
    ranges.
    """
    # Global read bounds: explicit global bounds when given, otherwise the data-union
    # extent (union_intervals is guaranteed non-empty by the caller).
    #
    # The end is the whole-stream container boundary that get_event_intervals clips a
    # censored interval to, so it must be a real edge of the recording. The raw union
    # end is not, whenever an EVENT measure's index padding is what reaches it --
    # see _observed_data_end. Clip it back to the last thing actually observed.
    global_start = int(start_time_n) if start_time_n is not None else int(union_intervals[0][0])
    if end_time_n is not None:
        global_end = int(end_time_n)
    elif measure_interval_arrays:
        global_end = _observed_data_end(sdk, measure_interval_arrays, device_id, patient_id,
                                        int(union_intervals[-1][1]))
    else:
        global_end = int(union_intervals[-1][1])
    if global_end <= global_start:
        global_end = int(union_intervals[-1][1])

    source_desc = (f"device_id {device_id}" if device_id is not None else f"patient_id {patient_id}")

    if 'measure' not in region_data:
        raise ValueError("An event-anchored region requires a 'measure' (the event/string measure tag or id).")
    measure_ref = region_data['measure']

    # Resolve the event measure tag/id against the dataset (section 23.2 #1). An unknown
    # tag/id raises here (get_measure_id_from_generic_measure -> "No matching measures").
    # For a TAG, prefer a STRING measure: the "best" rule ranks by block count and
    # ignores value_type, so a numeric measure sharing the tag could otherwise shadow a
    # valid string measure (P5 audit hazard). An int id / full spec is unambiguous.
    if isinstance(measure_ref, str):
        candidate_ids = [int(mid) for mid in
                         get_measure_id_from_generic_measure(sdk, measure_ref, measure_tag_match_rule="all")
                         if mid is not None]
        string_ids = [mid for mid in candidate_ids
                      if is_string_value_type(sdk.get_measure_kind(mid)[1])]
        if string_ids:
            best_id = int(get_measure_id_from_generic_measure(sdk, measure_ref, measure_tag_match_rule="best")[0])
            event_measure_id = best_id if best_id in string_ids else string_ids[0]
        elif candidate_ids:
            event_measure_id = candidate_ids[0]
        else:
            raise ValueError(f"No matching event measure for tag {measure_ref!r}.")
    else:
        event_measure_id = int(get_measure_id_from_generic_measure(sdk, measure_ref, measure_tag_match_rule="best")[0])
    # Reuse P4's guard: raises a clear error for a numeric (non-string) measure.
    string_dict = sdk._require_string_measure(event_measure_id)

    within = region_data.get('within', None)
    # Validate `within` up front so a bogus value raises deterministically -- even when a
    # source has zero occurrences (the anchor path used to return early before checking).
    if within is not None and within not in ("device_patient", "encounter", "none"):
        raise ValueError(
            f"Unknown within option {within!r}; expected None (cascade), "
            f"'device_patient', 'encounter', or 'none' (whole-stream).")
    windows = []

    if 'anchor' in region_data:
        anchor_value = region_data['anchor']
        # Vocabulary check (section 23.3): unknown anchor value -> raise at validate().
        if string_dict.code_for(anchor_value) is None:
            raise ValueError(
                f"anchor value {anchor_value!r} is not in the vocabulary of event measure "
                f"{measure_ref!r} (measure id {event_measure_id}). Known values come from "
                f"AtriumSDK.get_measure_string_vocabulary().")
        pre = int(region_data.get('pre', 0))
        post = int(region_data.get('post', 0))

        # Occurrence times for THIS source over the global bounds (reuse P4's read path).
        times, values = sdk.get_string_data(
            measure_id=event_measure_id, start_time_n=global_start, end_time_n=global_end,
            device_id=device_id, patient_id=patient_id, time_units="ns")
        occ_times = [int(t) for t, v in zip(times.tolist(), values.tolist()) if v == anchor_value]

        if len(occ_times) == 0:
            warnings.warn(
                f"{source_desc}: no occurrences of anchor value {anchor_value!r} in event measure "
                f"{measure_ref!r} over the requested bounds; this event region contributes no ranges.")
            return []

        windows = [[t - pre, t + post] for t in occ_times]

        # Honor the region's within by intersecting the emitted windows with the
        # resolved container (section 23.2 #4). from/to gets within via get_event_intervals.
        if within is not None:
            container_windows, _label = sdk._resolve_within_windows(
                within, device_id, patient_id, global_start, global_end)
            windows = list_intersection(_merge_windows(windows), _merge_windows(container_windows))

    else:
        from_value = region_data['from']
        to_value = region_data['to']
        pre = int(region_data.get('pre', 0))
        post = int(region_data.get('post', 0))
        max_duration = region_data.get('max_duration', None)
        on_censored = region_data.get('on_censored', 'clip')
        if on_censored not in ('clip', 'drop', 'keep'):
            raise ValueError(f"on_censored must be one of 'clip', 'drop', 'keep'; got {on_censored!r}.")

        # get_event_intervals validates the from/to vocabulary (raises on unknown values),
        # applies the within cascade, and clips censored ends to the container boundary.
        intervals = sdk.get_event_intervals(
            event_measure_id, from_value, to_value, device_id=device_id, patient_id=patient_id,
            start_time=global_start, end_time=global_end, within=within, time_units="ns")

        if len(intervals) == 0:
            warnings.warn(
                f"{source_desc}: no {from_value!r} -> {to_value!r} intervals in event measure "
                f"{measure_ref!r} over the requested bounds; this event region contributes no ranges.")
            return []

        any_censored = False
        for iv in intervals:
            s, e = int(iv['start_time_n']), int(iv['end_time_n'])
            if iv['start_censored'] or iv['end_censored']:
                any_censored = True
                if on_censored == 'drop':
                    continue
            # 'clip' (default) keeps the already-clipped interval; 'keep' keeps it too.
            s -= pre
            e += post
            if max_duration is not None and (e - s) > int(max_duration):
                e = s + int(max_duration)
            if s < e:
                windows.append([s, e])

        if on_censored == 'clip' and any_censored:
            warnings.warn(
                f"{source_desc}: one or more {from_value!r} -> {to_value!r} intervals were censored; "
                f"their censored ends were clipped to the container/range boundary (on_censored='clip').")

    if not windows:
        return []

    # Clip to global bounds, then intersect with the source's data union -- exactly the
    # constraint the classic region branches apply.
    clipped = []
    for s, e in windows:
        if start_time_n is not None:
            s = max(int(s), int(start_time_n))
        if end_time_n is not None:
            e = min(int(e), int(end_time_n))
        if s < e:
            clipped.append([int(s), int(e)])
    if not clipped:
        return []

    union_list = [[int(a), int(b)] for a, b in np.asarray(union_intervals).tolist()]
    return list_intersection(_merge_windows(clipped), union_list)


def compute_hash(data):
    """Compute a SHA256 hash of the given data."""
    data_string = json.dumps(data, sort_keys=True)
    return hashlib.sha256(data_string.encode()).hexdigest()

def compute_cache_key(definition_data_dict, gap_tolerance, measure_tag_match_rule, start_time_n, end_time_n):
    """Compute a unique cache key based on the function inputs."""

    # Compute a hash of the definition data dictionary
    definition_hash = compute_hash(definition_data_dict)

    # Prepare key elements
    key_elements = [
        definition_hash,
        str(gap_tolerance),
        measure_tag_match_rule,
        str(start_time_n),
        str(end_time_n)
    ]

    key_string = '|'.join(key_elements)
    key_hash = hashlib.sha256(key_string.encode()).hexdigest()

    return key_hash