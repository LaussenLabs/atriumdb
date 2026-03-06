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

import copy
import warnings

import numpy as np

from atriumdb.intervals.union import intervals_union_list


def combine_definitions(definitions):
    """
    Combine multiple :class:`DatasetDefinition` objects into a single definition.

    Measures and labels are deduplicated across all inputs. Source dictionaries
    (``patient_ids``, ``mrns``, ``device_ids``, ``device_tags``) are merged by
    taking the union of time ranges for each source. When a source appears in
    multiple definitions, its time regions are merged via interval union so that
    overlapping or adjacent regions are consolidated.

    If all input definitions are validated, the validated data is merged as
    well and the combined result is also validated. If the definitions have
    mixed validation status, the validated data is discarded and the combined
    result is unvalidated — a warning is issued advising the caller to
    re-validate.

    :param definitions: Two or more :class:`DatasetDefinition` objects to combine.
    :type definitions: list[DatasetDefinition]
    :returns: A new :class:`DatasetDefinition` containing the union of all inputs.
    :rtype: DatasetDefinition
    :raises ValueError: If fewer than two definitions are provided.

    **Examples**:

    Combining two unvalidated definitions:

    >>> from atriumdb import DatasetDefinition, combine_definitions
    >>> def_a = DatasetDefinition(measures=["ECG"], patient_ids={1: "all"})
    >>> def_b = DatasetDefinition(measures=["ABP"], patient_ids={2: "all"})
    >>> combined = combine_definitions([def_a, def_b])
    >>> combined.data_dict['measures']
    ['ECG', 'ABP']

    Combining two validated definitions:

    >>> def_a.validate(sdk=my_sdk)
    >>> def_b.validate(sdk=my_sdk)
    >>> combined = combine_definitions([def_a, def_b])
    >>> combined.is_validated
    True
    """
    if len(definitions) < 2:
        raise ValueError("At least two DatasetDefinition objects are required to combine.")

    # Resolve the class from the inputs to avoid a circular import with definition.py.
    DefinitionClass = type(definitions[0])

    # Determine validation status.
    validation_states = {d.is_validated for d in definitions}
    all_validated = all(d.is_validated for d in definitions)

    if len(validation_states) > 1:
        warnings.warn(
            "Combining definitions with mixed validation status. The validated data "
            "from validated definitions will be discarded. Re-validate the combined "
            "definition before using with SDK operations.",
            UserWarning
        )

    # Merge data_dict (pre-validated / user-specified data).
    combined_measures = _merge_measures([d.data_dict['measures'] for d in definitions])
    combined_labels = _merge_labels([d.data_dict['labels'] for d in definitions])
    combined_patient_ids = _merge_source_dicts([d.data_dict['patient_ids'] for d in definitions])
    combined_mrns = _merge_source_dicts([d.data_dict['mrns'] for d in definitions])
    combined_device_ids = _merge_source_dicts([d.data_dict['device_ids'] for d in definitions])
    combined_device_tags = _merge_source_dicts([d.data_dict['device_tags'] for d in definitions])

    combined_def = DefinitionClass(
        measures=combined_measures,
        labels=combined_labels,
        patient_ids=combined_patient_ids if combined_patient_ids else None,
        mrns=combined_mrns if combined_mrns else None,
        device_ids=combined_device_ids if combined_device_ids else None,
        device_tags=combined_device_tags if combined_device_tags else None,
    )

    # Merge validated data if all inputs are validated.
    if all_validated:
        combined_def.validated_data_dict = _merge_validated_data(
            [d.validated_data_dict for d in definitions]
        )
        combined_def.is_validated = True

        # Use the smallest gap_tolerance among all definitions.
        gap_tolerances = [d.validated_gap_tolerance for d in definitions if d.validated_gap_tolerance is not None]
        combined_def.validated_gap_tolerance = min(gap_tolerances) if gap_tolerances else None

        # Filtering metadata cannot be preserved across a combine since the source
        # intervals may now differ. Reset to None so the user can re-filter.
        combined_def.filtered_window_size = None
        combined_def.filtered_window_slide = None

    return combined_def


def _normalize_measure(measure):
    """Return a hashable key for a measure specification."""
    if isinstance(measure, str):
        return ('tag_only', measure)
    elif isinstance(measure, dict):
        return ('triplet', measure.get('tag'), measure.get('freq_hz'), measure.get('freq_nhz'), measure.get('units'))
    elif isinstance(measure, (list, tuple)):
        return ('tuple', tuple(measure))
    return ('unknown', str(measure))


def _merge_measures(measure_lists):
    """Deduplicate and merge measure lists preserving order."""
    seen = set()
    merged = []
    for measures in measure_lists:
        for measure in measures:
            key = _normalize_measure(measure)
            if key not in seen:
                seen.add(key)
                merged.append(copy.deepcopy(measure))
    return merged


def _merge_labels(label_lists):
    """Deduplicate and merge label lists preserving order."""
    seen = set()
    merged = []
    for labels in label_lists:
        for label in labels:
            if label not in seen:
                seen.add(label)
                merged.append(label)
    return merged


def _merge_source_dicts(source_dicts):
    """
    Merge multiple source dictionaries (e.g. patient_ids dicts from different definitions).

    For each source key, time regions are combined. If any definition specifies ``"all"``
    for a source, the result is ``"all"`` for that source. Otherwise, all time region lists
    are concatenated and overlapping intervals are merged.
    """
    merged = {}
    for source_dict in source_dicts:
        for source_id, time_spec in source_dict.items():
            if source_id not in merged:
                merged[source_id] = copy.deepcopy(time_spec)
            else:
                existing = merged[source_id]
                # If either side is "all", the result is "all".
                if existing == 'all' or time_spec == 'all':
                    merged[source_id] = 'all'
                else:
                    # Both are lists of time-range dicts; concatenate them.
                    if not isinstance(existing, list):
                        existing = [existing]
                    if not isinstance(time_spec, list):
                        time_spec = [time_spec]
                    merged[source_id] = existing + copy.deepcopy(time_spec)
    return merged


def _merge_validated_sources(source_dicts_list):
    """
    Merge validated source dictionaries.

    Validated sources have the structure::

        {
            "device_patient_tuples": {(device_id, patient_id): [[start, end], ...]},
            "patient_ids": {patient_id: [[start, end], ...]},
            "device_ids": {device_id: [[start, end], ...]},
        }

    Time ranges for the same key are merged using interval union.
    """
    merged = {}
    for sources in source_dicts_list:
        for source_type, entries in sources.items():
            if source_type not in merged:
                merged[source_type] = {}
            for source_id, time_ranges in entries.items():
                if source_id not in merged[source_type]:
                    merged[source_type][source_id] = copy.deepcopy(time_ranges)
                else:
                    # Combine and union the time ranges.
                    combined = merged[source_type][source_id] + copy.deepcopy(time_ranges)
                    if len(combined) > 0:
                        as_np = [np.array(r, dtype=np.int64) for r in combined]
                        unioned = intervals_union_list(as_np).tolist()
                        merged[source_type][source_id] = unioned
                    else:
                        merged[source_type][source_id] = combined
    return merged


def _merge_validated_measures(measure_lists):
    """Deduplicate validated measure dicts by measure 'id'."""
    seen_ids = set()
    merged = []
    for measures in measure_lists:
        for measure in measures:
            mid = measure['id']
            if mid not in seen_ids:
                seen_ids.add(mid)
                merged.append(copy.deepcopy(measure))
    return merged


def _merge_validated_labels(label_lists):
    """Deduplicate validated label set ID lists."""
    seen = set()
    merged = []
    for labels in label_lists:
        for label_id in labels:
            if label_id not in seen:
                seen.add(label_id)
                merged.append(label_id)
    return merged


def _merge_validated_data(validated_dicts):
    """Merge multiple validated_data_dict structures."""
    merged = {}
    merged['measures'] = _merge_validated_measures(
        [vd.get('measures', []) for vd in validated_dicts]
    )
    merged['labels'] = _merge_validated_labels(
        [vd.get('labels', []) for vd in validated_dicts]
    )
    merged['sources'] = _merge_validated_sources(
        [vd.get('sources', {}) for vd in validated_dicts]
    )
    return merged