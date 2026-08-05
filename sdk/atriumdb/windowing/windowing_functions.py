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
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from bisect import bisect_right

from atriumdb.windowing.window import Window
from atriumdb.string_dictionary import UNKNOWN_STRING_CODE

# 1 second nominal raster period, used when an aperiodic measure has no usable
# grid period (design section 21.2 #1).
ONE_SECOND_NS = 1_000_000_000

# Per-signal_kind default fill rules (design section 21.3).
_DEFAULT_FILL_FOR_KIND = {
    "waveform": "grid",          # existing NaN grid -- numeric path, untouched
    "sample": "carry_forward",
    "state": "carry_forward",    # carry-forward with left-censoring
    "event": "presence",
}

# Which fill rules each signal_kind accepts. "grid" is the waveform-numeric
# path and is never selected for aperiodic kinds.
_RULES_FOR_KIND = {
    "waveform": ("grid",),
    "sample": ("carry_forward", "sparse",
               "aggregate:last", "aggregate:mean", "aggregate:min", "aggregate:max"),
    "state": ("carry_forward",),
    "event": ("presence", "count"),
}


def default_fill_for_kind(signal_kind):
    """The default fill rule for a signal_kind (design section 21.3)."""
    return _DEFAULT_FILL_FOR_KIND.get(signal_kind, "carry_forward")


def resolve_fill_rule(signal_kind, value_type, override=None, global_default=None):
    """Resolve the effective fill rule for one measure.

    Precedence (design section 21.2 #3): a per-measure ``override``
    (``fill_overrides[measure_id]``) wins and is validated strictly (an
    incompatible rule for the kind raises). Otherwise a ``global_default``
    (``get_iterator(aperiodic_fill=...)``) is applied when it is *compatible*
    with the kind, else the per-kind default is used. ``waveform`` always
    resolves to the untouched ``grid`` path.

    String value_type additionally forbids numeric reductions
    (``aggregate:mean|min|max``) -- only ``aggregate:last`` makes sense for
    codes.
    """
    if signal_kind == "waveform":
        return "grid"

    allowed = _RULES_FOR_KIND.get(signal_kind, ())

    def _reject_string_numeric_agg(rule):
        if value_type == "string" and rule in ("aggregate:mean", "aggregate:min", "aggregate:max"):
            raise ValueError(
                f"Fill rule '{rule}' is numeric and cannot apply to a string measure; "
                f"only 'aggregate:last' (or 'carry_forward'/'sparse') is valid for string codes.")

    if override is not None:
        if override not in allowed:
            raise ValueError(
                f"Fill rule '{override}' is not valid for a '{signal_kind}' measure. "
                f"Allowed: {allowed}.")
        _reject_string_numeric_agg(override)
        return override

    if global_default is not None and global_default in allowed:
        try:
            _reject_string_numeric_agg(global_default)
            return global_default
        except ValueError:
            # A global default that is incompatible with this particular
            # (string) measure silently falls back to the per-kind default;
            # only an explicit per-measure override raises.
            pass

    return default_fill_for_kind(signal_kind)


def resolve_nominal_period_ns(measure, period_override=None):
    """Resolve a measure's nominal raster period (design section 21.2 #1).

    Order: ``get_iterator`` override -> ``measure['period_ns']`` for waveform ->
    1 s default for aperiodic kinds. The freq-derived ``period_ns`` carried on an
    aperiodic measure is *not* a meaningful raster period, so aperiodic kinds
    grid at 1 s unless explicitly overridden."""
    if period_override is not None:
        return int(period_override)
    signal_kind = measure.get("signal_kind") or "waveform"
    if signal_kind == "waveform":
        return int(measure["period_ns"])
    return ONE_SECOND_NS


def _rasterize_grid(grid_times, period_ns, sample_times, sample_values, rule, is_string,
                    seed_time=None, seed_value=None):
    """Rasterize sparse readings onto a regular grid per one fill ``rule``.

    Returns ``(values, known)`` where ``known`` is a *separable* internal boolean
    array -- "is this cell a genuine, observed value" -- exactly the seam design
    section 21.2 #2(b) requires so a real per-signal ``known`` mask can later be
    emitted additively. The caller applies the sentinel; here we already fold the
    sentinel into ``values`` (NaN for float channels, the reserved
    ``UNKNOWN_STRING_CODE`` for string/int64 channels) using ``known``.

    ``grid_times`` is the evenly spaced grid (cell i spans
    ``[grid_times[i], grid_times[i] + period_ns)``). ``sample_times`` must be
    ascending (as returned by ``get_data(sort=True)``).

    ``seed_time`` / ``seed_value`` (carry-forward only) is the single most-recent
    reading that occurred *before* this batch grid but at/after the definition's
    range start. It makes carry-forward deterministic and batch-size independent:
    cells before the first in-batch reading carry the seed value instead of being
    left-censored. It is always earlier than ``grid_times[0]`` and must respect
    the range floor (the caller bounds the lookback at ``range_start``).
    """
    n = int(grid_times.shape[0])

    # event presence/count is always numeric 0/1 (or a count); absence is a
    # meaningful 0, so every cell is "known" and there is no sentinel.
    if rule in ("presence", "count"):
        values = np.zeros(n, dtype=np.float64)
        known = np.ones(n, dtype=bool)
        if n == 0 or sample_times.size == 0:
            return values, known
        grid_start = grid_times[0]
        cell = ((np.asarray(sample_times, dtype=np.int64) - grid_start) // period_ns).astype(np.int64)
        m = (cell >= 0) & (cell < n)
        cell = cell[m]
        counts = np.bincount(cell, minlength=n)[:n]
        values = (counts > 0).astype(np.float64) if rule == "presence" else counts.astype(np.float64)
        return values, known

    # Non-event kinds: unknown cells carry a sentinel.
    if is_string:
        values = np.full(n, UNKNOWN_STRING_CODE, dtype=np.int64)
    else:
        values = np.full(n, np.nan, dtype=np.float64)
    known = np.zeros(n, dtype=bool)

    # Normalize the sample arrays to the channel's target dtype (int64 codes for
    # string channels, float64 for numeric).
    target_dtype = np.int64 if is_string else np.float64
    st = np.asarray(sample_times, dtype=np.int64)
    sv = np.asarray(sample_values).astype(target_dtype, copy=False)

    # Carry-forward seed (design Bug-1 fix): prepend the pre-batch reading so
    # cells before the first in-batch reading carry the value already in effect.
    # The seed is always earlier than grid_times[0], so after this merge every
    # grid cell has a prior reading and carry-forward is batch-size independent.
    if rule == "carry_forward" and seed_time is not None:
        st = np.concatenate((np.asarray([seed_time], dtype=np.int64), st))
        sv = np.concatenate((np.asarray([seed_value], dtype=target_dtype), sv))

    if n == 0 or st.size == 0:
        return values, known

    grid_start = grid_times[0]

    if rule == "carry_forward":
        # Each cell takes the most recent reading at or before its start time.
        # Cells before the first reading (and with no seed) are left-censored ->
        # sentinel (known stays False). This is the state/sample carry-forward
        # machinery.
        idx = np.searchsorted(st, grid_times, side="right") - 1
        valid = idx >= 0
        sel = np.where(valid, idx, 0)
        picked = sv[sel]
        if is_string:
            values = np.where(valid, picked, UNKNOWN_STRING_CODE)
        else:
            values = np.where(valid, picked, np.nan)
        known = valid
        return values, known

    # sparse / aggregate: bucket readings by the cell they fall in.
    cell = ((st - grid_start) // period_ns).astype(np.int64)
    m = (cell >= 0) & (cell < n)
    c = cell[m]
    v = sv[m]
    if c.size == 0:
        return values, known

    if rule in ("sparse", "aggregate:last"):
        # Later (more recent) readings in the same cell overwrite earlier ones.
        if is_string:
            values[c] = v.astype(np.int64)
        else:
            values[c] = v.astype(np.float64)
        known[c] = True
        return values, known

    # Numeric reductions (guarded to numeric measures by resolve_fill_rule).
    vf = v.astype(np.float64)
    counts = np.bincount(c, minlength=n)[:n]
    nz = counts > 0
    if rule == "aggregate:mean":
        sums = np.bincount(c, weights=vf, minlength=n)[:n]
        values[nz] = sums[nz] / counts[nz]
    elif rule == "aggregate:min":
        acc = np.full(n, np.inf)
        np.minimum.at(acc, c, vf)
        values[nz] = acc[nz]
    elif rule == "aggregate:max":
        acc = np.full(n, -np.inf)
        np.maximum.at(acc, c, vf)
        values[nz] = acc[nz]
    known[nz] = True
    return values, known


def get_threshold_labels(sliced_labels, label_threshold=0.5):
    # Calculate the percentage of 1s for each label in each window
    percentages = np.mean(sliced_labels, axis=-1)
    # Apply threshold
    return (percentages > label_threshold).astype(int)


def find_closest_measurement(time, measurements):
    """
    Find the measurement with the time value closest, but less than or equal to the given time,
    directly using bisect on the list of tuples.

    :param time: The time (epoch timestamp) to find the closest measurement for.
    :param measurements: A list of tuples containing the measurement value, units, and epoch timestamp.
    :return: The tuple from the measurements list with the closest time less than or equal to the given time.
    """
    # Use bisect_right with a key function that extracts the timestamp part of the tuple
    idx = bisect_right(measurements, time, key=lambda x: x[5])

    if idx > 0:
        return measurements[idx - 1]
    else:
        return None


def get_signal_dictionary(sdk, device_id, query_patient_id, window_duration_ns, window_slide_ns, measures,
                          batch_start_time, batch_end_time, batch_num_windows, range_start_time, range_end_time,
                          render_config=None, definition_range_start_time=None):
    """Build the per-measure sliding windows for one batch.

    ``render_config`` (Phase 3) maps ``measure_id -> {signal_kind, value_type,
    period_ns, fill_rule, is_string}``. When it is ``None`` (older iterators /
    the definition-filter path) or a measure resolves to the untouched
    waveform-numeric case, the exact legacy NaN-grid path runs -- this keeps the
    numeric windowing path byte-for-byte identical. Aperiodic / string measures
    take the per-kind fill path instead.
    """
    # Reset and populate the batch data signal dictionary
    source_batch_data_dictionary = {}
    for i, measure in enumerate(measures):
        measure_id = measure['id']
        cfg = render_config.get(measure_id) if render_config else None

        # ---- Legacy waveform-numeric path (unchanged, byte-for-byte) -------- #
        if cfg is None or cfg['fill_rule'] == 'grid':
            period_ns = measure['period_ns'] if cfg is None else cfg['period_ns']

            # Create a time array for this specific measure
            measure_window_size = int(window_duration_ns // period_ns)
            measure_slide_size = int(window_slide_ns // period_ns)
            measure_batch_size = measure_window_size + (batch_num_windows - 1) * measure_slide_size
            measure_quantized_end_time = batch_start_time + (measure_batch_size * period_ns)
            measure_filled_time_array = np.arange(batch_start_time, measure_quantized_end_time, period_ns)
            measure_filled_value_array = np.full(measure_filled_time_array.shape, np.nan)

            # If partial windows are allowed, we need to make room for an extra full window,
            # but then only partially populate it. So find just the region where we actually want data
            data_start_time = max(range_start_time, batch_start_time)
            data_end_time = min(range_end_time, batch_end_time)

            start_index = np.searchsorted(measure_filled_time_array, data_start_time, side='left')

            expected_num_values = int(round((data_end_time - data_start_time) / period_ns))
            if expected_num_values > measure_filled_value_array.size - start_index:
                data_end_time = data_start_time + int(
                    round((measure_filled_value_array.size - start_index) * period_ns))
                expected_num_values = int(round((data_end_time - data_start_time) / period_ns))

            nan_filled_out = measure_filled_value_array[start_index:start_index + expected_num_values]

            if expected_num_values > 0:
                sdk.get_data(
                    measure_id, data_start_time, data_end_time, device_id=device_id, patient_id=query_patient_id,
                    return_nan_filled=nan_filled_out)

            # Create Windows
            windowed_measure_times = sliding_window_view(measure_filled_time_array, measure_window_size)
            windowed_measure_values = sliding_window_view(measure_filled_value_array, measure_window_size)

            # Slide the windows
            sliced_windowed_measure_times = windowed_measure_times[::measure_slide_size]
            sliced_windowed_measure_values = windowed_measure_values[::measure_slide_size]

            # Store the measure's time and value arrays in the batch data dictionary
            source_batch_data_dictionary[measure_id] = \
                (sliced_windowed_measure_times, sliced_windowed_measure_values, measure_window_size)
            continue

        # ---- Phase 3 aperiodic / string fill path -------------------------- #
        period_ns = cfg['period_ns']
        rule = cfg['fill_rule']
        is_string = cfg['is_string']

        measure_window_size = int(window_duration_ns // period_ns)
        measure_slide_size = int(window_slide_ns // period_ns)
        measure_batch_size = measure_window_size + (batch_num_windows - 1) * measure_slide_size
        measure_quantized_end_time = batch_start_time + (measure_batch_size * period_ns)
        measure_filled_time_array = np.arange(batch_start_time, measure_quantized_end_time, period_ns)

        # Whole-batch value array pre-filled with the sentinel (or 0 for event
        # presence/count, where absence is a meaningful 0 -- no sentinel).
        if rule in ("presence", "count"):
            measure_filled_value_array = np.zeros(measure_filled_time_array.shape, dtype=np.float64)
        elif is_string:
            measure_filled_value_array = np.full(measure_filled_time_array.shape, UNKNOWN_STRING_CODE, dtype=np.int64)
        else:
            measure_filled_value_array = np.full(measure_filled_time_array.shape, np.nan, dtype=np.float64)

        data_start_time = max(range_start_time, batch_start_time)
        data_end_time = min(range_end_time, batch_end_time)

        start_index = np.searchsorted(measure_filled_time_array, data_start_time, side='left')

        expected_num_values = int(round((data_end_time - data_start_time) / period_ns))
        if expected_num_values > measure_filled_value_array.size - start_index:
            data_end_time = data_start_time + int(
                round((measure_filled_value_array.size - start_index) * period_ns))
            expected_num_values = int(round((data_end_time - data_start_time) / period_ns))

        if expected_num_values > 0:
            grid_slice = measure_filled_time_array[start_index:start_index + expected_num_values]
            # String measures read raw int64 codes (analog=False); numeric
            # aperiodic measures read analog-scaled values. Neither uses the
            # numeric return_nan_filled path (design section 21.5).
            _, r_times, r_values = sdk.get_data(
                measure_id, data_start_time, data_end_time, device_id=device_id,
                patient_id=query_patient_id, analog=not is_string)

            # Carry-forward seed (Bug-1 fix): make carry-forward deterministic and
            # independent of the batch size (num_windows_prefetch). The per-batch
            # read only covers [data_start_time, data_end_time); a reading that
            # precedes this batch would otherwise be invisible and a genuinely
            # KNOWN cell would be emitted as the unknown sentinel. So, for a
            # carry_forward measure, fetch the single most-recent reading in
            # [seed_floor, data_start_time) and seed the grid with it. The floor is
            # the DEFINITION's range start (not the per-batch sub-range start,
            # which the batcher moves forward every batch) so a value set early in
            # the range still carries into a later batch -- but never a reading
            # before the definition range (Bug-2: that hard floor is intentional).
            # Only the last element is retained, so peak RAM stays bounded and the
            # batching behavior is unchanged. presence/count/sparse/aggregate touch
            # only a reading's own cell and are already batch-independent -- no seed.
            seed_floor = range_start_time if definition_range_start_time is None \
                else definition_range_start_time
            seed_time = seed_value = None
            if rule == "carry_forward" and data_start_time > seed_floor:
                _, s_times, s_values = sdk.get_data(
                    measure_id, seed_floor, data_start_time, device_id=device_id,
                    patient_id=query_patient_id, analog=not is_string)
                if np.asarray(s_times).size > 0:
                    seed_time = int(np.asarray(s_times, dtype=np.int64)[-1])
                    seed_value = np.asarray(s_values)[-1]

            values, _known = _rasterize_grid(
                grid_slice, period_ns, np.asarray(r_times, dtype=np.int64), r_values, rule, is_string,
                seed_time=seed_time, seed_value=seed_value)
            measure_filled_value_array[start_index:start_index + expected_num_values] = values

        windowed_measure_times = sliding_window_view(measure_filled_time_array, measure_window_size)
        windowed_measure_values = sliding_window_view(measure_filled_value_array, measure_window_size)

        sliced_windowed_measure_times = windowed_measure_times[::measure_slide_size]
        sliced_windowed_measure_values = windowed_measure_values[::measure_slide_size]

        source_batch_data_dictionary[measure_id] = \
            (sliced_windowed_measure_times, sliced_windowed_measure_values, measure_window_size)
    return source_batch_data_dictionary


def get_label_data(device_id, query_patient_id, batch_start_time, batch_end_time, batch_time_array, sdk,
                   row_size, slide_size, label_sets, label_threshold, label_exact_match=False):
    sliced_labels, threshold_labels = None, None
    # If labels exist, calculate them
    if len(label_sets) > 0:
        # Preallocate label matrix
        label_matrix = np.zeros((len(label_sets), len(batch_time_array)), dtype=np.int8)

        # Populate label matrix
        for idx, label_set_id in enumerate(label_sets):
            sdk.get_label_time_series(
                label_name_id=label_set_id,
                device_id=device_id if device_id else None,
                patient_id=query_patient_id if query_patient_id else None,
                start_time=batch_start_time,
                end_time=batch_end_time,
                timestamp_array=batch_time_array,
                out=label_matrix[idx],
                include_descendants=not label_exact_match,
            )

        # Create label windows
        windowed_label_views = sliding_window_view(
            label_matrix, (len(label_sets), row_size), axis=None)
        sliced_labels = windowed_label_views[0][::slide_size]

        threshold_labels = get_threshold_labels(sliced_labels, label_threshold=label_threshold)
    return sliced_labels, threshold_labels


def get_label_dictionary(sdk, device_id, query_patient_id, source_batch_start_time, source_batch_end_time, label_sets,
                         label_threshold, range_num_windows, row_period_ns, row_size, slide_size, label_exact_match=False):
    range_size = row_size + (range_num_windows - 1) * slide_size
    quantized_end_time = source_batch_start_time + (range_size * row_period_ns)
    source_time_array = np.arange(source_batch_start_time, quantized_end_time, row_period_ns)
    sliced_labels, threshold_labels = get_label_data(
        device_id, query_patient_id, source_batch_start_time, source_batch_end_time, source_time_array,
        sdk, row_size, slide_size, label_sets, label_threshold, label_exact_match=label_exact_match
    )
    return sliced_labels, threshold_labels

def _get_patient_info_from_cache(patient_id, window_start_time, patient_info_cache, patient_history_cache):
    window_patient_info = patient_info_cache.get(patient_id, {})
    for field, history_timeseries in patient_history_cache.get(patient_id, {}).items():
        best_match = find_closest_measurement(window_start_time, history_timeseries)
        if best_match is None:
            continue

        _, _, _, value, units, time = best_match
        window_patient_info[field] = {
            'value': value,
            'units': units,
            'time': time,
        }
    return window_patient_info

def get_window_list(device_id, patient_id, validated_measure_list, source_batch_data_dictionary,
                    batch_start_time, num_windows, window_slide_ns, threshold_labels, sliced_labels,
                    patient_history_cache, patient_history_fields, patient_info_cache):
    batch_window_list = []
    window_start_time = batch_start_time
    for window_i in range(num_windows):
        signal_dictionary = {}
        for measure in validated_measure_list:
            measure_id = measure['id']
            measure_tag = measure['tag']
            measure_freq_nhz = measure['freq_nhz']
            measure_freq_hz = float(measure_freq_nhz / (10 ** 9))
            measure_units = measure['units']

            window_times = source_batch_data_dictionary[measure_id][0][window_i]

            window_values = source_batch_data_dictionary[measure_id][1][window_i]
            measure_expected_count = source_batch_data_dictionary[measure_id][2]

            # actual_count = number of genuinely known cells. Float channels use
            # NaN as the unknown sentinel (waveform / numeric aperiodic); int64
            # string-code channels use the reserved UNKNOWN_STRING_CODE. The
            # float branch is left exactly as before (no numeric-path change).
            if np.issubdtype(window_values.dtype, np.floating):
                actual_count = np.sum(~np.isnan(window_values))
            else:
                actual_count = np.sum(window_values != UNKNOWN_STRING_CODE)

            signal_dictionary[(measure_tag, measure_freq_hz, measure_units)] = \
                {
                    'times': np.copy(window_times),
                    'values': np.copy(window_values),
                    'expected_count': measure_expected_count,
                    'actual_count': actual_count,
                    'measure_id': measure_id,
                }

        label_time_series = np.copy(sliced_labels[window_i]) if \
            sliced_labels is not None else None

        window_classification = threshold_labels[window_i] if \
            threshold_labels is not None else None

        window_patient_info = patient_info_cache.get(patient_id, {})
        if patient_history_fields:
            window_patient_info = _get_patient_info_from_cache(
                patient_id, window_start_time, patient_info_cache, patient_history_cache)

        result_window = Window(
            signals=signal_dictionary,
            start_time=int(window_start_time),
            device_id=device_id,
            patient_id=patient_id,
            label_time_series=label_time_series,
            label=window_classification,
            patient_info=window_patient_info
        )
        batch_window_list.append(result_window)
        window_start_time += window_slide_ns
    return batch_window_list


def _load_patient_cache(patient_id, patient_info_cache, patient_history_cache, sdk, patient_history_fields):
    if patient_id is None:
        return
    # Check if patient id is in the info cache.
    if patient_id not in patient_info_cache:
        patient_info_cache[patient_id] = sdk.get_patient_info(patient_id=patient_id)
        if patient_info_cache[patient_id] is None:
            patient_info_cache[patient_id] = {}
        else:
            # Delete Height And Weight From Static Info
            if 'height' in patient_info_cache[patient_id]:
                del patient_info_cache[patient_id]['height']

            if 'weight' in patient_info_cache[patient_id]:
                del patient_info_cache[patient_id]['weight']

    if patient_history_fields and patient_id not in patient_history_cache:
        patient_history_cache[patient_id] = {}
        for field in ['height', 'weight']:
            if field not in patient_history_fields:
                continue
            patient_history_cache[patient_id][field] = sdk.get_patient_history(
                patient_id=patient_id, field=field)
