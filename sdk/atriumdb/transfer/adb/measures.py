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
from atriumdb.measure_kinds import changed_kind_fields

measure_tag_match_rule_options = ["all", "best"]


def transfer_measures(from_sdk, to_sdk, measure_id_list=None, measure_tag_list=None, measure_tag_match_rule=None):
    if measure_id_list is not None and measure_tag_list is not None:
        raise ValueError("Only one of measure_id_list or measure_tag_list should be provided")

    measure_tag_match_rule = "all" if measure_tag_match_rule is None else measure_tag_match_rule
    assert measure_tag_match_rule in measure_tag_match_rule_options, \
        f"measure_tag_match_rule must be one of {measure_tag_match_rule_options}"

    from_measures = from_sdk.get_all_measures()
    measure_map = {}

    if measure_id_list:
        for from_measure_id in measure_id_list:
            if from_measure_id in from_measures:
                measure_map[from_measure_id] = _transfer_measure(to_sdk, from_measures[from_measure_id])

    elif measure_tag_list:
        for tag in measure_tag_list:
            if measure_tag_match_rule == "best":
                matching_ids = from_sdk.get_measure_id_list_from_tag(tag, approx=True)
                best_id = matching_ids[0] if matching_ids else None
                if best_id in from_measures:
                    measure_map[best_id] = _transfer_measure(to_sdk, from_measures[best_id])
            elif measure_tag_match_rule == "all":
                for from_measure_id, measure_info in from_measures.items():
                    if measure_info['tag'] == tag:
                        measure_map[from_measure_id] = _transfer_measure(to_sdk, measure_info)

    return measure_map


def preflight_measure_value_types(from_sdk, to_sdk, measure_id_list=None, measure_tag_list=None,
                                  measure_tag_match_rule=None):
    """Check every measure this transfer would touch for a value-type collision, BEFORE
    anything is written to the destination.

    ``_transfer_measure`` hands back an existing destination measure whenever
    (tag, freq, units) match, without looking at ``value_type``. Without this
    preflight, a string source measure landing on a numeric destination measure of the
    same identity is discovered only when the first string write trips the invariant --
    by which point destination measures, devices, patients, encounters and the *other*
    measures' data are already committed, with no rollback and an error naming only
    the destination measure id. This resolves the same identity mapping read-only and
    raises up-front, naming the source measure, its tag and the destination measure.

    Raises ``ValueError`` listing every colliding measure; returns None when the
    transfer is safe to start.
    """
    measure_tag_match_rule = "all" if measure_tag_match_rule is None else measure_tag_match_rule
    from_measures = from_sdk.get_all_measures()

    if measure_id_list is not None:
        candidates = [from_measures[m_id] for m_id in measure_id_list if m_id in from_measures]
    elif measure_tag_list is not None:
        candidates = []
        for tag in measure_tag_list:
            if measure_tag_match_rule == "best":
                matching_ids = from_sdk.get_measure_id_list_from_tag(tag, approx=True)
                best_id = matching_ids[0] if matching_ids else None
                if best_id in from_measures:
                    candidates.append(from_measures[best_id])
            else:
                candidates.extend(info for info in from_measures.values() if info['tag'] == tag)
    else:
        candidates = list(from_measures.values())

    collisions = []
    for measure_info in candidates:
        src_value_type = measure_info.get('value_type')
        if src_value_type is None:
            continue

        # Resolve the destination measure the same way _transfer_measure does: the
        # source id when it names the same measure, otherwise the (tag, freq, units)
        # identity. Both are read-only lookups.
        dest_measure_id = None
        check_measure_info = to_sdk.get_measure_info(measure_info['id'])
        if check_measure_info is not None and \
                check_measure_info['tag'] == measure_info['tag'] and \
                check_measure_info['freq_nhz'] == measure_info['freq_nhz'] and \
                check_measure_info['unit'] == measure_info['unit']:
            dest_measure_id = measure_info['id']
        else:
            dest_measure_id = to_sdk.get_measure_id(
                measure_info['tag'], freq=measure_info['freq_nhz'], freq_units="nHz",
                units=measure_info['unit'])

        if dest_measure_id is None:
            continue

        dest_value_type = to_sdk._established_value_type(dest_measure_id)
        if dest_value_type is not None and dest_value_type != src_value_type:
            collisions.append(
                f"source measure {measure_info['id']} (tag={measure_info['tag']!r}, "
                f"freq_nhz={measure_info['freq_nhz']}, units={measure_info['unit']!r}) is "
                f"'{src_value_type}', but destination measure {dest_measure_id} with the same "
                f"identity already holds '{dest_value_type}' data")

    if collisions:
        raise ValueError(
            "Transfer aborted before writing anything: value_type collision between source and "
            "destination measures. " + "; ".join(collisions) +
            ". A measure is either numeric or string -- rename or re-tag the source measure, or "
            "transfer it into a destination that does not already hold the other kind of data.")


def _transfer_measure(to_sdk, measure_info):
    measure_tag = measure_info['tag']
    freq = measure_info['freq_nhz']
    units = measure_info['unit']
    measure_name = measure_info['name']
    from_measure_id = measure_info['id']
    # Carry the measure-kind metadata across the transfer so the destination measure
    # keeps its temporal shape / value encoding. get_all_measures /
    # get_measure_info always resolve these (defaults: waveform / numeric), so they are
    # present for every source measure.
    signal_kind = measure_info.get('signal_kind')
    value_type = measure_info.get('value_type')

    # Check if measure_id already exists
    check_measure_info = to_sdk.get_measure_info(from_measure_id)
    if check_measure_info is not None:
        # if its the same measure, return the id without inserting
        if check_measure_info['tag'] == measure_tag and \
                check_measure_info['freq_nhz'] == freq and \
                check_measure_info['unit'] == units:
            # Carry the metadata onto the EXISTING destination measure too. Returning
            # early without this would keep the destination's default 'waveform', so an
            # incremental / repeat transfer would degrade state|event -> waveform and
            # produce the un-iterable waveform+string combination.
            _carry_measure_kind(to_sdk, from_measure_id, signal_kind, value_type)
            return from_measure_id
        else:
            # The measure_id is taken but its a different measure so ask for a new id when inserting
            from_measure_id = None

    to_measure_id = to_sdk.insert_measure(
        measure_tag=measure_tag, freq=freq, units=units, measure_name=measure_name, measure_id=from_measure_id,
        signal_kind=signal_kind, value_type=value_type)
    return to_measure_id


def _carry_measure_kind(to_sdk, dest_measure_id, signal_kind, value_type):
    """Apply the source measure's kind metadata to an existing destination measure.

    Only fields that actually differ are written. A value_type that conflicts with data
    already in the destination is left to the transfer preflight
    (:func:`preflight_measure_value_types`), which reports it before anything is written,
    so it is skipped here rather than raising mid-transfer."""
    if signal_kind is None and value_type is None:
        return
    current_info = to_sdk.get_measure_info(dest_measure_id)
    current = None if current_info is None else (current_info.get('signal_kind'), current_info.get('value_type'))
    if current is None:
        return
    current_signal_kind, current_value_type = current
    new_signal_kind, new_value_type = changed_kind_fields(
        current_signal_kind, current_value_type, signal_kind, value_type)
    if new_value_type is not None and to_sdk._established_value_type(dest_measure_id) is not None:
        # The destination already has data of a settled kind; never relabel it here.
        new_value_type = None
    if new_signal_kind is None and new_value_type is None:
        return
    to_sdk.update_measure(dest_measure_id, signal_kind=new_signal_kind, value_type=new_value_type)
