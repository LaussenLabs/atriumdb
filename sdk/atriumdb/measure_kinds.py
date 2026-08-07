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
"""The two orthogonal axes that describe a measure (design section 4).

Every measure is classified on two INDEPENDENT axes, stored as two nullable
columns on the ``measure`` table:

``signal_kind`` -- the measure's *temporal shape*, i.e. what its timestamps mean:

===========  ====================================================================
waveform     Regularly sampled at the measure's own frequency (ECG, SpO2 pleth).
             The legacy path; a window is the stored sample grid, NaN-filled.
sample       Irregular point measurements of a continuous quantity (NIBP, labs).
             A value stays "in effect" until the next reading -> carry-forward.
event        Instantaneous occurrences (alarms, HL7 messages). Nothing is in
             effect between them -> the window reports occupancy, not identity.
state        Step function: a value holds until explicitly replaced (vent mode).
             Carry-forward, plus left-censoring before the first observation.
===========  ====================================================================

``value_type`` -- how a value is *encoded* in a block:

=========  ======================================================================
numeric    Ordinary int64 / float64 samples.
string     Text, stored as int64 codes into the measure's append-only
           :class:`~atriumdb.string_dictionary.MeasureStringDictionary`.
=========  ======================================================================

The axes really are orthogonal: a ``sample``/``numeric`` NIBP stream and a
``state``/``string`` ventilator-mode stream are both legitimate, and so is a
``waveform``/``numeric`` ECG. Only ``waveform`` + ``string`` is a practical dead
end -- the windowing layer has no NaN grid for text -- which is why measures
auto-created by an ingest pipeline (always ``waveform`` by default) need
:meth:`~atriumdb.atrium_sdk.AtriumSDK.set_measure_kind` before they can be
iterated.

Both columns are nullable so existing datasets need no backfill; a ``NULL`` is
read-time defaulted to ``waveform`` / ``numeric``. This module owns the
vocabulary, the defaults and the small predicates that go with them, so the
several consumers -- the SDK write/read guards, the windowing render config, the
definition validator and the transfer layer -- can agree by construction instead
of by repeating bare string literals.
"""
from __future__ import annotations

# --- signal_kind ---------------------------------------------------------- #
SIGNAL_KIND_WAVEFORM = "waveform"
SIGNAL_KIND_SAMPLE = "sample"
SIGNAL_KIND_EVENT = "event"
SIGNAL_KIND_STATE = "state"

SIGNAL_KIND_VALUES = (SIGNAL_KIND_WAVEFORM, SIGNAL_KIND_SAMPLE,
                      SIGNAL_KIND_EVENT, SIGNAL_KIND_STATE)

#: Kinds whose timestamps are irregular, i.e. everything except ``waveform``.
#: These are the kinds the Phase 3 rasterizer renders onto a nominal grid.
APERIODIC_SIGNAL_KINDS = (SIGNAL_KIND_SAMPLE, SIGNAL_KIND_EVENT, SIGNAL_KIND_STATE)

# --- value_type ----------------------------------------------------------- #
VALUE_TYPE_NUMERIC = "numeric"
VALUE_TYPE_STRING = "string"

VALUE_TYPE_VALUES = (VALUE_TYPE_NUMERIC, VALUE_TYPE_STRING)

# --- read-time defaults for the nullable columns -------------------------- #
DEFAULT_SIGNAL_KIND = SIGNAL_KIND_WAVEFORM
DEFAULT_VALUE_TYPE = VALUE_TYPE_NUMERIC


def is_string_value_type(value_type) -> bool:
    """True when a measure's ``value_type`` means "values are dictionary codes".

    The one predicate every consumer should use, rather than comparing against
    the bare literal ``'string'``. A ``None`` (un-migrated / never-written
    measure) is NOT string: the read-time default is ``numeric``."""
    return value_type == VALUE_TYPE_STRING


def is_aperiodic_signal_kind(signal_kind) -> bool:
    """True for the irregular kinds (``sample``/``event``/``state``).

    ``None`` defaults to ``waveform`` and is therefore not aperiodic."""
    return (signal_kind or DEFAULT_SIGNAL_KIND) in APERIODIC_SIGNAL_KINDS


def measure_kind_of(measure_info, default_signal_kind=DEFAULT_SIGNAL_KIND,
                    default_value_type=DEFAULT_VALUE_TYPE):
    """``(signal_kind, value_type)`` from a measure-info mapping, defaults applied.

    ``get_measure_info`` already applies the read-time defaults, but the
    validated measure dicts carried inside a ``DatasetDefinition`` may hold an
    explicit ``None`` (they were built from a measure row, or unpickled from an
    older definition). This applies the same defaults everywhere so the windowing
    layer cannot disagree with the SDK about the same measure."""
    signal_kind = measure_info.get("signal_kind") or default_signal_kind
    value_type = measure_info.get("value_type") or default_value_type
    return signal_kind, value_type


#: The signal_kind an invalid ``waveform`` + ``string`` measure is auto-corrected to.
#: ``event`` is the only string-bearing kind that carries nothing forward, so
#: repairing into it can never fabricate a value that was never observed; the
#: caller is told in the same breath how to pick ``state``/``sample`` instead.
STRING_SIGNAL_KIND_FALLBACK = SIGNAL_KIND_EVENT


def is_invalid_kind_combination(signal_kind, value_type) -> bool:
    """True for the one combination the design forbids: ``waveform`` + ``string``.

    Design §4/§21.3: a string measure's timestamps cannot mean "the measure's own
    sample grid", because there is no NaN grid for text. Such a measure passes
    ``insert_measure``, passes ``get_string_data`` and then dies deep inside the
    windowing fill path (``get_iterator`` -> ``get_signal_dictionary`` ->
    ``get_data``: "its values cannot be NaN-filled"), hours after the mistake.

    ``None`` is resolved through the read-time defaults first, so an unstated
    ``signal_kind`` on a string measure counts as invalid -- that is exactly the
    shape the docs' former string example produced."""
    return (signal_kind or DEFAULT_SIGNAL_KIND) == SIGNAL_KIND_WAVEFORM and is_string_value_type(value_type)


def invalid_kind_combination_message(measure_id, corrected_signal_kind=STRING_SIGNAL_KIND_FALLBACK,
                                     measure_exists=True) -> str:
    """The one message every ``waveform`` + ``string`` site reports, so the SDK
    write guards, the kind setter and the transfer layer all point the caller at
    the same fix instead of inventing their own wording.

    ``measure_exists=False`` is the ``insert_measure`` case, where there is no id to
    hand back yet, so the remedy is stated as the ``insert_measure`` argument rather
    than as a :meth:`~atriumdb.atrium_sdk.AtriumSDK.set_measure_kind` call."""
    subject = f"Measure {measure_id}" if measure_exists else f"Measure {measure_id} (being created)"
    remedy = (
        f"sdk.set_measure_kind({measure_id}, signal_kind='{SIGNAL_KIND_STATE}')"
        if measure_exists else
        f"insert_measure(..., signal_kind='{SIGNAL_KIND_STATE}', value_type='{VALUE_TYPE_STRING}')")
    return (
        f"{subject} is a string measure with signal_kind='{SIGNAL_KIND_WAVEFORM}'. "
        f"That combination is not usable: a 'waveform' measure's timestamps are its own "
        f"sample grid, and there is no NaN grid for text, so get_iterator() fails deep in the "
        f"fill path even though get_string_data() works. A string measure needs a signal_kind "
        f"of '{SIGNAL_KIND_EVENT}' (instantaneous occurrences - alarms, messages), "
        f"'{SIGNAL_KIND_STATE}' (a value holds until replaced - ventilator mode) or "
        f"'{SIGNAL_KIND_SAMPLE}' (irregular point measurements). It has been auto-corrected to "
        f"'{corrected_signal_kind}'; state the signal_kind you actually want with {remedy}.")


def changed_kind_fields(current_signal_kind, current_value_type, signal_kind, value_type):
    """Which of the two requested fields actually change what is stored.

    Returns ``(signal_kind_or_None, value_type_or_None)`` where ``None`` means
    "leave this field alone" -- the convention every kind setter already uses. A
    field that is unrequested, or requested with the value already stored, comes
    back ``None``, so the common case (a repeated ``insert_measure`` or a repeated
    transfer of an unchanged measure) resolves to "nothing to do" and neither
    writes a row nor emits a warning.

    Both callers -- :meth:`~atriumdb.atrium_sdk.AtriumSDK._apply_kind_to_existing_measure`
    and transfer's ``_carry_measure_kind`` -- reach the same "is this a real
    change?" question by different routes and used to answer it with their own
    copy of this comparison."""
    return (signal_kind if signal_kind is not None and signal_kind != current_signal_kind else None,
            value_type if value_type is not None and value_type != current_value_type else None)
