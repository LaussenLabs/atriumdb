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

"""Phase 6 (§24.1#3) — transfer of the ``encounter`` family.

Transfers ``encounter`` + ``device_encounter`` (and the ``bed`` / ``unit`` /
``institution`` rows they reference, for referential integrity) for the patients and
devices already being transferred. Applies the §15.3 field-level de-identification and
time-shifting:

* ``patient_id`` / ``device_id`` remapped via the existing transfer maps.
* ``bed_id`` remapped (a fresh bed/unit/institution chain is created at the destination).
* ``visit_number`` scrambled to a random int via a consistent per-transfer map (the same
  id-scramble style used for patient ids — see :mod:`atriumdb.transfer.adb.patients`).
* location NAMES (bed / unit / institution) pseudonymized to stable pseudonyms.
* ``start_time`` / ``end_time`` / ``last_updated`` shifted by ``time_shift_nano`` — every
  time-bearing column is wired explicitly (there is no global shift pass).

Everything above is governed by the ``keep_identified`` per-table allowlist (§24.2#2): a
``dict[str, list[str] | "all"]`` opting fields back to identified. When ``deidentify`` is
False every field is identified and ``keep_identified`` is a no-op.

``log_hl7_adt`` is **never** transferred (§24.1#4) — this module simply never touches it.
"""

import random

# The per-table inventory of identifying / quasi-identifying fields that de-identification
# pseudonymizes or scrambles by default. This is the authoritative list that
# ``keep_identified`` opts back to identified (see the module docstring and design §24.4).
#   encounter.visit_number   -> scrambled to a random int (direct identifier)
#   bed.name / unit.name / institution.name -> pseudonymized to a stable pseudonym
#   device_encounter has no free identifying field of its own (ids are always remapped for
#   referential integrity; times are always shifted) -- listed for completeness/uniformity.
SENSITIVE_FIELDS = {
    "encounter": ["visit_number"],
    "device_encounter": [],
    "bed": ["name"],
    "unit": ["name"],
    "institution": ["name"],
}


def _field_identified(deidentify, keep_identified, table, field):
    """Return True if ``table.field`` should remain identified (i.e. NOT pseudonymized /
    scrambled). Identified when de-id is off, or when the caller opted the field back in
    via ``keep_identified`` (``"all"`` shorthand keeps the whole table identified)."""
    if not deidentify:
        return True
    keep = keep_identified.get(table)
    if keep is None:
        return False
    if keep == "all":
        return True
    return field in keep


def _validate_keep_identified(keep_identified):
    if keep_identified is None:
        return {}
    if not isinstance(keep_identified, dict):
        raise ValueError(
            "keep_identified must be a dict of {table: [field names] | 'all'}")
    for table, keep in keep_identified.items():
        if table not in SENSITIVE_FIELDS:
            raise ValueError(
                f"keep_identified table {table!r} is not one of {list(SENSITIVE_FIELDS.keys())}")
        if keep == "all":
            continue
        if not isinstance(keep, list):
            raise ValueError(
                f"keep_identified[{table!r}] must be a list of field names or the string 'all'")
        for field in keep:
            if field not in SENSITIVE_FIELDS[table]:
                raise ValueError(
                    f"keep_identified[{table!r}] field {field!r} is not a sensitive field of "
                    f"{table!r}; sensitive fields are {SENSITIVE_FIELDS[table]}")
    return keep_identified


class _ScrambleMap:
    """Consistent per-transfer map from an original value to a random int, mirroring the
    patient-id scramble in :func:`atriumdb.transfer.adb.patients.generate_patient_ids`."""

    def __init__(self):
        self._map = {}
        self._used = set()

    def get(self, original):
        if original in self._map:
            return self._map[original]
        # Draw a fresh random int not already handed out in this transfer.
        while True:
            candidate = random.randint(10000, 99999999)
            if candidate not in self._used:
                break
        self._used.add(candidate)
        self._map[original] = candidate
        return candidate


def transfer_encounters(src_sdk, dest_sdk, patient_id_map, device_id_map, deidentify=False,
                        keep_identified=None, time_shift_nano=None):
    """Transfer the encounter family for the already-transferred patients/devices.

    :param src_sdk: Source SDK.
    :param dest_sdk: Destination SDK.
    :param dict patient_id_map: ``{src_patient_id: dest_patient_id}`` (from patient transfer).
    :param dict device_id_map: ``{src_device_id: dest_device_id}`` (from device transfer).
    :param deidentify: Truthy to pseudonymize/scramble sensitive fields by default.
    :param dict keep_identified: Per-table allowlist opting fields back to identified.
    :param int time_shift_nano: Nanoseconds to add to every time-bearing column.
    """
    keep_identified = _validate_keep_identified(keep_identified)

    src_patient_ids = [pid for pid in patient_id_map.keys()]
    if not src_patient_ids:
        return

    encounter_rows = src_sdk.sql_handler.select_encounters(patient_id_list=src_patient_ids)
    if not encounter_rows:
        return

    visit_scramble = _ScrambleMap()
    pseudonyms = {"bed": {}, "unit": {}, "institution": {}}

    # Caches so a bed / unit / institution shared by many encounters is only created once.
    bed_id_map = {}
    unit_id_map = {}
    institution_id_map = {}
    source_id_map = {}

    def pseudonymize_name(table, name):
        """Return a stable pseudonym for a location name (or the original if kept)."""
        if _field_identified(deidentify, keep_identified, table, "name"):
            return name
        table_map = pseudonyms[table]
        if name not in table_map:
            table_map[name] = f"{table}_{visit_scramble.get((table, name))}"
        return table_map[name]

    def resolve_source_id(src_source_id):
        """Ensure ``src_source_id`` exists at the destination; return the dest source id.
        Sources are not a de-id surface (§15.3); they transfer for referential integrity."""
        if src_source_id is None:
            return None
        if src_source_id in source_id_map:
            return source_id_map[src_source_id]
        # The default AtriumDB system source (id 1) always exists at the destination.
        if dest_sdk.sql_handler.select_source(source_id=src_source_id) is not None:
            source_id_map[src_source_id] = src_source_id
            return src_source_id
        src_source = src_sdk.sql_handler.select_source(source_id=src_source_id)
        if src_source is None:
            source_id_map[src_source_id] = 1
            return 1
        _, name, description = src_source
        dest_source_id = dest_sdk.sql_handler.insert_source(name=name, description=description)
        source_id_map[src_source_id] = dest_source_id
        return dest_source_id

    def resolve_institution_id(src_institution_id):
        if src_institution_id is None:
            return None
        if src_institution_id in institution_id_map:
            return institution_id_map[src_institution_id]
        row = src_sdk.sql_handler.select_institution(institution_id=src_institution_id)
        if row is None:
            institution_id_map[src_institution_id] = None
            return None
        _, name = row
        dest_id = dest_sdk.sql_handler.insert_institution(
            name=pseudonymize_name("institution", name))
        institution_id_map[src_institution_id] = dest_id
        return dest_id

    def resolve_unit_id(src_unit_id):
        if src_unit_id is None:
            return None
        if src_unit_id in unit_id_map:
            return unit_id_map[src_unit_id]
        row = src_sdk.sql_handler.select_unit(unit_id=src_unit_id)
        if row is None:
            unit_id_map[src_unit_id] = None
            return None
        _, src_institution_id, name, unit_type = row
        dest_institution_id = resolve_institution_id(src_institution_id)
        dest_id = dest_sdk.sql_handler.insert_unit(
            institution_id=dest_institution_id, name=pseudonymize_name("unit", name),
            unit_type=unit_type)
        unit_id_map[src_unit_id] = dest_id
        return dest_id

    def resolve_bed_id(src_bed_id):
        if src_bed_id is None:
            return None
        if src_bed_id in bed_id_map:
            return bed_id_map[src_bed_id]
        row = src_sdk.sql_handler.select_bed(bed_id=src_bed_id)
        if row is None:
            bed_id_map[src_bed_id] = None
            return None
        _, src_unit_id, name = row
        dest_unit_id = resolve_unit_id(src_unit_id)
        dest_id = dest_sdk.sql_handler.insert_bed(
            unit_id=dest_unit_id, name=pseudonymize_name("bed", name))
        bed_id_map[src_bed_id] = dest_id
        return dest_id

    encounter_id_map = {}
    for (enc_id, patient_id, bed_id, start_time, end_time, source_id, visit_number,
         last_updated) in encounter_rows:
        if patient_id not in patient_id_map:
            continue
        dest_patient_id = patient_id_map[patient_id]
        dest_bed_id = resolve_bed_id(bed_id)
        dest_source_id = resolve_source_id(source_id)

        # visit_number: scramble to a random int unless kept identified.
        if visit_number is not None and not _field_identified(
                deidentify, keep_identified, "encounter", "visit_number"):
            dest_visit_number = visit_scramble.get(str(visit_number))
        else:
            dest_visit_number = visit_number

        # Shift every time-bearing column explicitly (§24.1#5).
        dest_start_time = start_time
        dest_end_time = end_time
        dest_last_updated = last_updated
        if time_shift_nano is not None:
            if dest_start_time is not None:
                dest_start_time += time_shift_nano
            if dest_end_time is not None:
                dest_end_time += time_shift_nano
            if dest_last_updated is not None:
                dest_last_updated += time_shift_nano

        dest_encounter_id = dest_sdk.sql_handler.insert_encounter(
            patient_id=dest_patient_id, bed_id=dest_bed_id, start_time=dest_start_time,
            end_time=dest_end_time, source_id=(dest_source_id if dest_source_id is not None else 1),
            visit_number=dest_visit_number, last_updated=dest_last_updated)
        encounter_id_map[enc_id] = dest_encounter_id

    if not encounter_id_map:
        return

    # device_encounter: rows are keyed by the source encounter ids we just transferred.
    device_encounter_rows = src_sdk.sql_handler.select_all_device_encounters_by_encounter_list(
        list(encounter_id_map.keys()))
    for (_de_id, device_id, encounter_id, start_time, end_time, source_id) in device_encounter_rows:
        if device_id not in device_id_map:
            # The device is outside the transfer set; skip to preserve referential integrity.
            continue
        if encounter_id not in encounter_id_map:
            continue
        dest_source_id = resolve_source_id(source_id)
        dest_start_time = start_time
        dest_end_time = end_time
        if time_shift_nano is not None:
            if dest_start_time is not None:
                dest_start_time += time_shift_nano
            if dest_end_time is not None:
                dest_end_time += time_shift_nano

        dest_sdk.sql_handler.insert_device_encounter(
            device_id=device_id_map[device_id], encounter_id=encounter_id_map[encounter_id],
            start_time=dest_start_time, end_time=dest_end_time,
            source_id=(dest_source_id if dest_source_id is not None else 1))
