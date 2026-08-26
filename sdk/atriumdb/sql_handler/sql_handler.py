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
import time
import math
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional


def block_insert_tuples(block_data: List[Dict], file_id: int):
    """The ``block_index`` insert parameters for ``block_data``, in column order.

    The column order has to match the INSERT in both backends and in every caller that
    writes blocks, so it is spelled once here rather than re-typed at each call site."""
    return [(block["measure_id"], block["device_id"], file_id, block["start_byte"], block["num_bytes"],
             block["start_time_n"], block["end_time_n"], block["num_values"]) for block in block_data]


def interval_insert_tuples(interval_data: List[Dict]):
    """The ``interval_index`` insert parameters for ``interval_data``, in column order."""
    return [(interval["measure_id"], interval["device_id"], interval["start_time_n"],
             interval["end_time_n"]) for interval in interval_data]


class SQLHandler(ABC):
    # How this backend's driver words an error about a column that isn't there.
    # SQLite says "no such column", MariaDB says "unknown column"; each backend
    # only ever emits its own, so the wording stays per-backend rather than being
    # matched loosely against both.
    _MISSING_COLUMN_PHRASE = ""

    # How this backend reports a query against a table that was never created --
    # the driver exception class(es) to catch, and the phrases that must all appear
    # in the message. Both are per-backend for the same reason as the column phrase
    # above; an empty tuple of error types catches nothing, which is the right
    # default for a backend that has not declared them.
    _MISSING_TABLE_ERRORS = ()
    _MISSING_TABLE_PHRASES = ()

    @abstractmethod
    def create_schema(self):
        # Creates Tables if they don't exist.
        pass

    @staticmethod
    def _int_id_list_clauses(**id_lists):
        """Build ``<column> IN (?, ?, ...)`` clauses for the given ``column=id_list``
        pairs, skipping any list that is ``None`` or empty.

        Returns ``(clauses, args)``. A skipped list means "no filter on this column";
        note that this is the *filter* reading of an empty list, which is why these
        call sites differ from :meth:`_select_rows_in_list`, where an empty list is
        the query itself and must match nothing."""
        clauses, args = [], ()
        for column, id_list in id_lists.items():
            if id_list is None or len(id_list) == 0:
                continue
            clauses.append("{} IN ({})".format(column, ','.join(['?'] * len(id_list))))
            args += tuple(int(value) for value in id_list)
        return clauses, args

    @staticmethod
    def _append_limit_offset(query: str, limit, offset) -> str:
        """Append ``LIMIT``/``OFFSET`` to a listing query when asked for.
        An ``offset`` without a ``limit`` is ignored -- SQL has no bare OFFSET."""
        if limit is None:
            return query
        return query + (f" LIMIT {limit} OFFSET {offset}" if offset is not None else f" LIMIT {limit}")

    def _select_all_ordered(self, query: str, limit=None, offset=None):
        """Run a parameterless listing query with optional LIMIT/OFFSET.

        Returns ``[]`` when the table does not exist yet, so listing a dataset that
        predates a table's migration reads as empty rather than raising."""
        try:
            with self.connection(begin=False) as (conn, cursor):
                cursor.execute(self._append_limit_offset(query, limit, offset))
                return cursor.fetchall()
        except self._MISSING_TABLE_ERRORS as e:
            message = str(e)
            if self._MISSING_TABLE_PHRASES and all(phrase in message for phrase in self._MISSING_TABLE_PHRASES):
                return []
            raise

    def _reraise_missing_measure_column(self, error):
        """Translate a driver error about an absent measure column into an actionable
        ValueError naming the upgrade.

        The measure-kind columns (``period_ns``, ``signal_kind``, ``value_type``) are an
        additive migration, so a dataset written by an older version reaches these
        queries without them, and the raw driver error names neither the cause nor the
        fix. Returns normally when ``error`` is anything else, leaving the caller to
        re-raise it unchanged."""
        message = str(error).lower()
        if "period_ns" in message or self._MISSING_COLUMN_PHRASE in message:
            raise ValueError(
                "A required column is missing from the measure table (e.g. "
                "'period_ns', 'signal_kind' or 'value_type'). "
                "Please run AtriumSDK(auto_upgrade=True) to update the database schema."
            ) from error

    @abstractmethod
    def connection(self, begin: bool = False):
        pass

    @abstractmethod
    def update_measure_schema(self):
        """Add the additive nullable measure columns (period_ns, signal_kind,
        value_type) if they do not exist. Idempotent."""
        pass

    def measure_has_blocks(self, measure_id: int) -> bool:
        """True if any block exists for this measure (used to detect an
        already-established numeric measure when the value_type column is NULL)."""
        with self.connection() as (conn, cursor):
            cursor.execute("SELECT 1 FROM block_index WHERE measure_id = ? LIMIT 1", (int(measure_id),))
            return cursor.fetchone() is not None

    def update_measure_metadata(self, measure_id: int, signal_kind: str = None, value_type: str = None):
        """Set the measure-kind columns for a measure. Only the provided
        (non-None) fields are written; used to persist first-write value_type
        inference and the opportunistic string backfill. Idempotent."""
        sets, params = [], []
        if signal_kind is not None:
            sets.append("signal_kind = ?")
            params.append(signal_kind)
        if value_type is not None:
            sets.append("value_type = ?")
            params.append(value_type)
        if not sets:
            return
        params.append(int(measure_id))
        with self.connection() as (conn, cursor):
            cursor.execute(f"UPDATE measure SET {', '.join(sets)} WHERE id = ?", params)
            conn.commit()

    # ------------------------------------------------------------------ #
    # String dictionary high-water mark
    #
    # The per-measure string dictionary lives on the FILESYSTEM
    # (``meta/string_dict/measure_<id>.jsonl``) while the blocks that reference its
    # codes are indexed in THIS database. The two can therefore be restored
    # independently -- a DB + ``tsc/`` restore that omits ``meta/`` is the ordinary
    # case -- and nothing tied the dictionary's length to the codes already committed.
    # After such a loss the next write started again at code 0, so every historical
    # code silently decoded to a DIFFERENT string: undetectable, permanent clinical
    # data corruption.
    #
    # The invariant to enforce is "len(dictionary) >= max code in blocks + 1". Deriving
    # the right-hand side means decoding blocks, which is far too expensive to do on
    # every write, so instead the vocabulary size is recorded here (in the metadata
    # database, which survives the loss) each time a string write commits. Comparing
    # the file's length against that high-water mark is one indexed primary-key lookup
    # per string write and catches both total loss and tail truncation.
    #
    # Stored in the existing ``setting`` table so no schema migration is needed and
    # every already-deployed dataset picks the guard up on its next string write.
    # ------------------------------------------------------------------ #
    STRING_DICT_SIZE_SETTING_PREFIX = "string_dict_size_measure_"

    @classmethod
    def _string_dict_size_setting_name(cls, measure_id: int) -> str:
        return f"{cls.STRING_DICT_SIZE_SETTING_PREFIX}{int(measure_id)}"

    def get_string_dict_watermark(self, measure_id: int):
        """Largest vocabulary size ever committed for this measure's string
        dictionary, or None when the measure has never had a string write commit
        (including every dataset written before this guard existed)."""
        row = self.select_setting(self._string_dict_size_setting_name(measure_id))
        if row is None:
            return None
        try:
            return int(row[1])
        except (TypeError, ValueError, IndexError):  # pragma: no cover - corrupt row
            return None

    def set_string_dict_watermark(self, measure_id: int, vocabulary_size: int):
        """Raise the recorded vocabulary size for a measure. Monotonic: a smaller
        value is ignored, so an out-of-order or concurrent writer can never lower the
        mark and weaken the guard. Backends override with a single-statement upsert;
        this fallback is read-then-write and is only safe for a single writer."""
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Dataset schema-version marker
    #
    # Nothing recorded which SDK shaped a dataset -- a dataset created by an older or
    # newer SDK all look the same from the outside except
    # for what they do or don't contain. That makes an upgrade gap detectable only by
    # tripping over it (a query naming a column that isn't there) and makes a
    # downgrade -- an older SDK opening a dataset a newer SDK wrote -- undetectable
    # in either direction.
    #
    # This stamp does not, and cannot, close that second gap for datasets already in
    # the wild: an SDK build that predates this marker has no code path that looks for
    # it, so it will never refuse to open a stamped dataset. What it buys is real
    # only for code that already knows to check it -- this build and anything after
    # it -- which is why the version check in AtriumSDK._init_local_mode refuses to
    # open a dataset stamped with a version NEWER than CURRENT_DATASET_SCHEMA_VERSION,
    # rather than silently misreading it. Recorded in the existing `setting` table, so
    # no schema migration is needed for the marker itself.
    # ------------------------------------------------------------------ #
    DATASET_SCHEMA_VERSION_SETTING_NAME = "dataset_schema_version"

    # Bumped whenever a change requires something a dataset stamped with the previous
    # version would not have (so far: the MariaDB insert_interval_union stored
    # procedure). A dataset with no stamp at all predates the marker entirely -- every
    # dataset ever written by `main` or by an SDK before this one.
    CURRENT_DATASET_SCHEMA_VERSION = 1

    def get_dataset_schema_version(self):
        """The schema version this dataset was last stamped with, or None when it has
        never been stamped. A pure read -- see :meth:`record_dataset_schema_version`
        for the write side, and :meth:`pending_schema_upgrades` for how this feeds
        detection."""
        row = self.select_setting(self.DATASET_SCHEMA_VERSION_SETTING_NAME)
        if row is None:
            return None
        try:
            return int(row[1])
        except (TypeError, ValueError):  # pragma: no cover - corrupt row
            return None

    def record_dataset_schema_version(self) -> bool:
        """Stamp this dataset with the version this SDK build produces. A no-op, and
        the single indexed read it costs is the "one cheap check" auto_upgrade must
        pay on an already-current dataset, when the existing stamp already matches.
        Unlike :meth:`set_string_dict_watermark` this is not monotonic -- the stamp
        always becomes whatever this build's CURRENT_DATASET_SCHEMA_VERSION is,
        because it is only ever written by code that just finished bringing the
        dataset up to exactly that version."""
        if self.get_dataset_schema_version() == self.CURRENT_DATASET_SCHEMA_VERSION:
            return False
        self._upsert_setting(self.DATASET_SCHEMA_VERSION_SETTING_NAME, str(self.CURRENT_DATASET_SCHEMA_VERSION))
        return True

    def _upsert_setting(self, name: str, value: str):
        """Unconditional single-statement upsert for one `setting` row -- the new
        value always wins, unlike the monotonic comparison in
        set_string_dict_watermark. Backends override with a single statement; this
        fallback is read-then-write and only safe for a single writer."""
        with self.connection(begin=True) as (conn, cursor):
            cursor.execute("SELECT 1 FROM setting WHERE name = ?", (name,))
            if cursor.fetchone() is None:
                cursor.execute("INSERT INTO setting (name, value) VALUES (?, ?)", (name, value))
            else:
                cursor.execute("UPDATE setting SET value = ? WHERE name = ?", (value, name))

    # ------------------------------------------------------------------ #
    # Legacy aperiodic measures (freq_nhz = 0)
    #
    # Before this SDK declared signal_kind, a writer expressed "this signal is
    # aperiodic" by giving the measure a frequency of zero -- an intermittent
    # measurement like a non-invasive blood pressure cuff, which has no sampling
    # rate to state. The DATA those writers produced is already exactly what this
    # SDK calls an aperiodic sample measure: an explicit timestamp array
    # (T_TYPE_TIMESTAMP_ARRAY_INT64_NANO), with the period left unset at write time
    # so it was detected from the timestamps and stored in each block header. Only
    # the `measure` row is expressed in the old vocabulary.
    #
    # So this is a metadata rename, not a re-encode: give the row the nominal period
    # its own blocks already demonstrate, and say signal_kind='sample' outright. No
    # block is rewritten and no stored value changes.
    #
    # It has to happen during auto_upgrade because `freq_nhz = 0` is otherwise fatal
    # on open -- AtriumSDK.__init__ caches every measure and computes
    # 10 ** 18 // freq_nhz -- and the repair must therefore run before that cache is
    # built.
    # ------------------------------------------------------------------ #
    DEFAULT_APERIODIC_PERIOD_NS = 10 ** 9

    def select_zero_freq_measures(self):
        """`(id, tag, unit)` for every measure row carrying the legacy zero-frequency
        aperiodic marker. Read-only."""
        with self.connection() as (conn, cursor):
            cursor.execute(
                "SELECT id, tag, unit FROM measure WHERE freq_nhz IS NULL OR freq_nhz <= 0")
            return cursor.fetchall()

    def _observed_period_ns(self, measure_id: int) -> int:
        """The measure's typical sample spacing, from the block index alone.

        Deliberately avoids reading block headers: this runs mid-construction, before
        the file API and the codec are usable, and `block_index` already carries
        everything needed -- a block's span divided by its gaps gives that block's mean
        spacing. The median across blocks is robust to the odd sparse block, and mirrors
        how the write path classifies a signal (observed median spacing), so a converted
        measure is described the same way a freshly written one would be.

        Falls back to one second for a measure with no multi-sample block, matching
        ``detect_period``'s own fallback -- there is no evidence to do better, and the
        value only ever serves as a nominal.
        """
        with self.connection() as (conn, cursor):
            cursor.execute(
                "SELECT start_time_n, end_time_n, num_values FROM block_index "
                "WHERE measure_id = ? AND num_values > 1", (int(measure_id),))
            rows = cursor.fetchall()
        spacings = sorted((int(end) - int(start)) // (int(count) - 1)
                          for start, end, count in rows
                          if int(end) > int(start))
        if not spacings:
            return self.DEFAULT_APERIODIC_PERIOD_NS
        return spacings[len(spacings) // 2]

    def repair_zero_freq_measures(self) -> list:
        """Convert legacy zero-frequency rows into declared aperiodic sample measures.

        Returns `(measure_id, tag, period_ns)` per converted row; empty when there is
        nothing to do, which is the common case and costs one indexed read.

        ``value_type`` is left alone on purpose -- ``_backfill_string_value_types``
        runs after this and decides it from the presence of a string dictionary, which
        is better evidence than anything available here.
        """
        converted = []
        for measure_id, tag, unit in self.select_zero_freq_measures():
            period_ns = self._observed_period_ns(measure_id)
            freq_nhz = (10 ** 18) // period_ns
            # `UNIQUE (tag, freq_nhz, unit)` means the new frequency can collide with a
            # real measure that already occupies it. Leave those alone and report them:
            # merging two measures is a data decision, not a migration's to make.
            with self.connection() as (conn, cursor):
                cursor.execute(
                    "SELECT id FROM measure WHERE tag = ? AND unit = ? AND freq_nhz = ? AND id != ?",
                    (tag, unit, freq_nhz, int(measure_id)))
                if cursor.fetchone() is not None:
                    continue
                cursor.execute(
                    "UPDATE measure SET freq_nhz = ?, period_ns = ?, "
                    "signal_kind = COALESCE(signal_kind, 'sample') WHERE id = ?",
                    (freq_nhz, period_ns, int(measure_id)))
                conn.commit()
            converted.append((int(measure_id), tag, period_ns))
        return converted

    def interval_union_procedure_current(self) -> bool:
        """True when this dataset already has whatever this backend needs for
        union-merge interval writes. SQLite's merge mode is plain Python
        (SQLiteHandler._insert_intervals), not a stored procedure, so there is
        nothing to check and the base answer is always True; MariaDB overrides this
        with a real check against information_schema.ROUTINES, because a dataset
        without the procedure does not have it. ``MariaDBHandler.create_schema`` creates
        it for a new dataset."""
        return True

    def ensure_interval_union_procedure(self) -> bool:
        """Create whatever this backend needs for union-merge interval writes, if
        this dataset predates it. No-op on SQLite (see
        :meth:`interval_union_procedure_current`); MariaDB overrides this to run
        `CREATE PROCEDURE IF NOT EXISTS insert_interval_union`. Returns True if
        anything was actually created."""
        return False

    def pending_schema_upgrades(self) -> List[str]:
        """Schema gaps auto_upgrade would close on this dataset, checked without
        modifying anything -- the detection half kept separate from the repair half.
        Empty means nothing to do, which is the one cheap check `auto_upgrade=True`
        must pay on an already-current dataset, and what `auto_upgrade=False` uses to
        raise before a caller trips over the gap some other way.

        The additive measure columns (period_ns/signal_kind/value_type) and the mrn
        column type are deliberately NOT re-checked here: both already fail fast with
        their own actionable ValueError the moment a query names them
        (_reraise_missing_measure_column; the check_mrn_column_is_text call in
        AtriumSDK._init_local_mode), so probing them a second time here would just be
        a redundant read of the same columns on every open.

        An absent version stamp is likewise not listed. Every dataset written before
        the marker existed lacks one, and the marker is bookkeeping this SDK invented
        for its own benefit -- nothing reads it at query time. Reporting its absence as
        a pending upgrade would make the entire existing installed base look broken,
        and on SQLite, where there is no stored procedure to miss, it would be the only
        thing ever reported."""
        pending = []
        if not self.interval_union_procedure_current():
            pending.append("the insert_interval_union stored procedure")
        return pending

    @abstractmethod
    def upgrade_mrn_schema(self):
        """Upgrade the patient table mrn column from INTEGER to TEXT if needed."""
        pass

    @abstractmethod
    def check_mrn_column_is_text(self) -> bool:
        """Check if the mrn column in the patient table is TEXT/VARCHAR. Returns True if it is TEXT."""
        pass

    @abstractmethod
    def _column_exists(self, cursor, table_name: str, column_name: str) -> bool:
        """Check if a column exists in a table."""
        pass

    def select_all_devices(self):
        with self.connection() as (conn, cursor):
            cursor.execute("SELECT id, tag, name, manufacturer, model, type, bed_id, source_id FROM device")
            rows = cursor.fetchall()
        return rows

    @abstractmethod
    def select_all_measures(self):
        pass

    @abstractmethod
    def select_all_patients(self):
        pass

    def select_patient_history(self, patient_id: int, field: Optional[str], start_time: int, end_time: int):
        # Dynamically construct the query based on whether `field` is None
        if field is None:
            query = "SELECT id, patient_id, field, value, units, time FROM patient_history WHERE patient_id = ? AND time BETWEEN ? AND ? ORDER BY time"
            query_params = (int(patient_id), int(start_time), int(end_time))
        else:
            query = "SELECT id, patient_id, field, value, units, time FROM patient_history WHERE patient_id = ? AND field = ? AND time BETWEEN ? AND ? ORDER BY time"
            query_params = (int(patient_id), field, int(start_time), int(end_time))

        with self.connection(begin=False) as (conn, cursor):
            cursor.execute(query, query_params)
            return cursor.fetchall()

    def select_closest_patient_history(self, patient_id: int, field: str, time: int):
        # Find the patient history that is closest to the timestamp
        query = "SELECT id, patient_id, field, value, units, time FROM patient_history WHERE patient_id = ? and field = ? and time <= ? ORDER BY time DESC LIMIT 1"
        with self.connection(begin=False) as (conn, cursor):
            cursor.execute(query, (int(patient_id), field, int(time)))
            return cursor.fetchone()

    def select_unique_history_fields(self) -> List[str]:
        query = "SELECT DISTINCT field FROM patient_history"
        with self.connection(begin=False) as (conn, cursor):
            cursor.execute(query)
            # Fetch all results
            results = cursor.fetchall()
            # Extract field values from the results
            fields = [row[0] for row in results]
        return fields

    @abstractmethod
    def insert_measure(self, measure_tag: str, freq_nhz: int, units: Optional[str] = None, measure_name: Optional[str] = None,
                       measure_id: Optional[int] = None, code: Optional[str] = None, unit_label: Optional[str] = None, unit_code: Optional[str] = None,
                       source_id: Optional[int] = None, period_ns: Optional[int] = None, signal_kind: Optional[str] = None,
                       value_type: Optional[str] = None):
        pass

    @abstractmethod
    def select_measure(self, measure_id: Optional[int] = None, measure_tag: Optional[str] = None, freq_nhz: Optional[int] = None, units: Optional[str] = None):
        # Select a measure either by its id, or by a tag, freq, units triplet.
        pass

    @abstractmethod
    def insert_device(self, device_tag: str, device_name: Optional[str] = None, device_id: Optional[int] = None, manufacturer: Optional[str] = None,
                      model: Optional[str] = None, device_type: Optional[str] = None, bed_id: Optional[int] = None, source_id: Optional[int] = None):
        pass

    def select_device(self, device_id: Optional[int] = None, device_tag: Optional[str] = None):
        # Select a measure either by its id, or by its unique tag.
        device_columns = "id, tag, name, manufacturer, model, type, bed_id, source_id"
        with self.connection() as (conn, cursor):
            if device_id is not None:
                cursor.execute(f"SELECT {device_columns} FROM device WHERE id = ?", (device_id,))
            else:
                cursor.execute(f"SELECT {device_columns} FROM device WHERE tag = ?", (device_tag,))
            row = cursor.fetchone()
        return row

    @abstractmethod
    def insert_tsc_file_data(self, file_path: str, block_data: List[Dict], interval_data: List[Dict],
                             interval_index_mode: str, gap_tolerance: int = 0):
        # Insert a file path to file index.
        # Insert block_index rows with foreign key file_id.
        # Insert interval_index rows.
        pass

    def insert_tsc_file_blocks(self, file_path: str, block_data: List[Dict]):
        with self.connection(begin=True) as (conn, cursor):
            # insert file_path into file_index and get id
            cursor.execute("INSERT INTO file_index (path) VALUES (?);", (file_path,))
            file_id = cursor.lastrowid

            # insert into block_index
            block_query = """INSERT INTO block_index
                (measure_id, device_id, file_id, start_byte, num_bytes, start_time_n, end_time_n, num_values)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?);"""
            cursor.executemany(block_query, block_insert_tuples(block_data, file_id))

    def update_block_times(self,
                           block_ids: List[int],
                           time_ranges: List[Tuple[int, int]]):
        """
        Update start_time_n and end_time_n for multiple blocks.

        :param block_ids:        list of block_index.id values
        :param time_ranges:      list of (start_time_n, end_time_n) tuples,
                                 same order as block_ids
        """
        if len(block_ids) != len(time_ranges):
            raise ValueError("block_ids and time_ranges must be the same length")

        update_sql = """
            UPDATE block_index
               SET start_time_n = ?,
                   end_time_n   = ?
             WHERE id            = ?;
        """

        # prepare the list of (start, end, id) tuples
        params = [
            (start, end, blk_id)
            for blk_id, (start, end) in zip(block_ids, time_ranges)
        ]

        with self.connection(begin=True) as (conn, cursor):
            cursor.executemany(update_sql, params)

    def insert_intervals(self, interval_data):
        with self.connection(begin=True) as (conn, cursor):
            interval_index_query = "INSERT INTO interval_index (measure_id, device_id, start_time_n, end_time_n) VALUES (?, ?, ?, ?);"
            cursor.executemany(interval_index_query, interval_insert_tuples(interval_data))

    def insert_and_delete_tsc_file_data(self, file_path: str, block_data: List[Dict], block_ids_to_delete: List[int]):
        with self.connection(begin=True) as (conn, cursor):
            # Insert file_path into file_index and get id
            cursor.execute("INSERT INTO file_index (path) VALUES (?);", (file_path,))
            file_id = cursor.lastrowid

            # Insert into block_index
            insert_block_query = """INSERT INTO block_index (measure_id, device_id, file_id, start_byte, num_bytes,
                start_time_n, end_time_n, num_values) VALUES (?, ?, ?, ?, ?, ?, ?, ?);"""
            cursor.executemany(insert_block_query, block_insert_tuples(block_data, file_id))

            # Delete blocks with matching block ids in the block_ids list
            if block_ids_to_delete:
                delete_query = "DELETE FROM block_index WHERE id = ?;"
                cursor.executemany(delete_query, [(block_id,) for block_id in block_ids_to_delete])

    def replace_intervals(self, measure_id: int, device_id: int, interval_list: List[List[int]],
                          batch_size: int = 1024):
        if len(interval_list) == 0:
            raise ValueError("This function deletes and replaces all intervals. `interval_list` cannot be empty")

        with self.connection(begin=True) as (conn, cursor):
            # Delete existing intervals for the given measure_id and device_id
            delete_query = """
                DELETE FROM interval_index
                WHERE measure_id = ? AND device_id = ?;
            """
            cursor.execute(delete_query, (measure_id, device_id))

            # Prepare interval tuples
            interval_tuples = [(int(measure_id), int(device_id), int(start_time), int(end_time))
                               for (start_time, end_time) in interval_list]

            # Insert new intervals in batches
            insert_query = """
                INSERT INTO interval_index (measure_id, device_id, start_time_n, end_time_n)
                VALUES (?, ?, ?, ?);
            """

            # Process intervals in batches
            for i in range(0, len(interval_tuples), batch_size):
                batch = interval_tuples[i:i + batch_size]
                cursor.executemany(insert_query, batch)

    def _interval_optimizer_lock_clause(self) -> str:
        """SQL suffix used while an interval-optimizer batch is being changed.

        SQLite obtains its writer lock on the first change, while InnoDB needs
        explicit row locks to keep a concurrent merge-mode writer from changing
        one of the rows selected by the optimizer.  Backends which need the
        latter override this with ``" FOR UPDATE"``.
        """
        return ""

    def optimize_interval_index(self, gap_tolerance_by_measure: Dict[int, int], *,
                                measure_id: Optional[int] = None,
                                device_id: Optional[int] = None,
                                batch_size: int = 10_000) -> Dict[str, int]:
        """Coalesce legacy interval-index rows without loading a stream into memory.

        This is deliberately an in-place, keyset-paginated maintenance operation.
        Each transaction reads and changes at most ``batch_size`` rows from one
        measure/device pair.  Its only destructive operation is a delete by the
        primary keys read in that same transaction, so an interval inserted by a
        concurrent writer is never deleted by this method.  A writer can add a
        row just behind the scan cursor; that row is simply considered by the
        next optimizer run, which is the appropriate eventually-convergent
        behavior for an index that remains online during ingestion.

        ``gap_tolerance_by_measure`` must map every selected measure to its
        desired tolerance in nanoseconds.  It is kept outside this low-level
        method because resolving the SDK's smart default requires measure
        metadata.
        """
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if measure_id is not None:
            measure_id = int(measure_id)
        if device_id is not None:
            device_id = int(device_id)
        if device_id is not None and measure_id is None:
            raise ValueError("device_id requires measure_id")

        tolerances = {int(key): int(value) for key, value in gap_tolerance_by_measure.items()}
        if any(value < 0 for value in tolerances.values()):
            raise ValueError("gap tolerances must be non-negative")

        stats = {"pairs_processed": 0, "rows_examined": 0, "rows_merged": 0}
        pair_cursor = None
        while True:
            clauses, params = [], []
            if measure_id is not None:
                clauses.append("measure_id = ?")
                params.append(measure_id)
            if device_id is not None:
                clauses.append("device_id = ?")
                params.append(device_id)
            if pair_cursor is not None:
                clauses.append("(measure_id > ? OR (measure_id = ? AND device_id > ?))")
                params.extend((pair_cursor[0], pair_cursor[0], pair_cursor[1]))
            where = " WHERE " + " AND ".join(clauses) if clauses else ""
            with self.connection(begin=False) as (conn, cursor):
                cursor.execute(
                    "SELECT measure_id, device_id FROM interval_index" + where +
                    " GROUP BY measure_id, device_id ORDER BY measure_id, device_id LIMIT ?",
                    tuple(params + [batch_size]))
                pairs = cursor.fetchall()

            if not pairs:
                break
            for pair_measure_id, pair_device_id in pairs:
                pair_measure_id, pair_device_id = int(pair_measure_id), int(pair_device_id)
                tolerance = tolerances.get(pair_measure_id)
                if tolerance is None:
                    raise ValueError(f"No gap tolerance was provided for measure_id={pair_measure_id}")
                pair_stats = self._optimize_interval_index_pair(
                    pair_measure_id, pair_device_id, tolerance, batch_size)
                stats["pairs_processed"] += 1
                stats["rows_examined"] += pair_stats["rows_examined"]
                stats["rows_merged"] += pair_stats["rows_merged"]
            pair_cursor = (int(pairs[-1][0]), int(pairs[-1][1]))
            # A bounded pair page also prevents a database with millions of
            # measure/device combinations from consuming memory in this pass.
            if len(pairs) < batch_size:
                break
        return stats

    def _optimize_interval_index_pair(self, measure_id: int, device_id: int,
                                      gap_tolerance: int, batch_size: int) -> Dict[str, int]:
        """Merge one pair using one retained row plus one bounded source page.

        The retained row is persisted after every page.  Thus even a single
        continuous run containing hundreds of millions of old rows uses constant
        Python memory and no transaction has to retain the whole run's locks or
        undo log.
        """
        stats = {"rows_examined": 0, "rows_merged": 0}
        last_start, last_id = None, None
        carry_id = carry_start = carry_end = None
        lock_clause = self._interval_optimizer_lock_clause()

        while True:
            with self.connection(begin=True) as (conn, cursor):
                if last_start is None:
                    cursor.execute(
                        "SELECT id, start_time_n, end_time_n FROM interval_index "
                        "WHERE measure_id = ? AND device_id = ? "
                        "ORDER BY start_time_n, id LIMIT ?" + lock_clause,
                        (measure_id, device_id, batch_size))
                else:
                    cursor.execute(
                        "SELECT id, start_time_n, end_time_n FROM interval_index "
                        "WHERE measure_id = ? AND device_id = ? AND "
                        "(start_time_n > ? OR (start_time_n = ? AND id > ?)) "
                        "ORDER BY start_time_n, id LIMIT ?" + lock_clause,
                        (measure_id, device_id, last_start, last_start, last_id, batch_size))
                rows = cursor.fetchall()
                if not rows:
                    break

                # A concurrent writer may have absorbed the retained row between
                # pages.  Refresh it under the same transaction; when it no longer
                # exists the current page starts a new retained interval.  Nothing
                # is lost, and a later pass will compact any newly bridged boundary.
                if carry_id is not None:
                    cursor.execute(
                        "SELECT id, start_time_n, end_time_n FROM interval_index WHERE id = ?" + lock_clause,
                        (carry_id,))
                    carry = cursor.fetchone()
                    if carry is None:
                        carry_id = carry_start = carry_end = None
                    else:
                        carry_id, carry_start, carry_end = map(int, carry)

                rows_to_delete = []
                for row_id, start, end in rows:
                    row_id, start, end = int(row_id), int(start), int(end)
                    if carry_id is None:
                        carry_id, carry_start, carry_end = row_id, start, end
                    elif start - carry_end <= gap_tolerance:
                        carry_end = max(carry_end, end)
                        rows_to_delete.append((row_id,))
                    else:
                        cursor.execute(
                            "UPDATE interval_index SET start_time_n = ?, end_time_n = ? WHERE id = ?",
                            (carry_start, carry_end, carry_id))
                        carry_id, carry_start, carry_end = row_id, start, end

                cursor.execute(
                    "UPDATE interval_index SET start_time_n = ?, end_time_n = ? WHERE id = ?",
                    (carry_start, carry_end, carry_id))
                if rows_to_delete:
                    cursor.executemany("DELETE FROM interval_index WHERE id = ?", rows_to_delete)

                stats["rows_examined"] += len(rows)
                stats["rows_merged"] += len(rows_to_delete)
                last_start, last_id = int(rows[-1][1]), int(rows[-1][0])
        return stats

    @abstractmethod
    def update_tsc_file_data(self, file_data: Dict[str, Tuple[List[Dict], List[Dict]]], block_ids_to_delete: List[int],
                             file_ids_to_delete: List[int], gap_tolerance: int = 0):
        pass

    @abstractmethod
    def insert_merged_block_data(self, file_path: str, block_data: List[Dict], old_block_id: int, interval_data: List[Dict],
                                 interval_index_mode: str, gap_tolerance: int = 0):
        pass

    @staticmethod
    def _delete_merged_block(cursor, old_block: tuple):
        """Drop the block a merge superseded, and unlink its tsc file if that block was
        the last one in it. Returns the orphaned file's path for the caller to delete
        from disk, or ``None`` when the file still holds other blocks.

        Runs inside the caller's transaction, on the caller's cursor.
        """
        # delete the old block data
        cursor.execute("DELETE FROM block_index WHERE id = ?", (old_block[0],))

        # check if the old tsc file only contains the old block
        cursor.execute("SELECT 1 FROM block_index WHERE file_id = ? LIMIT 1", (old_block[3],))
        block_exists = cursor.fetchone()

        # if there are no blocks with that file_id then delete the file from the file index
        if block_exists is None:
            # get the tsc file name
            cursor.execute("SELECT path FROM file_index WHERE id = ?", (old_block[3],))
            file_name = cursor.fetchone()

            # delete it from the file_index
            cursor.execute("DELETE FROM file_index WHERE id = ?", (old_block[3],))
            # A racing writer that merged into the same old block can have removed the
            # file_index row already. The merge is serialized by a per-(measure,
            # device) lock so this should be unreachable, but indexing an absent row
            # would abort this transaction and lose the write entirely. The row
            # genuinely being gone means only that someone else has already unlinked
            # the file -- there is nothing left for the caller to remove, which is
            # what None means.
            return file_name[0] if file_name is not None else None

        return None

    def select_file(self, file_id: Optional[int] = None, file_path: Optional[str] = None):
        # Select a file path either by its id, or by its path.
        with self.connection() as (conn, cursor):
            if file_id is not None:
                cursor.execute("SELECT id, path FROM file_index WHERE id = ?;", (file_id,))
            else:
                cursor.execute("SELECT id, path FROM file_index WHERE path = ?;", (file_path,))
            row = cursor.fetchone()
        return row

    def select_files(self, file_id_list: List[int]):
        """Every ``file_index`` row named by ``file_id_list``.

        Raises ``RuntimeError`` if any requested id is absent: callers use this to
        resolve the files a block read is about to open, so a missing row means the
        read would fail later, further from the cause. An empty list asks for nothing
        and gets ``[]`` (see :meth:`_select_rows_in_list`)."""
        rows = self._select_rows_in_list("id, path", "file_index", file_id_list)

        # Compare against the DISTINCT ids asked for -- a repeated id yields one row.
        if len(rows) != len(set(file_id_list)):
            missing = set(file_id_list) - {row[0] for row in rows}
            raise RuntimeError(f"Cannot find file_ids={missing} in AtriumDB.")

        return rows

    def select_blocks_from_file(self, file_id: int):
        # Selects all blocks from file_id
        with self.connection() as (conn, cursor):
            cursor.execute(
                "SELECT id, measure_id, device_id, file_id, start_byte, num_bytes, start_time_n, end_time_n, "
                "num_values FROM block_index WHERE file_id = ? ORDER BY start_byte;", (file_id,))
            rows = cursor.fetchall()
        return rows

    def select_blocks_by_ids(self, block_id_list: List[int | str]):
        """Every ``block_index`` row named by ``block_id_list``, in read order
        (``file_id, start_byte``) so a caller can read them with sequential seeks.

        Raises ``RuntimeError`` if any requested id is absent, for the same reason as
        :meth:`select_files`. Ids may arrive as strings (they come off a URL query in
        API mode), hence the ``int`` coercion when reporting.
        """
        rows = self._select_rows_in_list(
            "id, measure_id, device_id, file_id, start_byte, num_bytes, start_time_n, end_time_n, num_values",
            "block_index", block_id_list, order_by="file_id, start_byte ASC")

        # Compare against the DISTINCT ids asked for. Comparing against the raw list
        # length instead would treat a repeated id as a missing row and then report
        # `block_ids=set()` -- an error naming nothing.
        requested = {int(block_id) for block_id in block_id_list}
        if len(rows) != len(requested):
            missing = requested - {row[0] for row in rows}
            raise RuntimeError(f"Cannot find block_ids={missing} in AtriumDB.")

        return rows

    @abstractmethod
    def select_block(self, block_id: Optional[int] = None, measure_id: Optional[int] = None, device_id: Optional[int] = None, file_id: Optional[int] = None,
                     start_byte: Optional[int] = None, num_bytes: Optional[int] = None, start_time_n: Optional[int] = None, end_time_n: Optional[int] = None,
                     num_values: Optional[int] = None):
        # Select a block either by its id or by all other params.
        pass

    def select_closest_block(self, measure_id: int, device_id: int, start_time: int, end_time: int):
        base_query = """SELECT id, measure_id, device_id, file_id, start_byte, num_bytes, start_time_n, end_time_n, num_values 
        FROM block_index WHERE measure_id = ? and device_id = ? """

        with self.connection(begin=False) as (conn, cursor):
            # First check if this block belongs on the end (most likely scenario)
            end_check_query = base_query + "ORDER BY end_time_n DESC LIMIT 1"
            cursor.execute(end_check_query, (int(measure_id), int(device_id)))
            result = cursor.fetchone()

            # Check if the start time is >= the max end time meaning the block goes on the end
            if result is not None and int(start_time) >= result[7]:
                # Return true meaning it goes on the end. We will check if the last block is full in write_data so we
                # need to know if this is an end block
                return result, True

            # If it overlaps with the last block
            if result is not None and int(start_time) <= result[7] and int(end_time) >= result[6]:
                return result, False

            # Check if there is a block that this data fits inside
            inside_query = base_query + "and start_time_n <= ? and end_time_n >= ? LIMIT 1"

            cursor.execute(inside_query, (int(measure_id), int(device_id), int(start_time), int(end_time)))
            result = cursor.fetchone()

            # If there is a block this data belongs inside return it
            if result is not None:
                return result, False

            # Get the closest block whose start time is <= my start time
            inside_query = base_query + "and start_time_n <= ? and end_time_n <= ? ORDER BY start_time_n DESC, end_time_n DESC LIMIT 1"
            cursor.execute(inside_query, (int(measure_id), int(device_id), int(start_time), int(end_time)))
            block_older = cursor.fetchone()

            # Get the closest block whose end time is >= my end time
            inside_query = base_query + "and start_time_n >= ? and end_time_n >= ? ORDER BY end_time_n ASC, start_time_n ASC LIMIT 1"
            cursor.execute(inside_query, (int(measure_id), int(device_id), int(start_time), int(end_time)))
            block_newer = cursor.fetchone()

            # Subtract the end time of the old block from the start time of the new block to see how far apart they are
            # If they overlap the number will become negative and therefore they are closer
            older_diff = int(start_time) - block_older[7] if block_older is not None else None

            # Subtract the start time of the old block from the end time of the new block to see how far apart they are
            # If they overlap the number will become negative and therefore they are closer
            newer_diff = block_newer[6] - int(end_time) if block_newer is not None else None

            if older_diff is None and newer_diff is not None:
                return block_newer, False
            elif newer_diff is None and older_diff is not None:
                return block_older, False
            elif older_diff is None and newer_diff is None:
                # Need this since if it's the first block there will be nothing to merge with
                return None, False
            elif older_diff <= newer_diff:
                return block_older, False
            elif older_diff > newer_diff:
                return block_newer, False

    def delete_block(self, block_id: int):
        with self.connection(begin=True) as (conn, cursor):
            # Delete block data
            cursor.execute("DELETE FROM block_index WHERE id = ?", (int(block_id),))

    def select_blocks_for_device(self, device_id: int, measure_ids: int|List[int] = None):
        """
        Fetch block index data for the device (and measures if specified).
        """
        # Normalize measure_ids to a list if it's an integer
        if measure_ids is not None:
            if isinstance(measure_ids, int):
                measure_ids = [measure_ids]
            elif not isinstance(measure_ids, list):
                raise TypeError("measure_ids must be an int, a list of ints, or None.")

            # None means "every measure"; an EMPTY list means "these measures", of which
            # there are none. Falling through would build `IN ()` -- accepted by SQLite,
            # a syntax error on MariaDB.
            if len(measure_ids) == 0:
                return []

            # Build placeholders for SQL IN clause
            placeholders = ','.join(['?'] * len(measure_ids))
            block_query = f"""
            SELECT id, measure_id, device_id, file_id, start_byte, num_bytes, start_time_n, end_time_n, num_values
            FROM block_index
            WHERE device_id = ? AND measure_id IN ({placeholders})
            ORDER BY measure_id, device_id, start_time_n ASC;
            """
            args = [int(device_id)] + [int(mid) for mid in measure_ids]
        else:
            # Fetch blocks for the device across all measures
            block_query = """
            SELECT id, measure_id, device_id, file_id, start_byte, num_bytes, start_time_n, end_time_n, num_values
            FROM block_index
            WHERE device_id = ?
            ORDER BY measure_id, device_id, start_time_n ASC;
            """
            args = [int(device_id)]

        with self.connection(begin=False) as (conn, cursor):
            cursor.execute(block_query, args)
            return cursor.fetchall()

    def select_blocks_for_devices(self, device_ids: List[int], measure_ids: List[int]):
        # Both filters are required here, so either being empty selects no rows. Build
        # nothing: `IN ()` is a syntax error on MariaDB (SQLite tolerates it).
        if len(device_ids) == 0 or len(measure_ids) == 0:
            return []

        # Build placeholders for SQL IN clauses
        device_placeholders = ','.join(['?'] * len(device_ids))
        measure_placeholders = ','.join(['?'] * len(measure_ids))

        block_query = f"""
        SELECT id, measure_id, device_id, file_id, start_byte, num_bytes, start_time_n, end_time_n, num_values
        FROM block_index
        WHERE device_id IN ({device_placeholders}) AND measure_id IN ({measure_placeholders})
        ORDER BY measure_id, device_id, start_time_n ASC;
        """

        # Prepare arguments for the SQL query
        args = [int(did) for did in device_ids] + [int(mid) for mid in measure_ids]

        with self.connection(begin=False) as (conn, cursor):
            cursor.execute(block_query, args)
            return cursor.fetchall()

    def select_interval(self, interval_id: Optional[int] = None, measure_id: Optional[int] = None, device_id: Optional[int] = None,
                        start_time_n: Optional[int] = None, end_time_n: Optional[int] = None):
        # Select an interval either by its id, or by all other params.
        interval_columns = "id, measure_id, device_id, start_time_n, end_time_n"
        with self.connection() as (conn, cursor):
            if interval_id is not None:
                cursor.execute(f"SELECT {interval_columns} FROM interval_index WHERE id = ?;", (interval_id,))
            else:
                cursor.execute(
                    f"SELECT {interval_columns} FROM interval_index WHERE measure_id = ? AND device_id = ? "
                    "AND start_time_n = ? AND end_time_n = ?;",
                    (measure_id, device_id, start_time_n, end_time_n))
            row = cursor.fetchone()
        return row

    @abstractmethod
    def insert_setting(self, setting_name: str, setting_value: str):
        # Inserts a setting into the database.
        pass

    def select_setting(self, setting_name: str):
        # Selects a setting from the database.
        with self.connection() as (conn, cursor):
            cursor.execute("SELECT name, value FROM setting WHERE name = ?", (setting_name,))
            setting = cursor.fetchone()
        return setting

    @abstractmethod
    def insert_patient(self, patient_id: Optional[int] = None, mrn: Optional[str] = None, gender: Optional[str] = None, dob: Optional[str] = None,
                       first_name: Optional[str] = None, middle_name: Optional[str] = None, last_name: Optional[str] = None, first_seen: Optional[int] = None,
                       last_updated: Optional[int] = None, source_id: int = 1, weight: Optional[float] = None, height: Optional[float] = None):
        # Insert patient if it doesn't exist, return id.
        pass

    def insert_patient_history(self, patient_id: int, field: str, value: float, units: str, time: int):
        with self.connection(begin=True) as (conn, cursor):
            # Find the most recent value for the field you're entering from the patient history table
            cursor.execute("SELECT MAX(time) FROM patient_history WHERE patient_id = ? and field = ?", (int(patient_id), field))
            newest_measurement_time = cursor.fetchone()[0]

            # If the new measurement is newer than the newest in the patient history table, update the field in the
            # patient table. If no history is found, also update it.
            if newest_measurement_time is None or int(time) > newest_measurement_time:
                cursor.execute(f"UPDATE patient SET {field} = {value} WHERE id = {int(patient_id)}")

            # Now insert the row to the patient history table
            query = "INSERT INTO patient_history (patient_id, field, value, units, time) VALUES (?, ?, ?, ?, ?)"
            cursor.execute(query, (int(patient_id), field, float(value), units, int(time)))
            conn.commit()
            return cursor.lastrowid

    @abstractmethod
    def insert_encounter(self, patient_id: int, bed_id: int, start_time: int, end_time: Optional[int] = None, source_id: int = 1,
                         visit_number: Optional[int] = None, last_updated: Optional[int] = None):
        # Insert encounter if it doesn't exist, return id.
        pass

    @abstractmethod
    def insert_device_encounter(self, device_id: int, encounter_id: int, start_time: int, end_time: Optional[int] = None,
                                source_id: int = 1):
        # Insert device_encounter if it doesn't exist, return id.
        pass

    def get_device_time_ranges_by_patient(self, patient_id: int, end_time_n: Optional[int],
                                          start_time_n: Optional[int]):
        patient_device_query = "SELECT device_id, start_time, end_time FROM device_patient WHERE patient_id = ?"
        args = (int(patient_id),)
        if start_time_n is not None:
            patient_device_query += " AND (end_time >= ? OR end_time is NULL) "
            args += (int(start_time_n),)
        if end_time_n is not None:
            patient_device_query += " AND start_time <= ? "
            args += (int(end_time_n),)
        patient_device_query += " ORDER BY start_time, end_time"
        with self.connection(begin=False) as (conn, cursor):
            cursor.execute(patient_device_query, args)
            results = cursor.fetchall()

            # Replace None end_time with current nanosecond epoch
            current_time_ns = time.time_ns()
            results = [
                (device_id, start_time, end_time if end_time is not None else current_time_ns)
                for device_id, start_time, end_time in results
            ]

            return results

    def select_all_settings(self):
        # Select all settings from settings table.
        with self.connection() as (conn, cursor):
            cursor.execute("SELECT name, value FROM setting")
            settings = cursor.fetchall()
        return settings

    def select_blocks(self, measure_id: int, start_time_n: Optional[int] = None, end_time_n: Optional[int] = None, device_id: Optional[int] = None, patient_id: Optional[int] = None):
        # Get all matching blocks.
        assert device_id is not None or patient_id is not None, "Either device_id or patient_id must be provided"

        block_columns = ("id, measure_id, device_id, file_id, start_byte, num_bytes, start_time_n, end_time_n, "
                         "num_values")

        # Query by patient.
        if patient_id is not None:
            device_time_ranges = self.get_device_time_ranges_by_patient(patient_id, end_time_n, start_time_n)

            block_query = (f"SELECT {block_columns} FROM block_index "
                           "WHERE measure_id = ? AND device_id = ? AND end_time_n >= ? AND start_time_n <= ? "
                           "ORDER BY file_id, start_byte ASC")

            block_results = []

            with self.connection(begin=False) as (conn, cursor):
                for encounter_device_id, encounter_start_time, encounter_end_time in device_time_ranges:
                    args = (measure_id, encounter_device_id, encounter_start_time, encounter_end_time)

                    cursor.execute(block_query, args)
                    block_results.extend(cursor.fetchall())

            # Filter results based on start_time_n and end_time_n parameters
            filtered_results = []
            for row in block_results:
                row_start_time = row[6]  # start_time_n
                row_end_time = row[7]  # end_time_n

                # Check for overlap with parameter time range
                if start_time_n is not None and row_end_time < start_time_n:
                    continue  # Row ends before parameter start - no overlap
                if end_time_n is not None and row_start_time > end_time_n:
                    continue  # Row starts after parameter end - no overlap

                filtered_results.append(row)

            return filtered_results

        # Query by device.
        block_query = f"SELECT {block_columns} FROM block_index WHERE measure_id = ? AND device_id = ?"

        args = (measure_id, device_id)

        if end_time_n is not None:
            block_query += " AND start_time_n <= ?"
            args += (end_time_n,)

        if start_time_n is not None:
            block_query += " AND end_time_n >= ?"
            args += (start_time_n,)

        block_query += " ORDER BY file_id, start_byte ASC"

        with self.connection(begin=False) as (conn, cursor):
            cursor.execute(block_query, args)
            return cursor.fetchall()

    def select_intervals(self, measure_id: int, start_time_n: Optional[int] = None, end_time_n: Optional[int] = None, device_id: Optional[int] = None, patient_id: Optional[int] = None):
        if device_id is None and patient_id is None:
            raise ValueError("Either device_id or patient_id must be provided")

        interval_query = "SELECT id, measure_id, device_id, start_time_n, end_time_n FROM interval_index WHERE measure_id = ? AND device_id = ?"

        # Query by patient.
        if patient_id is not None:
            device_time_ranges = self.get_device_time_ranges_by_patient(patient_id, end_time_n, start_time_n)
            # Add start and end time to the query
            interval_query += " AND end_time_n >= ? AND start_time_n <= ? ORDER BY start_time_n ASC, end_time_n ASC"

            interval_results = []
            with self.connection(begin=False) as (conn, cursor):
                for encounter_device_id, encounter_start_time, encounter_end_time in device_time_ranges:
                    encounter_end_time = time.time_ns() if encounter_end_time is None else encounter_end_time
                    args = (int(measure_id), int(encounter_device_id), int(encounter_start_time), int(encounter_end_time))

                    cursor.execute(interval_query, args)

                    #  Truncate Intervals to the Start, End of Encounter
                    encounter_intervals = cursor.fetchall()
                    encounter_intervals = [[interval_id,
                                            measure_id,
                                            device_id,
                                            max(start_time_n, encounter_start_time),
                                            min(end_time_n, encounter_end_time)]
                                           for interval_id, measure_id, device_id, start_time_n, end_time_n
                                           in encounter_intervals]

                    interval_results.extend(encounter_intervals)

            return interval_results

        # Query by device.
        args = (int(measure_id), int(device_id))

        if end_time_n is not None:
            interval_query += " AND start_time_n <= ?"
            args += (int(end_time_n),)

        if start_time_n is not None:
            interval_query += " AND end_time_n >= ?"
            args += (int(start_time_n),)

        # Add the ordering
        interval_query += " ORDER BY start_time_n ASC, end_time_n ASC"

        with self.connection(begin=False) as (conn, cursor):
            cursor.execute(interval_query, args)
            return cursor.fetchall()

    @abstractmethod
    def select_encounters(self, patient_id_list: Optional[List[int]] = None, mrn_list: Optional[List[str]] = None, start_time: Optional[int] = None,
                          end_time: Optional[int] = None):
        # Get all matching encounters.
        pass

    def _select_rows_in_list(self, columns: str, table: str, id_list, id_column: str = "id",
                             order_by: str = None):
        """Select ``columns`` from ``table`` for every row whose ``id_column`` appears
        in ``id_list``.

        ``columns``/``table``/``id_column``/``order_by`` are literals from the call sites
        below, never caller input; only the ids are bound, one placeholder each. Both
        backends take ``?`` placeholders, so a single implementation serves them.

        An empty ``id_list`` returns ``[]`` without touching the database. That is not
        just an optimisation: the alternative is the degenerate SQL ``IN ()``, which
        SQLite quietly accepts as "matches nothing" while MariaDB rejects as a syntax
        error -- so the two backends disagreed on the empty case until this guard.
        Matching no ids is a legitimate query with an empty answer, not an error.
        """
        if len(id_list) == 0:
            return []
        placeholders = ', '.join(['?'] * len(id_list))
        query = f"SELECT {columns} FROM {table} WHERE {id_column} IN ({placeholders})"
        if order_by is not None:
            query += f" ORDER BY {order_by}"
        with self.connection() as (conn, cursor):
            cursor.execute(query, id_list)
            rows = cursor.fetchall()
        return rows

    def select_all_measures_in_list(self, measure_id_list: List[int]):
        # Get all matching measures.
        return self._select_rows_in_list(
            "id, tag, name, freq_nhz, code, unit, unit_label, unit_code, source_id", "measure", measure_id_list)

    @abstractmethod
    def select_all_patients_in_list(self, patient_id_list: Optional[List[int]] = None, mrn_list: Optional[List[str]] = None):
        # Get all matching patients.
        pass

    def select_all_devices_in_list(self, device_id_list: List[int]):
        # Get all matching devices.
        return self._select_rows_in_list(
            "id, tag, name, manufacturer, model, type, bed_id, source_id", "device", device_id_list)

    def select_all_beds_in_list(self, bed_id_list: List[int]):
        # Get all matching beds.
        return self._select_rows_in_list("id, unit_id, name", "bed", bed_id_list)

    def select_all_units_in_list(self, unit_id_list: List[int]):
        # Get all matching units.
        return self._select_rows_in_list("id, institution_id, name, type", "unit", unit_id_list)

    def select_all_institutions_in_list(self, institution_id_list: List[int]):
        # Get all matching institutions.
        return self._select_rows_in_list("id, name", "institution", institution_id_list)

    def select_all_device_encounters_by_encounter_list(self, encounter_id_list: List[int]):
        # Get all matching device_encounters by encounter id list.
        return self._select_rows_in_list(
            "id, device_id, encounter_id, start_time, end_time, source_id", "device_encounter",
            encounter_id_list, id_column="encounter_id")

    def select_all_sources_in_list(self, source_id_list: List[int]):
        # Get all matching sources.
        return self._select_rows_in_list("id, name, description", "source", source_id_list)

    def select_device_patients(self, device_id_list: List[int] = None, patient_id_list: List[int] = None,
                               start_time: int = None, end_time: int = None):
        sqlite_select_device_patient_query = \
            "SELECT device_id, patient_id, start_time, end_time FROM device_patient"

        where_clauses, arg_tuple = self._int_id_list_clauses(
            device_id=device_id_list, patient_id=patient_id_list)

        # Handle start_time
        if start_time is not None:
            where_clauses.append("(end_time > ? OR end_time IS NULL)")
            arg_tuple += (int(start_time),)

        # Handle end_time
        if end_time is not None:
            where_clauses.append("start_time < ?")
            arg_tuple += (int(end_time),)

        # Combine where clauses
        if where_clauses:
            sqlite_select_device_patient_query += " WHERE " + " AND ".join(where_clauses)

        sqlite_select_device_patient_query += " ORDER BY id ASC"

        with self.connection() as (conn, cursor):
            cursor.execute(sqlite_select_device_patient_query, arg_tuple)
            return cursor.fetchall()

    def select_device_patient_encounters(self, timestamp: int, device_id_list: List[int] = None,
                                         patient_id_list: List[int] = None):
        sql_select_query = (
            "SELECT device_id, patient_id, start_time, end_time "
            "FROM device_patient WHERE start_time <= ? AND (end_time > ? OR end_time IS NULL)"
        )

        where_clauses, id_args = self._int_id_list_clauses(
            device_id=device_id_list, patient_id=patient_id_list)
        arg_tuple = (timestamp, timestamp) + id_args

        # Combine where clauses
        if where_clauses:
            sql_select_query += " AND " + " AND ".join(where_clauses)

        sql_select_query += " ORDER BY id DESC"

        with self.connection() as (conn, cursor):
            cursor.execute(sql_select_query, arg_tuple)
            results = cursor.fetchall()
            return results

    def insert_encounter_row(self, patient_id: int, bed_id: int, start_time: int, end_time: int = None,
                         source_id: int = 1, visit_number: str = None, last_updated: int = None):
        query = """
            INSERT INTO encounter (patient_id, bed_id, start_time, end_time, source_id, visit_number, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        with self.connection() as (conn, cursor):
            cursor.execute(query, (patient_id, bed_id, start_time, end_time, source_id, visit_number, last_updated))
            conn.commit()

    def select_encounters_from_range_or_timestamp(
            self, timestamp: int = None, start_time: int = None, end_time: int = None,
            bed_id: int = None, patient_id: int = None):
        query = "SELECT id, patient_id, bed_id, start_time, end_time, source_id, visit_number, last_updated FROM encounter"
        where_clauses = []
        args = []

        if timestamp is not None:
            where_clauses.append("start_time <= ? AND (end_time > ? OR end_time IS NULL)")
            args.extend([timestamp, timestamp])

        if start_time is not None:
            where_clauses.append("(end_time > ? OR end_time IS NULL)")
            args.append(start_time)

        if end_time is not None:
            where_clauses.append("start_time < ?")
            args.append(end_time)

        if bed_id is not None:
            where_clauses.append("bed_id = ?")
            args.append(bed_id)

        if patient_id is not None:
            where_clauses.append("patient_id = ?")
            args.append(patient_id)

        if where_clauses:
            query += " WHERE " + " AND ".join(where_clauses)

        query += " ORDER BY start_time ASC"

        with self.connection() as (conn, cursor):
            cursor.execute(query, args)
            return cursor.fetchall()

    def insert_device_patients(self, device_patient_data: List[Tuple[int, int, int, int]]):
        # Insert device_patient rows.
        with self.connection() as (conn, cursor):
            cursor.executemany(
                "INSERT INTO device_patient (device_id, patient_id, start_time, end_time) VALUES (?, ?, ?, ?)",
                device_patient_data)
            conn.commit()

    def insert_label_set(self, name: str, label_set_id: Optional[int] = None, parent_id: Optional[int] = None):
        if label_set_id is not None:
            existing_label_set = self.select_label_set(label_set_id)
            if existing_label_set:
                if existing_label_set[1] == name and existing_label_set[2] == parent_id:
                    # The provided ID exists and matches the name
                    return label_set_id
                else:
                    # The provided ID exists but with a different name
                    raise ValueError(f"The id {label_set_id} already exists under label name {existing_label_set[1]} "
                                     f"and parent {existing_label_set[2]}")

        existing_label_set_id = self.select_label_set_id(name)
        if existing_label_set_id is not None:
            # The name already exists with a different ID
            return existing_label_set_id

        # Insert the new label set
        query = "INSERT INTO label_set (id, name, parent_id) VALUES (?, ?, ?)"
        with self.connection(begin=True) as (conn, cursor):
            cursor.execute(query, (label_set_id, name, parent_id,))
            conn.commit()
            return cursor.lastrowid if label_set_id is None else label_set_id

    def select_label_sets(self, limit=None, offset=None):
        # Retrieve all label types
        return self._select_all_ordered(
            "SELECT id, name, parent_id FROM label_set ORDER BY id ASC", limit, offset)

    def select_label_set(self, label_set_id: int):
        query = "SELECT id, name, parent_id FROM label_set WHERE id = ? LIMIT 1"
        with self.connection() as (conn, cursor):
            cursor.execute(query, (int(label_set_id),))
            row = cursor.fetchone()
        return row

    def select_label_set_id(self, name: str) -> Optional[int]:
        # Retrieve the ID of a label type by its name.
        query = "SELECT id FROM label_set WHERE name = ? LIMIT 1"
        with self.connection(begin=False) as (conn, cursor):
            cursor.execute(query, (name,))
            result = cursor.fetchone()
            # Return the ID if it exists or None otherwise.
            return result[0] if result else None

    def select_label_name_parent(self, label_set_id: int) -> Optional[Tuple[int, str]]:
        query = """
        SELECT parent.id, parent.name FROM label_set
        INNER JOIN label_set AS parent ON label_set.parent_id = parent.id
        WHERE {}
        LIMIT 1;
        """

        with self.connection() as (conn, cursor):
            cursor.execute(query.format("label_set.id = ?"), (int(label_set_id),))
            return cursor.fetchone()

    def select_all_ancestors(self, label_set_id: Optional[int] = None, name: Optional[str] = None) -> Optional[List[Tuple[int, str]]]:
        if label_set_id is None and name is None:
            raise ValueError("Either label_set_id or name must be provided")

        query = """
        WITH RECURSIVE ancestors(id, name, parent_id) AS (
            SELECT id, name, parent_id FROM label_set WHERE {}
            UNION ALL
            SELECT ls.id, ls.name, ls.parent_id FROM label_set ls
            INNER JOIN ancestors ON ls.id = ancestors.parent_id
        )
        SELECT id, name FROM ancestors WHERE id != ?;
        """

        with self.connection() as (conn, cursor):
            if label_set_id is not None:
                cursor.execute(query.format("id = ?"), (int(label_set_id), int(label_set_id)))
            else:
                query_for_name = """
                SELECT id FROM label_set WHERE name = ? LIMIT 1;
                """
                cursor.execute(query_for_name, (name,))
                row = cursor.fetchone()
                if row:
                    cursor.execute(query.format("id = ?"), (row[0], row[0]))
                else:
                    return None
            return cursor.fetchall()

    def select_label_name_children(self, label_set_id: int) -> List[Tuple[int, str]]:
        query = """
        SELECT id, name FROM label_set
        WHERE parent_id = ?
        """

        with self.connection() as (conn, cursor):
            cursor.execute(query, (int(label_set_id),))
            return cursor.fetchall()

    def select_all_label_name_descendents(self, label_set_id: int):

        query = """
        WITH RECURSIVE descendants(id, name, parent_id) AS (
            SELECT id, name, parent_id FROM label_set WHERE parent_id = ?
            UNION ALL
            SELECT ls.id, ls.name, ls.parent_id FROM label_set ls
            INNER JOIN descendants ON ls.parent_id = descendants.id
        )
        SELECT id, name, parent_id FROM descendants WHERE id != ?;
        """

        with self.connection() as (conn, cursor):
            cursor.execute(query, (int(label_set_id), int(label_set_id)))
            return cursor.fetchall()

    def insert_label(self, label_set_id, device_id, start_time_n, end_time_n, label_source_id=None, measure_id=None):
        # Insert a new label record into the database.
        query = "INSERT INTO label (label_set_id, device_id, measure_id, label_source_id, start_time_n, end_time_n) VALUES (?, ?, ?, ?, ?, ?)"
        with self.connection() as (conn, cursor):
            label_set_id = None if label_set_id is None else int(label_set_id)
            device_id = None if device_id is None else int(device_id)
            start_time_n = None if start_time_n is None else int(start_time_n)
            end_time_n = None if end_time_n is None else int(end_time_n)
            label_source_id = None if label_source_id is None else int(label_source_id)
            measure_id = None if measure_id is None else int(measure_id)
            cursor.execute(query, (label_set_id, device_id, measure_id, label_source_id, start_time_n, end_time_n))
            conn.commit()
            # Return the ID of the newly inserted label.
            return cursor.lastrowid

    def insert_labels(self, labels):
        # Insert multiple label records into the database.
        formatted_labels = []
        for label_set_id, device_id, measure_id, label_source_id, start_time_n, end_time_n in labels:
            label_set_id = None if label_set_id is None else int(label_set_id)
            device_id = None if device_id is None else int(device_id)
            start_time_n = None if start_time_n is None else int(start_time_n)
            end_time_n = None if end_time_n is None else int(end_time_n)
            label_source_id = None if label_source_id is None else int(label_source_id)
            measure_id = None if measure_id is None else int(measure_id)
            formatted_labels.append([label_set_id, device_id, measure_id, label_source_id, start_time_n, end_time_n])
        query = "INSERT INTO label (label_set_id, device_id, measure_id, label_source_id, start_time_n, end_time_n) VALUES (?, ?, ?, ?, ?, ?)"
        with self.connection(begin=True) as (conn, cursor):
            cursor.executemany(query, formatted_labels)
            conn.commit()
            # Return the ID of the last inserted label.
            return cursor.lastrowid

    def delete_labels(self, label_ids):
        # Delete multiple label records from the database based on their IDs.
        query = "DELETE FROM label WHERE id = ?"
        with self.connection() as (conn, cursor):
            # Prepare a list of tuples for the executemany method.
            id_tuples = [(int(label_id),) for label_id in label_ids]
            cursor.executemany(query, id_tuples)
            conn.commit()

    def select_labels(self, label_set_id_list=None, device_id_list=None, patient_id_list=None, start_time_n=None,
                      end_time_n=None, label_source_id_list=None, measure_id_list=None, limit=None, offset=None):
        if device_id_list is not None and patient_id_list is not None:
            raise ValueError("Can only request labels by device or patient, not both")

        # If provided patient IDs, fetch device time ranges and recursively call select_labels.
        if patient_id_list is not None:
            results = []
            for patient_id in patient_id_list:
                # Get device time ranges associated with a patient.
                device_time_ranges = self.get_device_time_ranges_by_patient(patient_id, end_time_n, start_time_n)

                for device_id, device_start_time, device_end_time in device_time_ranges:
                    # Adjust the time range based on the provided boundaries.
                    final_start_time = max(start_time_n, device_start_time) if start_time_n else device_start_time
                    final_end_time = min(end_time_n, device_end_time) if end_time_n else device_end_time

                    # Recursively fetch labels for each device and accumulate the results.
                    results.extend(self.select_labels(label_set_id_list=label_set_id_list, device_id_list=[device_id],
                                                      start_time_n=final_start_time, end_time_n=final_end_time,
                                                      label_source_id_list=label_source_id_list, measure_id_list=measure_id_list))

            # Sort the results by start_time_n primarily and then by end_time_n secondarily
            results.sort(key=lambda x: (x[5], x[6], x[0]))
            return results

        # Construct the query for selecting labels based on the provided criteria.
        query = "SELECT id, label_set_id, device_id, measure_id, label_source_id , start_time_n, end_time_n FROM label WHERE 1=1"
        params = []

        # Add conditions for label type IDs, if provided.
        if label_set_id_list:
            placeholders = ', '.join(['?'] * len(label_set_id_list))
            query += f" AND label_set_id IN ({placeholders})"
            params.extend(label_set_id_list)

        # Add conditions for device IDs, if provided.
        if device_id_list:
            placeholders = ', '.join(['?'] * len(device_id_list))
            query += f" AND device_id IN ({placeholders})"
            params.extend(device_id_list)

        # Add conditions for measure IDs, if provided.
        if measure_id_list:
            placeholders = ', '.join(['?'] * len(measure_id_list))
            query += f" AND measure_id IN ({placeholders})"
            params.extend(measure_id_list)

        # Add conditions for label source IDs, if provided.
        if label_source_id_list:
            placeholders = ', '.join(['?'] * len(label_source_id_list))
            query += f" AND label_source_id IN ({placeholders})"
            params.extend(label_source_id_list)

        # Add conditions for start and end times, if provided.
        if start_time_n:
            query += " AND end_time_n >= ?"
            params.append(int(start_time_n))
        if end_time_n:
            query += " AND start_time_n <= ?"
            params.append(int(end_time_n))

        # Sort by start_time_n
        # Used in iterator logic, alter with caution.
        query += " ORDER BY start_time_n ASC, end_time_n ASC, label.id ASC"

        query = self._append_limit_offset(query, limit, offset)

        # Execute the query and return the results.
        with self.connection(begin=False) as (conn, cursor):
            cursor.execute(query, params)
            return cursor.fetchall()


    def select_labels_with_info(self, label_set_id_list=None, device_id_list=None, patient_id_list=None,
                                start_time_n=None, end_time_n=None, label_source_id_list=None, measure_id_list=None,
                                limit=None, offset=None):
        if device_id_list is not None and patient_id_list is not None:
            raise ValueError("Can only request labels by device or patient, not both")

        results = []

        if patient_id_list is not None:
            # Fetch device time ranges associated with a patient and adjust the time range based on provided boundaries.
            for patient_id in patient_id_list:
                device_time_ranges = self.get_device_time_ranges_by_patient(patient_id, end_time_n, start_time_n)

                for device_id, device_start_time, device_end_time in device_time_ranges:
                    final_start_time = max(start_time_n, device_start_time) if start_time_n else device_start_time
                    final_end_time = min(end_time_n, device_end_time) if end_time_n else device_end_time

                    # Recursively fetch labels and accumulate the results.
                    results.extend(
                        self.select_labels_with_info(label_set_id_list=label_set_id_list, device_id_list=[device_id],
                                                     start_time_n=final_start_time, end_time_n=final_end_time,
                                                     label_source_id_list=label_source_id_list,
                                                     measure_id_list=measure_id_list))

            # Sort the results by start_time_n primarily and then by end_time_n secondarily
            results.sort(key=lambda x: (x[7], x[8], x[0]))
            return results

        # Construct the query with additional joins for label_source, label_set, and device_patient.
        query = """
            SELECT
                label.id, label_set.name AS label_name, label_set_id, 
                label.device_id, label.measure_id, label_source.name AS label_source_name, 
                label_source_id, start_time_n, end_time_n, 
                device_patient.patient_id
            FROM label
            JOIN label_set ON label.label_set_id = label_set.id
            LEFT JOIN label_source ON label.label_source_id = label_source.id
            LEFT JOIN device_patient ON label.device_id = device_patient.device_id
                AND label.start_time_n >= device_patient.start_time
                AND label.end_time_n <= device_patient.end_time
        """
        params = []

        # Add conditions based on the provided criteria.
        conditions = ["1=1"]  # Placeholder for dynamic query construction

        if label_set_id_list:
            placeholders = ', '.join(['?'] * len(label_set_id_list))
            conditions.append(f"label.label_set_id IN ({placeholders})")
            params.extend(label_set_id_list)

        if device_id_list:
            placeholders = ', '.join(['?'] * len(device_id_list))
            conditions.append(f"label.device_id IN ({placeholders})")
            params.extend(device_id_list)

        if measure_id_list:
            placeholders = ', '.join(['?'] * len(measure_id_list))
            conditions.append(f"label.measure_id IN ({placeholders})")
            params.extend(measure_id_list)

        if label_source_id_list:
            placeholders = ', '.join(['?'] * len(label_source_id_list))
            conditions.append(f"label.label_source_id IN ({placeholders})")
            params.extend(label_source_id_list)

        if start_time_n:
            conditions.append("label.end_time_n >= ?")
            params.append(int(start_time_n))

        if end_time_n:
            conditions.append("label.start_time_n <= ?")
            params.append(int(end_time_n))

        # Combine conditions into the query.
        query += " WHERE " + " AND ".join(conditions)

        # Add sorting.
        query += " ORDER BY start_time_n ASC, end_time_n ASC, label.id ASC"

        # Handle limit and offset.
        if limit is not None:
            query += f" LIMIT {limit}"
            if offset is not None:
                query += f" OFFSET {offset}"

        # Execute the query and return the results.
        with self.connection(begin=False) as (conn, cursor):
            cursor.execute(query, params)
            fetched_results = cursor.fetchall()

        return fetched_results

    def insert_label_source(self, name, description=None):
        # First, check if the label_source with the given name already exists
        select_query = "SELECT id FROM label_source WHERE name = ?"
        with self.connection(begin=False) as (conn, cursor):
            cursor.execute(select_query, (name,))
            result = cursor.fetchone()
            if result:
                # A label_source with the given name already exists, return its id
                return result[0]

        # If not found, insert the new label_source
        insert_query = "INSERT INTO label_source (name, description) VALUES (?, ?)"
        with self.connection(begin=True) as (conn, cursor):
            cursor.execute(insert_query, (name, description))
            conn.commit()
            return cursor.lastrowid

    def select_label_source_id_by_name(self, name):
        query = "SELECT id FROM label_source WHERE name = ? LIMIT 1"
        with self.connection() as (conn, cursor):
            cursor.execute(query, (name,))
            result = cursor.fetchone()
            return result[0] if result else None

    def select_label_source_info_by_id(self, label_source_id):
        query = "SELECT id, name, description FROM label_source WHERE id = ? LIMIT 1"
        with self.connection() as (conn, cursor):
            cursor.execute(query, (int(label_source_id),))
            result = cursor.fetchone()
            return {'id': result[0], 'name': result[1], 'description': result[2]} if result else None
        pass

    def select_all_label_sources(self, limit=None, offset=None):
        # Retrieve all label sources from the database.
        return self._select_all_ordered(
            "SELECT id, name FROM label_source ORDER BY id ASC", limit, offset)

    def get_measure_id_with_most_rows(self, tag: str) -> Optional[int]:
        # Query to get all matching measure.ids
        measure_ids_query = """
        SELECT id FROM measure WHERE tag = ?
        """
        measure_ids = []

        with self.connection(begin=False) as (conn, cursor):
            cursor.execute(measure_ids_query, (tag,))
            measure_ids = [row[0] for row in cursor.fetchall()]

        # Query to find the measure.id with the most rows in block_index
        most_rows_query = """
        SELECT measure_id, COUNT(*) as row_count
        FROM block_index
        WHERE measure_id IN ({})
        GROUP BY measure_id
        ORDER BY row_count DESC
        LIMIT 1
        """.format(','.join(['?'] * len(measure_ids)))

        if not measure_ids:
            return None

        if len(measure_ids) == 1:
            return measure_ids[0]
        with self.connection(begin=False) as (conn, cursor):
            cursor.execute(most_rows_query, measure_ids)
            result = cursor.fetchone()
            return result[0] if result else None

    def get_tag_to_measure_ids_dict(self, approx: bool = True) -> Dict[str, List[int]]:
        # Retrieve all measures and construct id-to-tag mapping
        measure_query = """
        SELECT id, tag FROM measure
        """
        id_to_tag = {}

        with self.connection(begin=False) as (conn, cursor):
            cursor.execute(measure_query)
            for row in cursor.fetchall():
                measure_id, tag = row
                if tag not in id_to_tag:
                    id_to_tag[tag] = []
                id_to_tag[tag].append(int(measure_id))

        # Get count of rows for each measure ID from block_index
        if approx:
            block_index_query = """
            SELECT measure_id, COUNT(*) as approx_count
            FROM block_index
            WHERE id <= 100000
            GROUP BY measure_id;
            """
        else:
            block_index_query = """
            SELECT measure_id, COUNT(*) as row_count
            FROM block_index
            GROUP BY measure_id
            """

        measure_id_to_count: Dict[int, int] = {}
        with self.connection(begin=False) as (conn, cursor):
            cursor.execute(block_index_query)
            for row in cursor.fetchall():
                measure_id, count = row
                measure_id_to_count[int(measure_id)] = count

        # Construct the final dictionary
        tag_to_sorted_measure_ids = {}
        for tag, measure_ids in id_to_tag.items():
            # Sort the measure IDs by count, descending order
            sorted_measure_ids = sorted(
                measure_ids,
                key=lambda x: measure_id_to_count.get(x, 0),
                reverse=True
            )
            tag_to_sorted_measure_ids[tag] = sorted_measure_ids

        return tag_to_sorted_measure_ids

    def insert_source(self, name: str, description: Optional[str] = None) -> int:
        query = "INSERT INTO source (name, description) VALUES (?, ?)"
        with self.connection() as (conn, cursor):
            cursor.execute(query, (name, description))
            conn.commit()
            return cursor.lastrowid

    def select_source(self, source_id: Optional[int] = None, name: Optional[str] = None) -> Optional[Tuple[int, str, Optional[str]]]:
        query = "SELECT id, name, description FROM source WHERE "
        params = []
        if source_id:
            query += "id = ?"
            params.append(int(source_id))
        elif name:
            query += "name = ?"
            params.append(name)
        else:
            raise ValueError("Either source_id or name must be provided")

        with self.connection() as (conn, cursor):
            cursor.execute(query, params)
            return cursor.fetchone()

    def insert_institution(self, name: str) -> int:
        query = "INSERT INTO institution (name) VALUES (?)"
        with self.connection() as (conn, cursor):
            cursor.execute(query, (name,))
            conn.commit()
            return cursor.lastrowid

    def select_institution(self, institution_id: Optional[int] = None, name: Optional[str] = None) -> Optional[Tuple[int, str]]:
        query = "SELECT id, name FROM institution WHERE "
        params = []
        if institution_id:
            query += "id = ?"
            params.append(int(institution_id))
        elif name:
            query += "name = ?"
            params.append(name)
        else:
            raise ValueError("Either institution_id or name must be provided")

        with self.connection() as (conn, cursor):
            cursor.execute(query, params)
            return cursor.fetchone()

    def insert_unit(self, institution_id: int, name: str, unit_type: str) -> int:
        query = "INSERT INTO unit (institution_id, name, type) VALUES (?, ?, ?)"
        with self.connection() as (conn, cursor):
            cursor.execute(query, (int(institution_id), name, unit_type))
            conn.commit()
            return cursor.lastrowid

    def select_unit(self, unit_id: Optional[int] = None, name: Optional[str] = None) -> Optional[Tuple[int, int, str, str]]:
        query = "SELECT id, institution_id, name, type FROM unit WHERE "
        params = []
        if unit_id:
            query += "id = ?"
            params.append(int(unit_id))
        elif name:
            query += "name = ?"
            params.append(name)
        else:
            raise ValueError("Either unit_id or name must be provided")

        with self.connection() as (conn, cursor):
            cursor.execute(query, params)
            return cursor.fetchone()

    def insert_bed(self, unit_id: int, name: str, bed_id: Optional[int] = None) -> int:
        if bed_id is not None:
            query = "INSERT INTO bed (id, unit_id, name) VALUES (?, ?, ?)"
            params = (int(bed_id), int(unit_id), name)
        else:
            query = "INSERT INTO bed (unit_id, name) VALUES (?, ?)"
            params = (int(unit_id), name)

        with self.connection() as (conn, cursor):
            cursor.execute(query, params)
            conn.commit()
            return bed_id if bed_id is not None else cursor.lastrowid

    def select_bed(self, bed_id: Optional[int] = None, name: Optional[str] = None) -> Optional[Tuple[int, int, str]]:
        query = "SELECT id, unit_id, name FROM bed WHERE "
        params = []
        if bed_id:
            query += "id = ?"
            params.append(int(bed_id))
        elif name:
            query += "name = ?"
            params.append(name)
        else:
            raise ValueError("Either bed_id or name must be provided")

        with self.connection() as (conn, cursor):
            cursor.execute(query, params)
            return cursor.fetchone()

    def find_unreferenced_tsc_files(self):
        with self.connection() as (conn, cursor):
            cursor.execute("SELECT t1.* FROM file_index t1 LEFT JOIN (SELECT DISTINCT file_id FROM block_index) t2 "
                           "ON t1.id = t2.file_id WHERE t2.file_id IS NULL")
            return cursor.fetchall()


    def delete_files_by_ids(self, file_ids_to_delete: List[int | tuple]):
        if len(file_ids_to_delete) == 0:
            return
        if isinstance(file_ids_to_delete[0], int):
            file_ids_to_delete = [(file_id,) for file_id in file_ids_to_delete]

        with self.connection(begin=False) as (conn, cursor):
            # if you put too many rows in the delete statement mariadb will fail. So we split it up
            for i in range(math.ceil(len(file_ids_to_delete) / 100_000)):
                # delete old tsc files
                cursor.executemany("DELETE FROM file_index WHERE id = ?;", file_ids_to_delete[i * 100_000:(i + 1) * 100_000])
