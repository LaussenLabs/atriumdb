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
"""
Per-measure string value dictionary for AtriumDB.

String values are stored by encoding each unique string as an ``int64`` code and
reusing the ordinary int64 block write/read path -- there are no C changes and no
block-format changes. This module owns the mapping between strings and codes.

**File format.** One append-only JSON Lines file per measure at
``<meta_dir>/string_dict/measure_<measure_id>.jsonl``. The 0-based line index IS
the code, and each line is a single JSON-encoded string. JSON encoding makes
embedded newlines, quotes, commas, arbitrary unicode and the empty string all
safe and unambiguous (a blank line is ``""`` -> JSON ``""`` -> a non-empty line).

**Stability.** The file is append-only: existing codes are immutable so historical
blocks never need rewriting. New strings are appended, taking the next code.

**Concurrency / single-writer expectation.** Appends happen under a cross-process
file lock (``filelock`` if installed, else ``fcntl`` on POSIX), and the file is
re-read inside the lock before appending so two writers can never assign the same
code to different strings. This mirrors the single-writer-per-(measure, device)
expectation already documented on ``AtriumSDK.write_data``'s block-merge path:
the lock protects the dictionary specifically, but concurrent block writes for the
same measure/device remain the caller's responsibility.

Background: ``docs/design/aperiodic-and-text-support.md``.
"""
from __future__ import annotations

import json
from pathlib import Path, PurePath
from typing import Sequence, Union

import numpy as np

# Reserved "unknown" sentinel code for string / int64 code channels. Real
# dictionary codes are always >= 0 (line index in
# the append-only file), so a negative sentinel can never collide with a genuine
# string. NaN plays the same role for float channels. Chosen over "reserve code
# 0" because existing committed dictionaries already assigned code 0 to a real
# string -- a negative sentinel is safe to add without rewriting any data.
UNKNOWN_STRING_CODE = -1

# The value a reserved unknown code decodes to, by default. Kept distinct from
# any genuine string (it is not in the vocabulary). Note that a single sentinel
# conflates "unknown / censored" with a genuine missing reading.
UNKNOWN_STRING_VALUE = "<unknown>"

# The append is guarded by a cross-process advisory lock. The implementation is shared
# with the block-merge lock in ``write_data`` -- see :mod:`atriumdb.file_lock` for the
# filelock/fcntl selection and its caveats. Reads never need the lock because the file is
# append-only.
from atriumdb.file_lock import make_file_lock as _make_lock


class MeasureStringDictionary:
    """Append-only ``{str <-> int64 code}`` dictionary for a single measure.

    Instances are cheap in-memory views loaded via :meth:`load`; the on-disk file
    is the source of truth. ``encode`` extends both the in-memory map and the file
    (under a lock); ``decode`` is a pure lookup.
    """

    def __init__(self, path: Union[str, PurePath], strings: Sequence[str] = ()):
        self._path = Path(path)
        # index -> string (code == index); str -> code for fast encode lookups.
        self._strings: list = list(strings)
        self._code_of: dict = {s: i for i, s in enumerate(self._strings)}
        # Bookkeeping for the most recent :meth:`encode` so a caller whose write
        # fails downstream can undo the appends (see :meth:`rollback_appends`).
        self.last_appended: list = []
        self.last_length_before_append: int = len(self._strings)

    # ------------------------------------------------------------------ #
    # Location / existence
    # ------------------------------------------------------------------ #
    @classmethod
    def path_for(cls, meta_dir: Union[str, PurePath], measure_id: int) -> Path:
        """Return the dictionary file path for a measure (not necessarily existing)."""
        return Path(meta_dir) / "string_dict" / f"measure_{int(measure_id)}.jsonl"

    @classmethod
    def exists(cls, meta_dir: Union[str, PurePath], measure_id: int) -> bool:
        """True if a NON-EMPTY string dictionary file exists for this measure.

        The presence of this file is the fallback signal that a measure is
        string-typed, used for datasets written before the ``value_type`` column
        existed and never backfilled. Keep this the one call site that tests for
        the file, so the rule lives in a single place.

        A zero-byte file is deliberately NOT an establishment signal: it carries no
        vocabulary, so treating it as "this measure is string-typed" would let a
        crashed or rolled-back write permanently lock a numeric measure.
        """
        path = cls.path_for(meta_dir, measure_id)
        try:
            return path.is_file() and path.stat().st_size > 0
        except OSError:  # pragma: no cover - racing unlink
            return False

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #
    @classmethod
    def load(cls, meta_dir: Union[str, PurePath], measure_id: int) -> "MeasureStringDictionary":
        """Load the dictionary from disk (empty if the file does not yet exist)."""
        path = cls.path_for(meta_dir, measure_id)
        inst = cls(path)
        inst._reload_from_file()
        return inst

    def _reload_from_file(self) -> None:
        """(Re)read the whole file into memory. Called on load and again inside the
        append lock so concurrently-appended codes are picked up before encoding."""
        strings: list = []
        if self._path.is_file():
            with open(self._path, "r", encoding="utf-8") as f:
                for line_number, line in enumerate(f, start=1):
                    # Trailing newline is the record separator; blank final line
                    # (no content) is skipped, but ``""`` decodes to empty string.
                    line = line.rstrip("\n")
                    if line == "":
                        continue
                    try:
                        strings.append(json.loads(line))
                    except json.JSONDecodeError as decode_error:
                        # A crash mid-append leaves a partial final line. The bare
                        # JSONDecodeError names neither the measure, the file, nor a
                        # remedy, and every code from here on is unreadable.
                        raise ValueError(
                            f"String dictionary '{self._path}' is corrupt at line {line_number} "
                            f"(code {len(strings)}): {decode_error}. The file is append-only JSON "
                            f"Lines, so this is normally a truncated final line from an interrupted "
                            f"write, or a partial restore. Restore meta/string_dict/ from the backup "
                            f"taken with this dataset's .tsc files and metadata database, and do not "
                            f"write to this measure until you do -- writing now would re-issue codes "
                            f"that existing blocks already use."
                        ) from decode_error
        self._strings = strings
        self._code_of = {s: i for i, s in enumerate(strings)}

    # ------------------------------------------------------------------ #
    # Encode / decode
    # ------------------------------------------------------------------ #
    @staticmethod
    def _coerce(value) -> str:
        """Normalize a single value to a plain ``str`` or raise ``TypeError``.

        Accepts ``str`` (incl. ``numpy.str_``) and ``bytes`` (utf-8 decoded, so
        ``dtype.kind == 'S'`` arrays work). Anything else -- ints, floats, None,
        nested objects in an object array -- is a programming error for a string
        measure and raises with a clear message.
        """
        if isinstance(value, str):
            return str(value)  # collapse numpy.str_ to a builtin str
        if isinstance(value, (bytes, np.bytes_)):
            return value.decode("utf-8")
        raise TypeError(
            f"String measure values must be str (or bytes); got {type(value).__name__}: "
            f"{value!r}. Numeric measures use the ordinary write path.")

    def encode(self, values: Sequence[str]) -> np.ndarray:
        """Map string values to ``int64`` codes, appending genuinely new strings.

        New strings are appended to the file under a lock (the file is re-read
        inside the lock first, so a concurrent writer's appends are honored and no
        two writers assign the same code to different strings). Returns an
        ``int64`` ndarray the same length/order as ``values``.
        """
        # Normalize input (accepts list[str], numpy 'U'/'S'/'O' arrays, etc.).
        coerced = [self._coerce(v) for v in np.asarray(values, dtype=object).tolist()] \
            if isinstance(values, np.ndarray) else [self._coerce(v) for v in values]

        # Which distinct strings are not yet known in memory (order-preserving).
        unknown: list = []
        seen: set = set()
        for s in coerced:
            if s not in self._code_of and s not in seen:
                seen.add(s)
                unknown.append(s)

        # Reset the rollback bookkeeping for this call.
        self.last_appended = []
        self.last_length_before_append = len(self._strings)

        if unknown:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            lock = _make_lock(str(self._path) + ".lock")
            with lock:
                # Re-sync with the file: another process may have appended codes
                # (or the very strings we were about to add) since we loaded.
                self._reload_from_file()
                self.last_length_before_append = len(self._strings)
                to_append = [s for s in unknown if s not in self._code_of]
                if to_append:
                    with open(self._path, "a", encoding="utf-8") as f:
                        for s in to_append:
                            f.write(json.dumps(s, ensure_ascii=False) + "\n")
                            self._code_of[s] = len(self._strings)
                            self._strings.append(s)
                            self.last_appended.append(s)

        return np.fromiter((self._code_of[s] for s in coerced), dtype=np.int64, count=len(coerced))

    def rollback_appends(self) -> bool:
        """Undo the appends made by the most recent :meth:`encode` call.

        A string write encodes its values (appending genuinely new strings) *before*
        the block bytes and SQL rows are committed, because the codes have to be
        baked into the encoded block. If the write then fails, those appended
        strings describe data the dataset does not contain -- they retain free text
        (potentially PHI) from a rejected batch and, worse, make the mere existence
        of the dictionary file establish the measure as string-typed forever.
        This makes the append transactional with the write.

        The truncation is done under the same lock as the append and only when the
        file still ends with exactly the lines this instance wrote -- if another
        process appended in the meantime the codes are no longer ours to reclaim,
        so nothing is removed and ``False`` is returned. When the rollback empties
        the dictionary the file is unlinked, so no zero-byte husk is left behind.

        Returns True when the appends were undone (or there was nothing to undo).
        """
        appended = list(self.last_appended)
        if not appended:
            return True

        previous_length = self.last_length_before_append
        lock = _make_lock(str(self._path) + ".lock")
        with lock:
            self._reload_from_file()
            if self._strings[previous_length:] != appended:
                # A concurrent writer appended after us (or the file changed under
                # us). Truncating would destroy codes we do not own.
                self.last_appended = []
                return False

            if previous_length == 0:
                try:
                    self._path.unlink()
                except FileNotFoundError:  # pragma: no cover - already gone
                    pass
            else:
                kept = self._strings[:previous_length]
                tmp_path = self._path.with_suffix(self._path.suffix + ".rollback")
                with open(tmp_path, "w", encoding="utf-8") as f:
                    for s in kept:
                        f.write(json.dumps(s, ensure_ascii=False) + "\n")
                tmp_path.replace(self._path)

            self._reload_from_file()

        self.last_appended = []
        self.last_length_before_append = len(self._strings)
        return True

    def _out_of_range_error(self, code: int) -> ValueError:
        """The error for a code this dictionary cannot resolve.

        Three genuinely different situations land here, and at 3am the difference
        is what matters, so the message names which one it looks like:
        the reserved unknown sentinel decoded through the strict accessor, an
        emptied/rolled-back dictionary, and a dictionary that is merely older than
        the blocks (the restore-mismatch case ``_check_string_dictionary_not_lost``
        guards on the write side)."""
        n_vocab = len(self._strings)
        if code == UNKNOWN_STRING_CODE:
            cause = (f"{UNKNOWN_STRING_CODE} is the reserved 'unknown' sentinel that "
                     f"rasterized windows use for gap / censored cells, so these codes "
                     f"came from a window rather than from stored data: decode them with "
                     f"decode_with_unknown() (Window.decode_string_signal does this for "
                     f"you) instead of decode().")
        elif n_vocab == 0:
            cause = ("The dictionary is empty or missing, so NO code can be resolved. "
                     "Restore meta/string_dict/ from the backup taken with this dataset's "
                     ".tsc files and metadata database.")
        else:
            cause = (f"Codes 0..{n_vocab - 1} are known, so the blocks reference strings "
                     f"this dictionary file does not have: it is older than the data "
                     f"(a partial restore, or a truncated append). Restore "
                     f"meta/string_dict/ from the backup taken with this dataset's .tsc "
                     f"files and metadata database.")
        return ValueError(
            f"String code {code} is out of range for a dictionary of size {n_vocab} "
            f"(measure dictionary '{self._path}'). This indicates a dictionary/data "
            f"mismatch. {cause}")

    def _decode_codes(self, codes, unknown_code=None, unknown_value=None) -> np.ndarray:
        """Shared decode loop for :meth:`decode` / :meth:`decode_with_unknown`.

        ``unknown_code=None`` means "no sentinel is tolerated" (strict decode of
        stored data); otherwise that one code maps to ``unknown_value`` and every
        other out-of-range code still raises."""
        codes_arr = np.asarray(codes)
        n_vocab = len(self._strings)
        out = np.empty(codes_arr.shape, dtype=object)
        flat = codes_arr.reshape(-1)
        out_flat = out.reshape(-1)
        for i, c in enumerate(flat):
            code = int(c)
            if unknown_code is not None and code == unknown_code:
                out_flat[i] = unknown_value
                continue
            if code < 0 or code >= n_vocab:
                raise self._out_of_range_error(code)
            out_flat[i] = self._strings[code]
        return out

    def decode(self, codes: np.ndarray) -> np.ndarray:
        """Map ``int64`` codes back to strings, returning an object ndarray.

        Strict: this is the accessor for codes read straight out of stored blocks,
        where every code must resolve. A code outside ``[0, len(self))`` raises
        ``ValueError`` -- it indicates a dictionary/data mismatch, which is never
        expected within one dataset. Rasterized WINDOW codes may legitimately
        carry the unknown sentinel; decode those with :meth:`decode_with_unknown`.
        """
        return self._decode_codes(codes)

    def decode_with_unknown(self, codes: np.ndarray,
                            unknown_code: int = UNKNOWN_STRING_CODE,
                            unknown_value=UNKNOWN_STRING_VALUE) -> np.ndarray:
        """Decode ``int64`` codes to strings, mapping the reserved unknown
        sentinel code (default ``-1``) to ``unknown_value`` instead of raising.

        This is the window decode accessor: a rasterized string window
        carries genuine dictionary codes (>= 0) plus the reserved unknown
        sentinel for gap / censored cells. Genuine codes decode normally; the
        sentinel decodes to ``unknown_value`` (``"<unknown>"`` by default, or
        pass ``None`` to get ``None`` for those cells). Any other out-of-range
        code still raises, as it indicates a dictionary/data mismatch."""
        return self._decode_codes(codes, unknown_code=unknown_code,
                                  unknown_value=unknown_value)

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    def vocabulary(self) -> list:
        """Return the full list of strings in code order (code == index).

        This is a read-only snapshot of every value ever written to the measure,
        suitable for cheap event-type enumeration with no data scan. The list is
        a copy, so callers may not mutate the dictionary."""
        return list(self._strings)

    def code_for(self, value: str):
        """Return the int64 code for ``value`` if it is already in the vocabulary,
        else ``None``. Unlike :meth:`encode`, this NEVER appends -- it is a pure
        lookup used by query paths (e.g. event pairing) that must reject values
        which were never written rather than silently minting new codes."""
        return self._code_of.get(self._coerce(value))

    def __len__(self) -> int:
        """Vocabulary size (number of distinct strings / next code to assign)."""
        return len(self._strings)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"MeasureStringDictionary(path={self._path!s}, size={len(self._strings)})"


def decode_window_codes(sdk, measure_id: int, codes, unknown_value=None) -> np.ndarray:
    """Decode one string measure's rasterized WINDOW codes back to strings.

    The one implementation behind :meth:`Window.decode_string_signal` and
    :meth:`DatasetIterator.decode_string_codes`, which are the same three steps
    (locate the measure's dictionary under the dataset's ``meta/``, load it,
    decode tolerating the unknown sentinel) and must not drift apart.

    ``unknown_value=None`` selects the default :data:`UNKNOWN_STRING_VALUE`
    (``"<unknown>"``). To get Python ``None`` for unknown cells instead, decode
    via :meth:`MeasureStringDictionary.decode_with_unknown` directly.
    """
    if unknown_value is None:
        unknown_value = UNKNOWN_STRING_VALUE
    string_dict = MeasureStringDictionary.load(sdk._meta_dir, int(measure_id))
    return string_dict.decode_with_unknown(
        np.asarray(codes).astype(np.int64), unknown_value=unknown_value)
