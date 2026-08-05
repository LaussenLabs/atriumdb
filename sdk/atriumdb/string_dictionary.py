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
Per-measure string value dictionary for AtriumDB (Phase 1 string storage).

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
"""
from __future__ import annotations

import json
from pathlib import Path, PurePath
from typing import Sequence, Union

import numpy as np

# Reserved "unknown" sentinel code for string / int64 code channels (Phase 3,
# design section 21.2 #2). Real dictionary codes are always >= 0 (line index in
# the append-only file), so a negative sentinel can never collide with a genuine
# string. NaN plays the same role for float channels. Chosen over "reserve code
# 0" because existing committed dictionaries already assigned code 0 to a real
# string -- a negative sentinel is safe to add without rewriting any data.
UNKNOWN_STRING_CODE = -1

# The value a reserved unknown code decodes to, by default. Kept distinct from
# any genuine string (it is not in the vocabulary). A sentinel conflates
# "unknown / censored" with a genuine missing reading -- see design 21.2 #2(a).
UNKNOWN_STRING_VALUE = "<unknown>"

# Prefer filelock (a robust cross-platform advisory lock). Fall back to a small
# fcntl-based lock on POSIX so the dependency is optional. Both guard only the
# append; reads never need the lock because the file is append-only.
try:
    from filelock import FileLock as _FileLock

    _HAS_FILELOCK = True
except ImportError:  # pragma: no cover - exercised only when filelock is absent
    _HAS_FILELOCK = False
    try:
        import fcntl as _fcntl
    except ImportError:  # pragma: no cover - non-POSIX without filelock
        _fcntl = None


class _FcntlLock:
    """Minimal exclusive file lock using fcntl, used only when filelock is not
    installed. Creates/opens a ``.lock`` sidecar file and holds an exclusive
    ``flock`` for the duration of the ``with`` block."""

    def __init__(self, lock_path):
        self._lock_path = str(lock_path)
        self._fh = None

    def __enter__(self):
        if _fcntl is None:  # pragma: no cover
            raise RuntimeError(
                "String dictionary appends require either the 'filelock' package "
                "or POSIX fcntl; neither is available in this environment.")
        self._fh = open(self._lock_path, "w")
        _fcntl.flock(self._fh.fileno(), _fcntl.LOCK_EX)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            _fcntl.flock(self._fh.fileno(), _fcntl.LOCK_UN)
        finally:
            self._fh.close()
            self._fh = None


def _make_lock(lock_path):
    if _HAS_FILELOCK:
        return _FileLock(str(lock_path))
    return _FcntlLock(lock_path)


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

    # ------------------------------------------------------------------ #
    # Location / existence
    # ------------------------------------------------------------------ #
    @classmethod
    def path_for(cls, meta_dir: Union[str, PurePath], measure_id: int) -> Path:
        """Return the dictionary file path for a measure (not necessarily existing)."""
        return Path(meta_dir) / "string_dict" / f"measure_{int(measure_id)}.jsonl"

    @classmethod
    def exists(cls, meta_dir: Union[str, PurePath], measure_id: int) -> bool:
        """True if a string dictionary file exists for this measure.

        Phase 1 uses the presence of this file as the single signal that a measure
        is string-typed. Keep this the one detection call site so Phase 2 can swap
        it for a ``signal_kind`` schema column without touching callers.
        """
        return cls.path_for(meta_dir, measure_id).is_file()

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
                for line in f:
                    # Trailing newline is the record separator; blank final line
                    # (no content) is skipped, but ``""`` decodes to empty string.
                    line = line.rstrip("\n")
                    if line == "":
                        continue
                    strings.append(json.loads(line))
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

        if unknown:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            lock = _make_lock(str(self._path) + ".lock")
            with lock:
                # Re-sync with the file: another process may have appended codes
                # (or the very strings we were about to add) since we loaded.
                self._reload_from_file()
                to_append = [s for s in unknown if s not in self._code_of]
                if to_append:
                    with open(self._path, "a", encoding="utf-8") as f:
                        for s in to_append:
                            f.write(json.dumps(s, ensure_ascii=False) + "\n")
                            self._code_of[s] = len(self._strings)
                            self._strings.append(s)

        return np.fromiter((self._code_of[s] for s in coerced), dtype=np.int64, count=len(coerced))

    def decode(self, codes: np.ndarray) -> np.ndarray:
        """Map ``int64`` codes back to strings, returning an object ndarray.

        A code outside ``[0, len(self))`` raises ``ValueError`` -- it indicates a
        dictionary/data mismatch, which is never expected within one dataset.
        """
        codes_arr = np.asarray(codes)
        n_vocab = len(self._strings)
        out = np.empty(codes_arr.shape, dtype=object)
        flat = codes_arr.reshape(-1)
        out_flat = out.reshape(-1)
        for i, c in enumerate(flat):
            code = int(c)
            if code < 0 or code >= n_vocab:
                raise ValueError(
                    f"String code {code} is out of range for a dictionary of size "
                    f"{n_vocab} (measure dictionary '{self._path}'). This indicates "
                    f"a dictionary/data mismatch.")
            out_flat[i] = self._strings[code]
        return out

    def decode_with_unknown(self, codes: np.ndarray,
                            unknown_code: int = UNKNOWN_STRING_CODE,
                            unknown_value=UNKNOWN_STRING_VALUE) -> np.ndarray:
        """Decode ``int64`` codes to strings, mapping the reserved unknown
        sentinel code (default ``-1``) to ``unknown_value`` instead of raising.

        This is the Phase 3 window decode accessor: a rasterized string window
        carries genuine dictionary codes (>= 0) plus the reserved unknown
        sentinel for gap / censored cells. Genuine codes decode normally; the
        sentinel decodes to ``unknown_value`` (``"<unknown>"`` by default, or
        pass ``None`` to get ``None`` for those cells). Any other out-of-range
        code still raises, as it indicates a dictionary/data mismatch."""
        codes_arr = np.asarray(codes)
        n_vocab = len(self._strings)
        out = np.empty(codes_arr.shape, dtype=object)
        flat = codes_arr.reshape(-1)
        out_flat = out.reshape(-1)
        for i, c in enumerate(flat):
            code = int(c)
            if code == unknown_code:
                out_flat[i] = unknown_value
                continue
            if code < 0 or code >= n_vocab:
                raise ValueError(
                    f"String code {code} is out of range for a dictionary of size "
                    f"{n_vocab} (measure dictionary '{self._path}'). This indicates "
                    f"a dictionary/data mismatch.")
            out_flat[i] = self._strings[code]
        return out

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    def vocabulary(self) -> list:
        """Return the full list of strings in code order (code == index).

        This is a read-only snapshot of every value ever written to the measure,
        suitable for cheap event-type enumeration (design §22.1.1) with no data
        scan. The list is a copy, so callers may not mutate the dictionary."""
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
