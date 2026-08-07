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
"""Cross-process advisory file locks shared by AtriumDB's write paths.

Two write paths need to serialize concurrent processes against each other:

* the per-measure string dictionary append (:mod:`atriumdb.string_dictionary`), and
* the small-write block merge (``AtriumSDK.write_data``), which is a
  read-modify-write of an existing block.

Both are guarded the same way: an exclusive advisory lock on a ``.lock`` sidecar file
under the dataset's ``meta/`` directory. ``filelock`` is used when installed (robust and
cross-platform); otherwise a small ``fcntl.flock`` implementation is used on POSIX.

Both back-ends lock the *open file description*, so two separate lock objects conflict
with each other even inside a single process -- which is what makes these locks work for
threads as well as processes. Each call therefore returns a FRESH lock object; never
cache and reuse one across threads. Neither is reentrant: a lock must not be re-acquired
while already held on the same path by the same thread.

The OS releases the lock when the holding process dies, so a crashed writer cannot wedge
a dataset. As with any advisory lock this only coordinates processes that reach the same
file path -- writers on different mounts, or on a filesystem without working ``flock``
(some NFS configurations), are not protected.
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

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
                "AtriumDB write locking requires either the 'filelock' package "
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


def make_file_lock(lock_path: Union[str, Path]):
    """Return a fresh exclusive-lock context manager for ``lock_path``.

    The parent directory is created if needed, so callers do not have to."""
    path = Path(lock_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:  # pragma: no cover - defensive; the lock open will report it
        pass
    if _HAS_FILELOCK:
        return _FileLock(str(path))
    return _FcntlLock(path)
