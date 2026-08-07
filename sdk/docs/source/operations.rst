.. _operations:

Operations
==========

This page covers the questions you have to answer before running AtriumDB as a service:
what is on disk and what must be backed up, what happens when a feed replays a batch, whether
more than one process may write, and what a returned write actually guarantees.

.. _dataset_layout:

What a dataset is, on disk
--------------------------

A dataset created with
`AtriumSDK.create_dataset <contents.html#atriumdb.AtriumSDK.create_dataset>`_ is a directory
with this shape::

    <dataset_location>/
        tsc/                                  # compressed block files, the bulk of the data
            <measure_id>/<device_id>/*.tsc
        meta/
            index.db                          # SQLite metadata DB (sqlite datasets only)
            string_dict/
                measure_<measure_id>.jsonl        # per-measure string dictionary
                measure_<measure_id>.jsonl.lock   # advisory lock sidecar
            locks/
                measure_<measure_id>/device_<device_id>.lock

For a **MariaDB / MySQL** dataset there is no ``meta/index.db``; the metadata lives in the
external database instead, and ``tsc/`` on the filesystem and that database are two halves of
one dataset.

Backup surface
--------------

All of the following are required. A restore missing any of them is not a working dataset:

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Component
     - Why
   * - ``tsc/``
     - The block data. Without it, metadata opens fine and every read raises
       ``FileNotFoundError`` on a missing ``.tsc``.
   * - ``meta/index.db`` **or** the external MariaDB/MySQL database
     - Measures, devices, patients, the block index, the interval index, labels, encounters.
       Without it the dataset does not open at all:
       ``ValueError: No Dataset found at location ...``
   * - ``meta/string_dict/*.jsonl``
     - The ``int64`` code → string mapping for every string measure.

.. danger::

   **String dictionaries are unrecoverable from the block data.** If ``meta/string_dict/`` is
   lost or partially restored, waveforms and numeric measures still read *perfectly* while every
   alarm string, ventilator mode and free-text value becomes permanently undecodable::

       ValueError: String code 0 is out of range for a dictionary of size 0 ...

   These are small files that backup tools and rsync filters are exactly the kind of thing to
   skip. A dataset that has lost them looks healthy. Verify after every restore by reading a
   known string measure with
   `AtriumSDK.get_measure_string_vocabulary <contents.html#atriumdb.AtriumSDK.get_measure_string_vocabulary>`_.

**Capture the components consistently.** The block files and the metadata database reference each
other; a snapshot that catches ``tsc/`` after a write and the database before it will have blocks
the index does not know about (harmless) or index rows pointing at files that do not exist
(read errors). Quiesce writers, or use a filesystem/LVM snapshot that covers ``tsc/`` and
``meta/`` atomically. For MariaDB-backed datasets, take the database dump and the filesystem
snapshot at the same quiesced point — they cannot be captured atomically by different tools
otherwise.

``meta/locks/`` holds zero-byte advisory lock sidecars only. It carries no data and does not need
to be restored, though copying it is harmless.

Durability
----------

AtriumDB writes block files and then commits the corresponding metadata rows. A write that has
returned has been written through the normal filesystem and database paths; the SDK does not
issue an explicit ``fsync`` of its own, so durability against a **machine** failure is whatever
your filesystem and (for MariaDB) your database server provide. Against a **process** failure —
the far more common case for an ingest service — the observed behaviour is clean: a new
``AtriumSDK`` object opened on the same location after a killed process reads all previously
returned writes and appends normally.

If you need a stronger guarantee than "the OS has it", enforce it at the storage layer
(journalled filesystem with barriers, MariaDB ``innodb_flush_log_at_trx_commit=1``).

.. note::

   Nothing in the API surfaces a "flush to stable storage" call.
   `AtriumSDK.write_buffer <contents.html#atriumdb.AtriumSDK.write_buffer>`_'s ``flush_all()``
   flushes the *in-memory buffer* into the ordinary write path; it is not an fsync.

.. _idempotency_and_replay:

Idempotency and replay
----------------------

A live feed that restarts will re-send a batch. What happens then depends on two things, and
**both of them matter**:

**1. The** ``merge_blocks`` **parameter** (default ``True`` on ``write_segment`` /
``write_segments`` / ``write_time_value_pairs`` / ``write_data``).

**2. Whether the batch is smaller than one optimal block** — ``sdk.block.block_size``, 131072
values by default. Merging is only attempted for writes below that size.

Taking a 5-value batch written twice:

.. list-table::
   :header-rows: 1
   :widths: 30 34 36

   * - Situation
     - ``get_data(..., allow_duplicates=True)``
     - ``get_data(..., allow_duplicates=False)``
   * - ``merge_blocks=True``, batch < ``block_size``
     - 5 values — the **replayed** values (last write wins)
     - 5 values — the replayed values
   * - ``merge_blocks=False``, batch < ``block_size``
     - 10 values — both copies interleaved
     - 5 values — the **most recently written** copy of each timestamp
   * - any ``merge_blocks``, batch ≥ ``block_size``
     - Both copies stored — the data is **duplicated**
     - Both copies stored on disk; the read collapses them to 5

.. warning::

   **The same code silently changes behaviour with batch size.** Testing replay safety with
   realistic 1-second live batches shows clean idempotency; the first backfill or catch-up job
   that writes a batch of 131072 values or more silently doubles that data. There is no error, no
   warning, and ``get_interval_array`` still looks correct (reads below are the default
   ``allow_duplicates=True``)::

       batch n=    100  read=    100  deduped
       batch n=   1000  read=   1000  deduped
       batch n=  20000  read=  20000  deduped
       batch n= 200000  read= 400000  DUPLICATED

   ``get_data(..., allow_duplicates=False)`` hides this on read, but the second copy is still on
   disk and still costs storage and decode time. Chunk your backfills.

.. _duplicate_survivors:

Which copy survives
####################

Duplicates are resolved on the **read** side, and the survivor follows the dataset's
``overwrite`` setting chosen at ``create_dataset`` time — so a read agrees with what a write
would have done had the two copies met in one block:

- ``"overwrite"`` (and the legacy default ``"ignore"``) — the **newest** stored copy wins;
- ``"protect"`` — the **earliest** stored copy wins;
- ``"error"`` — a merge whose write shares timestamps with the block it would merge into raises.

``get_data`` / ``get_string_data`` accept ``duplicate_keep="last"`` or ``"first"`` to override
that for a single call. It is only consulted when ``allow_duplicates=False``.

.. important::

   **This changed.** The older ``allow_duplicates=False`` kept the **first** stored copy of a
   duplicated timestamp regardless of the dataset's policy. It now keeps the **newest** by
   default. If you depended on first-wins, pass ``duplicate_keep="first"`` explicitly, or create
   the dataset with ``overwrite="protect"``.

Recommended pattern for a restarting feed
##########################################

1. Track a high-water mark (the last timestamp you successfully wrote) outside AtriumDB and
   resume from it, so replays are the exception rather than the norm.
2. Keep replayed writes **below** ``sdk.block.block_size``. Chunk backfills explicitly::

       for chunk_start in range(0, len(values), sdk.block.block_size // 2):
           chunk = slice(chunk_start, chunk_start + sdk.block.block_size // 2)
           sdk.write_segment(measure_id, device_id, values[chunk], times[chunk_start], ...)

3. Leave ``merge_blocks=True`` (the default) so a small replay overwrites rather than duplicates,
   and set the dataset's ``overwrite`` setting deliberately — ``"error"`` if you would rather a
   replay fail loudly than be resolved silently.
4. If duplicates do get in, ``get_data(..., allow_duplicates=False)`` de-duplicates on read; see
   :ref:`Which copy survives <duplicate_survivors>` for which one you get back.

Concurrency
-----------

**Multiple processes may write to one dataset**, and two write paths that are read-modify-write
are explicitly serialized with cross-process advisory locks (``atriumdb.file_lock``):

.. list-table::
   :header-rows: 1
   :widths: 34 30 36

   * - Protected operation
     - Lock file
     - Granularity
   * - The small-write **block merge**
       (``merge_blocks=True`` and batch < ``block_size``)
     - ``meta/locks/measure_<m>/device_<d>.lock``
     - Per (measure, device). Unrelated streams write fully in parallel; block-sized bulk writes
       never take the lock at all.
   * - The per-measure **string dictionary append**
     - ``meta/string_dict/measure_<m>.jsonl.lock``
     - Per measure.

Properties and limits of that mechanism:

- The locks are **advisory and filesystem-based**. They coordinate writers that share the same
  ``dataset_location``. Writers reaching the dataset by a different path, or on a filesystem
  without working ``flock`` (some NFS configurations), are **not** protected.
- They lock the open file description, so they serialize **threads as well as processes**.
- The OS releases them when the holding process dies, so a crashed writer cannot wedge a dataset.
- They are not reentrant, and a fresh lock object is taken per call — never cache one.
- In the API / storage-handler configurations there is no local ``dataset_location`` to key on,
  and the merge lock degrades to a no-op (nothing is merged locally there either).

Not covered by these locks:

- **The metadata database.** Concurrency there is your database's. SQLite serializes writers with
  its own file locking, which is adequate for a handful of ingest workers but will contend under
  many concurrent writers; MariaDB/MySQL is the right choice for a multi-writer production
  deployment.
- **Readers.** A reader can observe a block file that has been written but whose metadata is not
  yet committed (it simply does not see that data yet), or metadata for a block being merged.
  Reads are not torn — a ``.tsc`` file is written whole — but there is no read snapshot isolation
  across the file tree and the database.

.. note::

   Practical guidance: partition your writers by (measure, device) or by device where you can, so
   the merge lock is uncontended; use MariaDB rather than SQLite once you have more than a few
   concurrent writers; and do not run two writers against the same dataset over NFS.

Guarantees that already hold
-----------------------------

These behaviours are relied on in production and are worth knowing about explicitly:

**Out-of-order and late-arriving writes are re-sorted on read.** Segments written in the order
3, 1, 4, 2, 0 read back time-ordered and contiguous. Aperiodic points supplied unsorted **within a
single call** are sorted before storage, so ``get_data`` and ``get_string_data`` always return
ascending timestamps. A late string event backfilled before existing ones slots in correctly. You
do not have to buffer and sort an HL7 feed yourself.

**Small appends self-compact.** 600 consecutive 1-second appends of a 250 Hz waveform (150 000
values) leave **2** ``.tsc`` files and **one** interval, not 600 of each — that is
``merge_blocks=True`` doing its job. Un-buffered small writes are therefore safe for a live feed;
`write_buffer <contents.html#atriumdb.AtriumSDK.write_buffer>`_ is an optimisation that reduces
write amplification further, not a requirement. The buffer works for **strings and aperiodic
time-value pairs** as well as waveform segments, which its examples do not show.

**String dictionaries are append-only.** Existing codes are never rewritten, so historical blocks
stay valid as new vocabulary is appended, and 200 writes of a 5-word vocabulary leave a 5-line
dictionary file.

Things the API does not provide
--------------------------------

Called out so you do not spend time looking:

- **Selective deletion.** There is no "delete this patient" or "delete this time range" call.
  Removing data means rebuilding a dataset with ``transfer_data`` and a
  :ref:`DatasetDefinition <definition_file_format>` that excludes what you want gone, then
  swapping the directories.
- **Changing a measure's ``value_type`` after data is written.** Relabelling a measure that
  already holds string data as ``numeric`` (or vice versa) raises. ``signal_kind``, by contrast,
  *is* repairable — an ingest pipeline that auto-created a measure with the default ``waveform``
  shape can be corrected in place with
  `AtriumSDK.set_measure_kind <contents.html#atriumdb.AtriumSDK.set_measure_kind>`_, which
  rewrites no data::

      sdk.set_measure_kind(measure_id, signal_kind="state")

  This is the supported fix for the ``waveform`` + ``string`` measure that
  ``insert_measure`` produces when ``signal_kind`` is omitted. See
  :ref:`Choosing a signal_kind <choosing_signal_kind>`.
- **A dataset-wide time-coverage query.** There is no ``get_dataset_bounds()``; loop
  ``get_interval_array`` over the (measure, device) pairs you care about::

      import numpy as np
      bounds = []
      for mid in sdk.get_all_measures():
          for did in sdk.get_all_devices():
              arr = sdk.get_interval_array(mid, device_id=did)
              if arr is not None and len(arr):
                  bounds.append((int(arr[0][0]), int(arr[-1][1])))
      print(min(b[0] for b in bounds), max(b[1] for b in bounds))

- **A pandas integration.** See the DataFrame recipe in
  :ref:`Working with Datasets <to_pandas>`.
