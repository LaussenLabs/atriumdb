# Calculating Total Hours per Measure in AtriumDB

This document describes how to calculate the total wall-clock hours of data that exists in the database for each measure, aggregated across all patients and devices.

---

## Relevant Tables

### `block_index`

Each row in `block_index` represents one contiguous block of ingested signal data for a specific `(measure_id, device_id)` pair. The columns relevant to time coverage are:

| Column | Type | Meaning |
|---|---|---|
| `measure_id` | INTEGER | FK to `measures.id` |
| `device_id` | INTEGER | FK to `devices.id` |
| `start_time_n` | BIGINT | Inclusive block start, nanoseconds since Unix epoch |
| `end_time_n` | BIGINT | Exclusive block end, nanoseconds since Unix epoch |
| `num_values` | BIGINT | Number of data points stored in this block |

`end_time_n` is **exclusive** — the block covers `[start_time_n, end_time_n)`. The duration of one block in nanoseconds is therefore `end_time_n - start_time_n`.

There is a composite index on `(measure_id, device_id, start_time_n, end_time_n)` to support efficient time-range queries.

### `interval_index`

`interval_index` is a coarser-grained companion table. As blocks are written, the SDK merges adjacent or abutting blocks for the same `(measure_id, device_id)` pair into continuous coverage intervals and records them here. Columns:

| Column | Type | Meaning |
|---|---|---|
| `measure_id` | INTEGER | FK to `measures.id` |
| `device_id` | INTEGER | FK to `devices.id` |
| `start_time_n` | BIGINT | Continuous coverage start, nanoseconds |
| `end_time_n` | BIGINT | Continuous coverage end, nanoseconds |

Intervals within a `(measure_id, device_id)` pair are **non-overlapping** by construction — that is the SDK's invariant when inserting into `interval_index`. This makes them the natural aggregation source for total coverage.

---

## Why `interval_index` Is Preferred Over `block_index`

Within a single `(measure_id, device_id)`, individual blocks are also non-overlapping (each ingest produces its own block). However:

- `interval_index` already performs the merge — adjacent blocks that were ingested in separate calls are collapsed into a single interval, so fewer rows need to be summed.
- Using `interval_index` avoids any edge-case concern about abutting block boundaries being counted twice.
- The `select_intervals()` method in `sql_handler.py` is the canonical SDK API for retrieving coverage intervals, which confirms that `interval_index` is the intended table for coverage queries.

Both tables give the **same total** for a well-maintained database, but `interval_index` is cleaner.

---

## Caveat: Device-Level vs Patient-Level Coverage

Both tables store rows per `(measure_id, device_id)`, **not per patient**. A patient is linked to a device through the `device_patient` (or `device_encounter`) join table. If two devices record the same measure for the same patient concurrently, those intervals appear as separate rows in `interval_index` and will be summed independently, potentially double-counting real-world time.

For a **database-wide total** (the aggregate across all devices and patients, counting physical recording time), this double-counting is typically acceptable because each device-measure stream represents a distinct source. For a **patient-de-duplicated total** (how many unique patient-hours exist), a more complex overlap-merge is needed and is out of scope for this document.

---

## SQL Query: Total Hours per Measure (All Devices)

`interval_index` has one row per `(measure_id, device_id)` pair — not per patient and not per measure alone. To collapse all devices for the same measure into a single total, group exclusively by `measure_id`. The `SUM` then accumulates nanoseconds across every device that recorded that measure, which is the desired database-wide aggregate.

```sql
-- Total wall-clock hours covered in interval_index per measure
-- GROUP BY measure_id collapses all (measure_id, device_id) rows
-- into a single measure-level total.
SELECT
    m.measure_tag,
    m.measure_name,
    m.freq_nhz,
    m.units,
    COUNT(DISTINCT ii.device_id)                  AS num_devices,
    COUNT(*)                                      AS num_intervals,
    SUM(ii.end_time_n - ii.start_time_n)          AS total_ns,
    SUM(ii.end_time_n - ii.start_time_n)
        / 3600000000000.0                         AS total_hours
FROM interval_index ii
JOIN measures m ON m.id = ii.measure_id
GROUP BY ii.measure_id
ORDER BY total_hours DESC;
```

### Conversion factors (from nanoseconds)

| Target unit | Divisor |
|---|---|
| Seconds | `1_000_000_000` |
| Minutes | `60_000_000_000` |
| Hours | `3_600_000_000_000` |

---

## Alternative: Using `block_index`

If `interval_index` is not populated (it can be disabled at ingest time with `interval_index_mode="disable"`), fall back to `block_index`:

```sql
SELECT
    m.measure_tag,
    m.measure_name,
    COUNT(*)                                      AS num_blocks,
    SUM(bi.num_values)                            AS total_values,
    SUM(bi.end_time_n - bi.start_time_n)          AS total_ns,
    SUM(bi.end_time_n - bi.start_time_n)
        / 3600000000000.0                         AS total_hours
FROM block_index bi
JOIN measures m ON m.id = bi.measure_id
GROUP BY bi.measure_id
ORDER BY total_hours DESC;
```

`num_values` vs `total_hours`: `total_hours` reflects wall-clock span including any gaps within a block; `num_values` counts actual recorded samples. They diverge when data has internal gaps (which AtriumDB encodes as gap arrays inside the block file). Use `total_hours` for coverage and `num_values` for sample-count metrics.

---

## Python Helper Functions

Implemented in `sdk/atriumdb/dashboard/measure_queries.py`, following the same `query_xxx(sdk, ...)` convention used by `encounter_queries.py`.

### Primary: `query_measure_total_hours`

Reads `interval_index` (the SDK-maintained continuous-coverage table). This is the preferred path because `interval_index` already merges adjacent blocks into non-overlapping spans, so no deduplication is needed in Python.

```python
from atriumdb.dashboard.measure_queries import query_measure_total_hours

results = query_measure_total_hours(sdk)
for r in results:
    print(f"{r['measure_tag']}: {r['total_hours']:.1f} h across {r['num_devices']} devices")
```

### Fallback: `query_measure_total_hours_from_blocks`

Reads `block_index` instead. Use this when `interval_index` is disabled (`interval_index_mode="disable"` at ingest time) or known to be stale. Returns the same keys as the primary function plus `num_blocks` and `total_samples`.

```python
from atriumdb.dashboard.measure_queries import query_measure_total_hours_from_blocks

results = query_measure_total_hours_from_blocks(sdk)
```

---

## Exposing This as a FastAPI Endpoint

The AtriumDB mock API server (`sdk/tests/mock_api/`) uses **FastAPI** with a `Depends(get_sdk_instance)` injection pattern. No new framework, dependency, or connection management is needed — the endpoint follows the exact same shape as the existing `GET /measures/` route in `measures_endpoints.py`.

### What to add and where

| File | Change |
|---|---|
| `tests/mock_api/measures_endpoints.py` | Add one new route `GET /measures/hours` |
| `tests/mock_api/app.py` | No change needed — the new route sits under the existing `measures_router` already mounted at `/measures` |

### New route in `measures_endpoints.py`

Add this after the existing routes. No new imports are required beyond what the file already imports.

```python
@measures_router.get("/hours")
async def get_measure_total_hours(
        atriumdb_sdk: AtriumSDK = Depends(get_sdk_instance)):
    """
    Return total hours of data in interval_index for every measure,
    aggregated across all devices.
    """
    sql = """
        SELECT
            m.id                                                         AS measure_id,
            m.measure_tag,
            m.measure_name,
            m.freq_nhz,
            m.units,
            COUNT(DISTINCT ii.device_id)                                 AS num_devices,
            COUNT(*)                                                     AS num_intervals,
            SUM(ii.end_time_n - ii.start_time_n) / 3600000000000.0      AS total_hours
        FROM interval_index ii
        JOIN measures m ON m.id = ii.measure_id
        GROUP BY ii.measure_id
        ORDER BY total_hours DESC
    """
    with atriumdb_sdk.sql_handler.connection(begin=False) as (conn, cursor):
        cursor.execute(sql)
        rows = cursor.fetchall()

    keys = ["measure_id", "measure_tag", "measure_name", "freq_nhz", "units",
            "num_devices", "num_intervals", "total_hours"]
    return [dict(zip(keys, row)) for row in rows]
```

The route intentionally does not raise `HTTPException(404)` when the result is empty — an empty list is a valid, non-error response when no data has been ingested yet.

### Why `GET /measures/hours` and not `GET /measures/{measure_id}/hours`

A per-measure variant (`/{measure_id}/hours`) is possible but unnecessary here: the aggregation is cheap for all measures at once, and the caller can filter the returned list client-side. Adding a per-measure variant would only make sense if the table grew large enough that returning all rows became slow.

> **Route ordering note:** FastAPI matches routes top-to-bottom. `GET /measures/hours` must be registered **before** `GET /measures/{measure_id}`, otherwise FastAPI will capture the literal string `"hours"` as the `measure_id` path parameter and route to the wrong handler. In the current `measures_endpoints.py`, adding the new route before the `/{measure_id}` handler is sufficient.

### Response shape

```json
[
  {
    "measure_id": 1,
    "measure_tag": "HR",
    "freq_nhz": 500000000000,
    "units": "BPM",
    "num_devices": 12,
    "total_hours": 4821.6
  },
  {
    "measure_id": 2,
    "measure_tag": "SpO2",
    "freq_nhz": 125000000000,
    "units": "%",
    "num_devices": 10,
    "total_hours": 3104.2
  }
]
```

### Calling from an external server

```python
import httpx

response = httpx.get("http://<atriumdb-host>:8000/measures/hours")
response.raise_for_status()

for entry in response.json():
    print(f"{entry['measure_tag']}: {entry['total_hours']:.1f} h across {entry['num_devices']} devices")
```

---

## Key Facts to Remember

- All times in AtriumDB are **nanoseconds since Unix epoch** (UTC); no time zone conversion is applied by the database layer.
- `end_time_n` is **exclusive** in both `block_index` and `interval_index`.
- `interval_index` rows are non-overlapping within a `(measure_id, device_id)` pair — the SDK enforces this invariant on write.
- Blocks are linked to patients via `device_id → device_patient → patient_id`, not directly.
- The composite index `(measure_id, device_id, start_time_n, end_time_n)` on both tables means measure-level aggregation queries run efficiently even on large datasets.
