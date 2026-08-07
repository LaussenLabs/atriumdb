# Running the AtriumDB test suite

Three invocations, in increasing order of cost. All of them run **both** metadata
backends unless you say otherwise — nothing here trades coverage for speed.

Everything below assumes the `atriumdb-test:latest` image and this repo mounted at
`/atriumdb`. `-p no:cacheprovider` just keeps `--rm` containers from writing
`.pytest_cache` into the checkout.

## 1. Fast inner loop — target under 5 minutes

SQLite only, no `slow` tests, no network, no MIT-BIH heavy paths.

```bash
docker run --rm -v "$PWD:/atriumdb" \
  -e PYTHONPATH=/atriumdb/sdk \
  atriumdb-test:latest \
  python -m pytest /atriumdb/sdk/tests -q -p no:cacheprovider \
    -m "not slow and not mariadb"
```

`-m "not mariadb"` also switches the legacy `_test_for_both` helper to SQLite-only, so
files that have not yet been converted to real parametrization obey the same selection.
`--backend sqlite` does the same thing explicitly and is easier to combine with other
`-m` expressions.

Locally, without Docker:

```bash
cd sdk && python -m pytest tests -q -m "not slow and not mariadb"
```

## 2. Full pre-merge run — both backends

```bash
docker run --rm -v "$PWD:/atriumdb" \
  -e PYTHONPATH=/atriumdb/sdk \
  -e MARIA_DB_HOST=host.docker.internal \
  -e MARIA_DB_USER=root -e MARIA_DB_PASSWORD=atriumdb -e MARIA_DB_PORT=3308 \
  atriumdb-test:latest \
  python -m pytest /atriumdb/sdk/tests -q -p no:cacheprovider
```

This runs everything except the `nightly` numeric gate (see below); that exclusion comes
from `addopts` in `sdk/pyproject.toml`. If `MARIA_DB_*` is not set the MariaDB half is
skipped with a warning rather than erroring out.

## 3. Nightly — the full-fidelity numeric gate

`test_mit_bih.py` and `test_iterator.py` are the repo's numeric regression backbone.
They keep their data volume **exactly** (no truncation, `MAX_RECORDS` unchanged) and are
excluded from the runs above by the `nightly` marker, because between them they are the
better part of an hour. Run them on a schedule:

```bash
docker run --rm -v "$PWD:/atriumdb" \
  -e PYTHONPATH=/atriumdb/sdk \
  -e MARIA_DB_HOST=host.docker.internal \
  -e MARIA_DB_USER=root -e MARIA_DB_PASSWORD=atriumdb -e MARIA_DB_PORT=3308 \
  atriumdb-test:latest \
  python -m pytest /atriumdb/sdk/tests -q -p no:cacheprovider -m nightly
```

To run absolutely everything in one go, clear the default marker expression:

```bash
python -m pytest /atriumdb/sdk/tests -q -o addopts=""
```

## Markers

Registered in `sdk/pyproject.toml` and in `sdk/tests/conftest.py`.

| marker | meaning |
| --- | --- |
| `slow` | takes more than ~30s; excluded from the inner loop |
| `mariadb` | needs a running MariaDB; `-m "not mariadb"` makes the whole run SQLite-only |
| `network` | needs network access — there should be none (see below) |
| `mitbih` | needs the MIT-BIH wfdb cache in `sdk/tests/wfdb_data` |
| `numeric_gate` | protected numeric regression test — **do not shrink its data** |
| `nightly` | full-fidelity gate, excluded from the default run, scheduled nightly |

## Backend selection

* `--backend both` (default) — MariaDB and SQLite.
* `--backend sqlite` / `--backend mariadb` — one only.
* `-m "not mariadb"` — equivalent to `--backend sqlite`.

Converted tests are parametrized, so each backend has its own test id
(`test_x[sqlite]`, `test_x[mariadb]`) and can be selected with `-k` and reported
independently. Files still using `testing_framework._test_for_both` run both backends
inside a single test id; the helper honours the same selection, and files are being
converted incrementally.

## The MIT-BIH cache and the network

`sdk/tests/wfdb_data/mitdb` holds the 48 mitdb records the suite uses.
`generate_wfdb.py` is **cache-first**: it derives the record list from what is on disk
and only contacts PhysioNet if the cache is missing, in which case it fails with an
actionable message. A normal run makes no HTTP request at all.

`ATRIUMDB_WFDB_CACHE` redirects the cache; `conftest.py` defaults it to the in-tree
directory, so no invocation needs to set it. Point it at a read-only mount in CI if you
prefer to keep the cache outside the checkout.

## MariaDB

Set `MARIA_DB_HOST`, `MARIA_DB_USER`, `MARIA_DB_PASSWORD` and `MARIA_DB_PORT`, either in
the environment or in a `.env` file at the repo root (see `.env.example`). From inside a
container, `MARIA_DB_HOST=host.docker.internal` reaches a MariaDB running on the host.
Explicit `-e` flags take precedence over `.env`.

When MariaDB is not configured at all, the MariaDB half is **skipped**, not failed.

One container-specific trap this suite now handles for you: a repo-root `.env` is
visible inside the container and usually says `MARIA_DB_HOST=127.0.0.1`, which is not
reachable from in there. On a SQLite-only run (`--backend sqlite` / `-m "not mariadb"`)
`conftest.py` therefore clears the `MARIA_DB_*` variables and neutralises `load_dotenv()`
for the run, so tests that read those variables directly skip instead of erroring on a
connection they were never meant to make. The default both-backend run is untouched.

## A note on data volume

Several tests take a `max_samples_per_record` argument through
`test_mit_bih.write_mit_bih_to_dataset` / `assert_mit_bih_to_dataset`. It defaults to
`None` (full 650,000-sample records). Consumers that only need the *structure* the helper
builds — devices, patients, device↔patient mappings, labels, measures — pass
`TRUNCATED_SAMPLES_PER_RECORD` (20,000). That is still more than one block per measure at
every block size those tests use, so block splitting, gap handling and multi-block merge
stay covered. **Do not lower it**, and do not raise the block-size sweep above `2**14`
for a truncating caller.
