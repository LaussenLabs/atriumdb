# Changelog for AtriumDB

All notable changes to this project will be documented in this file.

## [2.1.0] - 2023-10-20

### Added

- **DatasetDefinition class**: Introduced a new `DatasetDefinition` class to define data sources, signals, and times. This encapsulates information vital for features like data exports and data iterators.
  
- **Definition File Format**: Introduced a file format to work alongside the `DatasetDefinition`. This allows the raw data of a `DatasetDefinition` object to be saved to or loaded from a file with `.yaml` or `.yml` extension.

- **DatasetIterator class**: Added a new `DatasetIterator` class which implements the `__next__`, `__len__`, and `__getitem__` methods. This can be used to iterate through time windows of data as described by the `DatasetDefinition` class.

- **AtriumSDK.get_iterator Method**: A new method in the main module's class. The `AtriumSDK.get_iterator` method consumes a `DatasetDefinition`, validates the definition with the actual data available in the datastore targeted by the `AtriumSDK` object, and returns a `DatasetIterator` object if the `DatasetDefinition` was successfully validated.

- **Patient Parameters**: Now supports `height` and `weight` as patient parameters/keys in dictionaries returned or set by AtriumSDK methods like `insert_patient`, `get_patient_info`, and `get_all_patients`.

### Fixed

- Resolved a bug encountered when the `atriumdb` library accessed datastores set up by different software versions. This was addressed by substituting `SELECT *` wildcards in SQL queries with the precise column names required by each query.
