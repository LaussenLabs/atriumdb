
def _write_parquet(file_path, times, values, measure_tag, engine=None, **kwargs):
    engines = {}
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        engines['pyarrow'] = (pa, pq)
    except ImportError:
        pass

    try:
        import fastparquet
        import pandas as pd
        engines['fastparquet'] = fastparquet
    except ImportError:
        pass

    if not engines:
        raise ImportError("Neither pyarrow nor fastparquet+pandas is installed. "
                          "Please install one of them to proceed with parquet file export.")

    # Default engine selection logic
    if engine is None:
        if 'pyarrow' in engines:
            engine = 'pyarrow'
        else:
            engine = 'fastparquet'
    elif engine not in engines:
        raise ValueError(f"The specified engine {engine} is not installed.")

    if engine == 'pyarrow':
        pa, pq = engines['pyarrow']
        # Create a PyArrow Table
        table = pa.table({
            'Nanosecond Epoch': times,
            measure_tag: values
        })
        # Write the table to a Parquet file
        pq.write_table(table, file_path, **kwargs)
    elif engine == 'fastparquet':
        fp = engines['fastparquet']
        # Create a DataFrame and write using fastparquet

        df = pd.DataFrame({
            'Nanosecond Epoch': times,
            measure_tag: values
        })
        fp.write(file_path, df, **kwargs)
