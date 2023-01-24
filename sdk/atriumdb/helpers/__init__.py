from importlib import resources
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

_cfg = tomllib.loads(resources.read_text("atriumdb.helpers", "config.toml"))
shared_lib_filename_linux = _cfg["resources"]["shared_lib_filename_linux"]
shared_lib_filename_windows = _cfg["resources"]["shared_lib_filename_windows"]

protected_mode_default_setting = _cfg["settings"]["protected_mode"]
overwrite_default_setting = _cfg["settings"]["overwrite"]
