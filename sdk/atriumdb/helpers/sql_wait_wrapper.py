from sqlalchemy.exc import OperationalError
import time

DB_LOCKED_STRING = "database is locked"


def sql_lock_wait(fn):
    def wrap(*args, **kwargs):
        while True:
            try:
                return fn(*args, **kwargs)
            except OperationalError as e:
                error_string = str(e)
                if DB_LOCKED_STRING in error_string:
                    time.sleep(0.1)
                    continue
                else:
                    raise e
    return wrap
