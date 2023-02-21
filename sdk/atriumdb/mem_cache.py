import functools


cache = {}


def cached(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        key = (func,) + args + tuple(sorted(kwargs.items()))
        if key in cache:
            return cache[key]
        else:
            result = func(*args, **kwargs)
            cache[key] = result
            return result
    return wrapper
