from time import time
from functools import wraps

timer_enabled = False
crash = False

def enable_timer():
    """
    Enable the timer. Disabled by default when util/timeit.py is imported.
    """
    global timer_enabled
    timer_enabled = True

def disable_timer():
    """
    Disable the timer. Disabled by default when util/timeit.py is imported.
    """
    global timer_enabled
    timer_enabled = False

def timeit(func):
    """
    Decorator to time a function.

    Args:
        func: The function to time.
    """
    @wraps(func)
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        if timer_enabled:
            print('-'*100)
            print(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s')
            print('-'*100)
        return result
    return wrap_func

timeit_dict = {}

def time_start(name: str):
    """
    Start a timer for a block of code.

    Args:
        name: The name of the block of code. Names must be unique compared to other currently running blocks.
    """
    if timer_enabled:
        if name in timeit_dict:
            if crash:
                raise ValueError(f'Block {name!r} already exists')
            else:
                print(f'Block {name!r} already exists')
        timeit_dict[name] = time()

def time_end(name: str):
    """
    End a timer for a block of code.

    Args:
        name: The name of the block. The name must match the name used in time_start.
    """
    try:
        if timer_enabled:
            print('-'*100)
            print(f'Block {name!r} executed in {(time()-timeit_dict[name]):.4f}s')
            print('-'*100)
            timeit_dict.pop(name)
    except KeyError:
        if crash:
            raise ValueError(f'Block {name!r} does not exist')
        else:
            print(f'Block {name!r} does not exist')