"""A context manager that redirects stdout, even for non-python code (C, Fortran, etc)."""
import io
import os
import sys
import tempfile
import contextlib


@contextlib.contextmanager
def forcefully_redirect_stdout():
    """Redirect stdout at the system level.

    Used to capture data from scipy.optimize.minimize

    Yields:
        `dict`: dict with a txt key after the context exits

    """
    if type(sys.stdout) is io.TextIOWrapper:
        target = sys.stdout
    else:
        target = sys.__stdout__

    fd = target.fileno()
    restore_fd = os.dup(fd)
    try:
        tmp, out = tempfile.SpooledTemporaryFile(mode='w+b'), {}
        os.dup2(tmp.fileno(), fd)
        yield out
        os.dup2(restore_fd, fd)
    finally:
        tmp.flush()
        tmp.seek(0)
        out['txt'] = tmp.read().decode('utf-8')
        tmp.close()
