"""A context manager that redirects stdout, even for non-python code (C, Fortran, etc)."""
import io
import os
import sys
import tempfile


class forcefully_redirect_stdout(object):
    """A context manager that redirects stdout, even non-python code.

    Forces stdout to be redirected, for both python code and C/C++/Fortran
    or other linked libraries.  Useful for scraping values from e.g. the
    disp option for scipy.optimize.minimize.

    Attributes
    ----------
    captured : str
        Description
    fd : TYPE
        Description
    old_stdout : TYPE
        Description
    target : TYPE
        Description
    to : TYPE
        Description
    captured : `str`
    The captured text.
    fd : filedescriptor
    System filedescriptor.  Do not access.
    old_stdout : either `sys.stdout` or `sys.__stdout__`
    The old stdout, useful to write to it with the context manager even during
    redirection.
    target : either `sys.stdout` or `sys.__stdout__`
    The target to redirect.
    to : `str` or file_like
    Where to redirect output to.

    """

    def __init__(self, to=None):
        """Create a new forcefully_redirect_stdout context manager.

        Parameters
        ----------
        to : `None` or `str`
            what to redirect to.  If type(to) is `None`,
            internally uses a `tempfile.SpooledTemporaryFile` and returns a
            UTF-8 string containing the captured output.  If type(to) is str,
            opens a file at that path and pipes output into it, erasing prior
            contents.

        """
        # typecheck sys.stdout -- if it's a textwrapper, we're in a shell
        # if it's not, we're likely in an IPython terminal or jupyter kernel
        # and need to target sys.__stdout__ instead of sys.stdout
        if type(sys.stdout) is io.TextIOWrapper:
            self.target = sys.stdout
        else:
            self.target = sys.__stdout__

        # initialize where we will redirect to and a file descriptor for python
        # stdout -- self.target is used by python, while os.fd(1) is used by
        # C/C++/Fortran/etc
        self.to = to
        self.fd = self.target.fileno()
        if self.to is None:
            self.to = tempfile.SpooledTemporaryFile(mode='w+b')
        else:
            self.to = open(to, 'w+b')

        self.old_stdout = os.fdopen(os.dup(self.fd), 'w')
        self.captured = ''

    def __enter__(self):
        """Upon entering the context.

        Returns
        -------
        forcefully_redirect_stoud
            This instance of the context manager.

        """
        self._redirect_stdout(to=self.to)
        return self

    def __exit__(self, *args):
        """Upon exiting the context.

        Parameters
        ----------
        *args
            Any arguments; signature required of __exit__.

        """
        self._redirect_stdout(to=self.old_stdout)
        self.to.seek(0)
        self.captured = self.to.read().decode('utf-8')
        self.to.close()

    def _redirect_stdout(self, to):
        """Redirect stdout to the new location.

        Parameters
        ----------
        to : `str` or file_like
            where to send stdout to.

        """
        self.target.close()  # implicit flush()
        os.dup2(to.fileno(), self.fd)  # fd writes to 'to' file
        self.target = os.fdopen(self.fd, 'w')  # Python writes to fd
