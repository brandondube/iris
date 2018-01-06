import io
import os
import sys
import tempfile


class forcefully_redirect_stdout(object):
    ''' Forces stdout to be redirected, for both python code and C/C++/Fortran
        or other linked libraries.  Useful for scraping values from e.g. the
        disp option for scipy.optimize.minimize.
    '''
    def __init__(self, to=None):
        ''' Creates a new forcefully_redirect_stdout context manager.

        Args:
            to (`None` or `str`): what to redirect to.  If type(to) is None,
                internally uses a tempfile.SpooledTemporaryFile and returns a UTF-8
                string containing the captured output.  If type(to) is str, opens a
                file at that path and pipes output into it, erasing prior contents.

        Returns:
            `str` if type(to) is None, else returns `None`.

        '''
        # typecheck sys.stdout -- if it's a textwrapper, we're in a shell
        # if it's not, we're likely in an IPython terminal or jupyter kernel
        # and need to target sys.__stdout__ instead of sys.stdout
        if type(sys.stdout) is io.TextIOWrapper:
            self.target = sys.stdout
        else:
            self.target = sys.__stdout__

        # initialize where we will redirect to and a file descriptor for python
        # stdout -- self.targetstdout is used by python, while os.fd(1) is used by
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
        self._redirect_stdout(to=self.to)
        return self

    def __exit__(self, *args):
        self._redirect_stdout(to=self.old_stdout)
        self.to.seek(0)
        self.captured = self.to.read().decode('utf-8')
        self.to.close()

    def _redirect_stdout(self, to):
        self.target.close()  # implicit flush()
        os.dup2(to.fileno(), self.fd)  # fd writes to 'to' file
        self.target = os.fdopen(self.fd, 'w')  # Python writes to fd
