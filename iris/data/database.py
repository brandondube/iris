"""A rudimentary database object."""
import os
import shutil
import uuid
import pickle
from pathlib import Path

import pandas as pd


class Database(object):
    """A database."""

    def __init__(self, path, fields=None, overwrite=False):
        """Initialize a database.

        Parameters
        ----------
        path : `pathlib.Path` or `str`
            path to the folder containing the database
        fields : iterable, optional
            set of fields to use if creating a new database, else ignored
        overwrite: `bool`, optional
            whether to overwrite an existing database at path

        Notes
        -----
        if fields is provided and a db exists at the given path already, an exception will be raised

        """
        self.path = Path(path)
        self.data_root = self.path / 'db'
        self.csvpath = self.path / 'index.csv'

        if fields is None:  # initialize the db from the path
            self.df = None
            self.fields = None
            self._load_from_disk()
        else:  # if not, create from scratch or raise if there is a db at the path and overwrite=False
            if os.path.isfile(self.csvpath):
                if not overwrite:
                    raise IOError('There is an existing database at this location.  Delete it, or use overwrite=True.')
                else:
                    shutil.rmtree(self.data_root)
                    os.remove(self.csvpath)
            if 'id' in fields:
                raise ValueError('cannot use ID as a field')
            self.fields = tuple(fields)
            self.df = pd.DataFrame(columns=(*fields, 'id'))
        self.data_root.mkdir(parents=True, exist_ok=True)  # ensure database folders exist

    def init_csv(self):
        """Initialize the database index to disk."""
        if self.df.empty:
            self._write_to_disk()
        else:
            raise UserWarning('Database is not empty.')

    def _load_from_disk(self):
        """Load a database from disk."""
        self.df = pd.read_csv(self.csvpath)
        fields = self.df.columns.tolist()
        self.fields = [f for f in fields if f != 'id']

    def _write_to_disk(self):
        """Write the dataframe to disk."""
        self.df.to_csv(self.csvpath, index=False)

    def append(self, document):
        """Append a document to the database.

        Parameters
        ----------
        document : `dict`
            a dictionary with several fields

        """
        id_ = str(uuid.uuid4())  # get a unique ID for the document
        row_item = {'id': id_}
        for field in self.fields:  # build the dataframe row
            row_item[field] = document[field]

        self.df = self.df.append(row_item, ignore_index=True)  # assign to df, does not modifiy in-place
        with open(self.data_root / f'{id_}.pkl', 'wb') as fid:  # write the file to disk
            pickle.dump(document, fid)

        self._write_to_disk()

    def get_document(self, id_):
        """Return a document from the database.

        Parameters
        ----------
        id : `str`
            string document id

        Returns
        -------
        object
            object keyed by this id

        """
        with open(self.data_root / f'{id_}.pkl', 'rb') as fid:
            doc = pickle.load(fid)
        return doc