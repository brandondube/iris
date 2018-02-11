"""A rudimentary database object."""
import uuid
import pickle
from pathlib import Path

import pandas as pd


class Database(object):
    """A database."""

    def __init__(self, path=None, fields=None):
        """Initialize a database.

        Parameters
        ----------
        path : `pathlib.Path` or `str` optional
            path to the folder containing the database; None if creating a new db
        fields : iterable, optional
            set of fields to use if creating a new database, else ignored

        Notes
        -----
        if both path and fields are provided, path takes precedence and fields will be ignored

        """
        self.path = Path(path)
        self.data_root = path / 'db'

        if path is not None:
            self.df = None
            self.fields = None
            self._load_from_disk()
        else:
            self.fields = fields
            self.df = pd.DataFrame(columns=fields)

    def _load_from_disk(self):
        """Load a database from disk."""
        self.df = pd.load_csv(self.path / 'index.csv')
        self.fields = list(self.df)

    def append(self, document):
        """Append a document to the database.

        Parameters
        ----------
        document : `dict`
            a dictionary with several fields

        """
        id = str(uuid.uuid4())
        row_item = {'id': id}
        for field in self.fields:
            row_item[field] = document[field]

        self.df.append(row_item)
        with open(self.data_root / id, 'wb') as fid:
            pickle.dump(document, fid)

    def get_document(self, id):
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
        with open(self.data_root / id, 'rb') as fid:
            doc = pickle.load(fid)
        return doc
