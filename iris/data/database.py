"""A rudimentary database object."""
import uuid
import pickle
import asyncio
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
        self.data_root.mkdir(parents=True, exist_ok=True)  # ensure database folders exist

        if path is not None:
            self.df = None
            self.fields = None
            self._load_from_disk()
        else:
            self.fields = fields
            self.df = pd.DataFrame(columns=fields)

        self.os_lock = None

    def _load_from_disk(self):
        """Load a database from disk."""
        self.df = pd.load_csv(self.path / 'index.csv')
        self.fields = self.df.columns.tolist()

    def append(self, document):
        """Append a document to the database.

        Parameters
        ----------
        document : `dict`
            a dictionary with several fields

        """
        id_ = str(uuid.uuid4())
        row_item = {'id': id_}
        for field in self.fields:
            row_item[field] = document[field]

        self.df.append(row_item)
        with open(self.data_root / id_, 'wb') as fid:
            pickle.dump(document, fid)

        if isinstance(self.os_lock, asyncio.Task):  # can be None if db just initialized
            if not self.os_lock.done():  # if csv write incomplete, interrupt
                self.os_lock.cancel()
        self.os_lock = asyncio.ensure_future(write_df_to_file(self.df, self.path))  # write to file

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
        with open(self.data_root / id_, 'rb') as fid:
            doc = pickle.load(fid)
        return doc


async def write_df_to_file(df, path):
    """Asyncronously write a dataframe to a csv file.

    Parameters
    ----------
    df : `pandas.DataFrame`
        a pandas df
    path : path_like
        where to writ the df to

    """
    await df.to_csv(path, index=False)
