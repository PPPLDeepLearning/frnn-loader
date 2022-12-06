# -*- coding: utf-8 -*-


class NotDownloadedError(Exception):
    """Raised when a signal is accessed that is not on the file system."""

    pass


class SignalCorruptedError(Exception):
    """Raised when a signal is unusable."""

    pass


class SignalNotFoundError(Exception):
    """Raised when a signal wasn't found."""

    pass


class BadDownloadError(Exception):
    """Errors for data downloads."""

    pass


class DataFormatError(Exception):
    """Raised when data format is not as expected."""

    pass


class BadShotException(Exception):
    """Raised when a bad shot is processed."""

    pass


class MDSNotFoundException(Exception):
    """Raised when a signal is not found in MDS."""

    pass


class BadDataException(Exception):
    """Raised when data analysis goes bad."""

    pass

# end of file errors.py
