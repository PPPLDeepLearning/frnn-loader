# -*- coding: utf-8 -*-

class NotDownloadedError(Exception):
    """Raised when a signal is accessed that is not on the file system."""
    pass


class SignalCorruptedError(Exception):
    """Raised when a signal is unusable."""
    pass

class SignalNotFoundError(Exception):
    """Raised when a signal wasn't found."""

class DataFormatError(Exception):
    """Raised when data format is not as expected."""

class BadShotException(Exception):
    """Raised when a bad shot is processed."""
    pass

# end of file errors.py