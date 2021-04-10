"""
__init__.py imports key functions from modules to package level
"""
from .__about__ import (
    __author__,
    __commit__,
    __copyright__,
    __email__,
    __license__,
    __summary__,
    __title__,
    __uri__,
    __version__,
)

from .utils.features import load_feature_file

from .extract import extract
from .predict import predict
from .select import select
from .parseconfig import parse_config

from . import metrics
from . import plot
from . import utils
