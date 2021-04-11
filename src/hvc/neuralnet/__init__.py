from .models import dilated
from .models import flatwindow
from . import utils

import warnings

warnings.warn(
    "the `neuralnet` package is deprecated, and will be removed in version 0.5.0."
    "Please consider using the library `vak` if you need neural network models: "
    "https://github.com/NickleDave/vak",
    FutureWarning,
)
