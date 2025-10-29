"""
Utility Functions Package

A comprehensive collection of utility functions for various tasks including
file operations, array processing, time conversions, statistics, string
manipulation, signal processing, data visualization, and state-space modeling.

Quick Usage:
    # Import everything
    from utils import *

    # Or import specific modules
    from utils import file_utils, time_utils, eegvis_utils, state_space_utils

    # Or import specific functions
    from utils import create_folder, sec2time, robustSTD, viz_eeg, estimateA

Available Modules:
    - file_utils: File system operations
    - array_utils: Array processing and windowing
    - time_utils: Time format conversions
    - stats_utils: Statistical analysis functions
    - string_utils: String parsing utilities
    - signal_utils: Signal processing functions
    - eegvis_utils: EEG/iEEG visualization functions
    - state_space_utils: State-space modeling and network analysis

Version: 1.0.0
"""

# Import all functions from each module
from .file_utils import (
    create_folder,
    create_folders
)

from .array_utils import (
    overlap
)

from .time_utils import (
    sec2h,
    sec2time,
    time2sec,
    time_sum,
    compute_date
)

from .stats_utils import (
    robustSTD,
    kernelModeEstimate,
    tukeys_method
)

from .string_utils import (
    text_num_split,
    split_label
)

from .signal_utils import (
    _xlogx,
    change_pnt,
    analyze_correlations,
    compute_spectrogram,
    compute_signal_energy
)

from .eegvis_utils import (
    viz_eeg
)

from .state_space_utils import (
    estimateA,
    estimateA_subject,
    reconstruct_signal,
    identifySS
)

# Also import the modules themselves for namespace access
from . import file_utils
from . import array_utils
from . import time_utils
from . import stats_utils
from . import string_utils
from . import signal_utils
from . import eegvis_utils
from . import state_space_utils

# Define what gets imported with "from utils import *"
__all__ = [
    # Modules
    'file_utils',
    'array_utils',
    'time_utils',
    'stats_utils',
    'string_utils',
    'signal_utils',
    'eegvis_utils',
    'state_space_utils',

    # File utilities
    'create_folder',
    'create_folders',

    # Array utilities
    'overlap',

    # Time utilities
    'sec2h',
    'sec2time',
    'time2sec',
    'time_sum',
    'compute_date',

    # Statistical utilities
    'robustSTD',
    'kernelModeEstimate',
    'tukeys_method',

    # String utilities
    'text_num_split',
    'split_label',

    # Signal utilities
    '_xlogx',
    'change_pnt',
    'analyze_correlations',
    'compute_spectrogram',
    'compute_signal_energy',

    # Visualization utilities
    'viz_eeg',

    # State-space utilities
    'estimateA',
    'estimateA_subject',
    'reconstruct_signal',
    'identifySS'
]

# Package metadata
__version__ = '1.0.0'
__author__ = 'Amir H Daraie'
__email__ = 'adaraie1@jhu.edu'
__description__ = 'Comprehensive utility functions for data processing and analysis'

# Optional: Package-level initialization message
import sys
if '--verbose' in sys.argv or '-v' in sys.argv:
    print(f"Utility Package v{__version__} loaded successfully")
    print(f"Available modules: {', '.join([m for m in __all__ if '_utils' in m])}")
    print(f"Total functions available: {len([f for f in __all__ if '_utils' not in f])}")