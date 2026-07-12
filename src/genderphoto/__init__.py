"""
genderphoto - Gender classification using name inference + photo face analysis + VLM fallback.
"""

__version__ = "0.1.0"

import logging

import os
import warnings

# Suppress TensorFlow C++ logs and oneDNN warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['ABSL_LOG_LEVEL'] = '3'
# Suppress Python warnings (like the tf_keras deprecation)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', module='tensorflow')
warnings.filterwarnings('ignore', module='tf_keras')

def _suppress_noisy_loggers():
    noisy_loggers = [
        'icrawler', 'icrawler.crawler', 'urllib3', 'tensorflow',
        'h5py', 'PIL', 'downloader', 'parser', 'feeder', 'tf_keras'
    ]
    for name in noisy_loggers:
        logging.getLogger(name).setLevel(logging.CRITICAL)

_suppress_noisy_loggers()

def classify_name(*args, **kwargs):
    """Lazy-loaded wrapper for name_classifier.classify_name."""
    from genderphoto.name_classifier import classify_name as _cn
    return _cn(*args, **kwargs)


def classify_inventor(*args, **kwargs):
    """Lazy-loaded wrapper for pipeline.classify_inventor."""
    from genderphoto.pipeline import classify_inventor as _ci
    return _ci(*args, **kwargs)



def classify_batch(*args, **kwargs):
    """Lazy-loaded wrapper for batch.classify_batch."""
    from genderphoto.batch import classify_batch as _cb
    return _cb(*args, **kwargs)


def list_available_vlm_models(*args, **kwargs):
    """Lazy-loaded wrapper for vlm_classifier.list_available_vlm_models."""
    from genderphoto.vlm_classifier import list_available_vlm_models as _lm
    return _lm(*args, **kwargs)


def compute_partial_identification_bounds(*args, **kwargs):
    """Lazy-loaded wrapper for utils.compute_partial_identification_bounds."""
    from genderphoto.utils import compute_partial_identification_bounds as _cpi
    return _cpi(*args, **kwargs)


__all__ = [
    "classify_name",
    "classify_inventor",
    "classify_batch",
    "list_available_vlm_models",
    "compute_partial_identification_bounds",
    "__version__",
]

