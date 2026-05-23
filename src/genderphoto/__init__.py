"""
genderphoto - Gender classification using name inference + photo face analysis + VLM fallback.
"""

__version__ = "0.1.0"

import logging

def _suppress_noisy_loggers():
    noisy_loggers = [
        'icrawler',
        'icrawler.crawler',
        'urllib3',
        'tensorflow',
        'h5py',
        'PIL',
        'downloader',
        'parser',
        'feeder'
    ]
    for name in noisy_loggers:
        logging.getLogger(name).setLevel(logging.CRITICAL)

_suppress_noisy_loggers()

from genderphoto.name_classifier import classify_name


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


__all__ = [
    "classify_name",
    "classify_inventor",
    "classify_batch",
    "list_available_vlm_models",
    "__version__",
]
