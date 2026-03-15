"""
genderphoto - Gender classification using name inference + photo face analysis + VLM fallback.
"""

__version__ = "0.1.0"

from genderphoto.name_classifier import classify_name


def classify_inventor(*args, **kwargs):
    """Lazy-loaded wrapper for pipeline.classify_inventor."""
    from genderphoto.pipeline import classify_inventor as _ci
    return _ci(*args, **kwargs)


def classify_batch(*args, **kwargs):
    """Lazy-loaded wrapper for batch.classify_batch."""
    from genderphoto.batch import classify_batch as _cb
    return _cb(*args, **kwargs)


__all__ = [
    "classify_name",
    "classify_inventor",
    "classify_batch",
    "__version__",
]
