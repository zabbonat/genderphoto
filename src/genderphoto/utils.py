"""
Utility functions: image loading, saving, logging setup, name parsing.
"""

from __future__ import annotations

import logging
import os
import re
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image

log = logging.getLogger(__name__)


def setup_logging(
    log_file: str = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Configure the root genderphoto logger.

    Parameters
    ----------
    log_file : str, optional
        Path to a log file. If provided, logs are also written to this file
        with UTF-8 encoding (Windows-safe).
    level : int
        Logging level (default INFO).

    Returns
    -------
    logging.Logger
        The configured 'genderphoto' logger.
    """
    logger = logging.getLogger('genderphoto')
    logger.setLevel(level)

    # Avoid duplicate handlers on repeated calls
    if not logger.handlers:
        fmt = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

        # Console handler
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)

        # File handler (utf-8 for Windows compatibility)
        if log_file:
            fh = logging.FileHandler(log_file, encoding='utf-8')
            fh.setFormatter(fmt)
            logger.addHandler(fh)

    return logger


def load_image(path_or_url: str, timeout: int = 10) -> Image.Image | None:
    """
    Load an image from a local path or URL.

    Returns None if the image cannot be loaded or is smaller than 50x50.
    """
    try:
        if os.path.isfile(path_or_url):
            img = Image.open(path_or_url).convert('RGB')
        else:
            r = requests.get(
                path_or_url,
                headers={'User-Agent': 'Mozilla/5.0'},
                timeout=timeout,
            )
            r.raise_for_status()
            img = Image.open(BytesIO(r.content)).convert('RGB')
        if img.width >= 50 and img.height >= 50:
            return img
        return None
    except Exception:
        return None


def save_photo(
    img: Image.Image,
    name: str,
    photo_dir: str = './inventor_photos',
) -> str:
    """
    Save an image to disk with a filesystem-safe name derived from the
    inventor's name.

    Returns the saved file path as a string.
    """
    photo_dir = Path(photo_dir)
    photo_dir.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
    path = photo_dir / f"{safe}.jpg"
    img.save(path, quality=90)
    log.debug("Saved photo -> %s", path)
    return str(path)


def extract_first_name(full_name: str) -> str:
    """
    Extract the first name from a full name string.

    Handles both 'First Last' and 'Last, First' formats.
    """
    full_name = full_name.strip()
    if ',' in full_name:
        return full_name.split(',')[1].strip().split()[0]
    return full_name.split()[0]
