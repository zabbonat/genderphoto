"""
Photo search via Bing image search (icrawler).

ONLY BingImageCrawler is used. Google parser is broken (raises
'NoneType is not iterable'). DDG API is unreliable.
"""

from __future__ import annotations

import atexit
import glob
import logging
import os
import shutil
import tempfile
import time

from genderphoto.constants import DEFAULT_MAX_IMAGES

log = logging.getLogger(__name__)

# Track temp dirs for cleanup on process exit
_temp_dirs: list[str] = []


def _cleanup_temp_dirs() -> None:
    for d in _temp_dirs:
        shutil.rmtree(d, ignore_errors=True)
    _temp_dirs.clear()


atexit.register(_cleanup_temp_dirs)


def search_photos(
    name: str,
    affiliation: str = None,
    max_images: int = DEFAULT_MAX_IMAGES,
    sleep: float = 1.0,
) -> list[dict]:
    """
    Search for photos of a person using Bing image search.

    Strategy (tiered, stops at first success):
      1. "{name} {affiliation}" (institutional)
      2. "{name} researcher"   (role keywords)
      3. "{name}"              (name only)

    Parameters
    ----------
    name : str
        Full name of the person.
    affiliation : str, optional
        Institutional affiliation to improve search quality.
    max_images : int
        Maximum images to download per query.
    sleep : float
        Sleep between queries (seconds).

    Returns
    -------
    list[dict]
        Each dict has keys: 'url' (local file path), 'query', 'query_type',
        'query_rank'.
    """
    queries = []
    if affiliation:
        queries.append((f'{name} {affiliation}', 'institutional'))
    queries.append((f'{name} researcher', 'role_keywords'))
    queries.append((f'{name}', 'name_only'))

    results = []
    from icrawler.builtin import BingImageCrawler
    for rank, (query, query_type) in enumerate(queries):
        tmp_dir = tempfile.mkdtemp(prefix='inv_photo_')
        try:
            crawler = BingImageCrawler(
                storage={'root_dir': tmp_dir},
                log_level=logging.WARNING,
            )
            crawler.crawl(
                keyword=query,
                max_num=max_images,
                min_size=(100, 100),
                file_idx_offset=0,
            )
            for fpath in glob.glob(os.path.join(tmp_dir, '*')):
                results.append({
                    'url': fpath,
                    'query': query,
                    'query_type': query_type,
                    'query_rank': rank,
                })
            if results:
                # Keep dir alive until process exit (images are local paths)
                _temp_dirs.append(tmp_dir)
                log.info(
                    "Found %d images for '%s' via %s",
                    len(results), name, query_type,
                )
                return results
            else:
                # No results from this query, clean up immediately
                shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception as e:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            log.warning("Search failed for '%s': %s", query, e)
        time.sleep(sleep)

    log.warning("No images found for '%s'", name)
    return []
