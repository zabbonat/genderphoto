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

import requests

from genderphoto.constants import DEFAULT_MAX_IMAGES, DEFAULT_SEARCH_ENGINE

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
    search_engine: str = DEFAULT_SEARCH_ENGINE,
) -> list[dict]:
    """
    Search for photos of a person using Bing or DuckDuckGo image search.

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
    search_engine : str
        'bing' (default) or 'duckduckgo'.

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
    
    if search_engine.lower() == 'duckduckgo':
        from ddgs import DDGS
        for rank, (query, query_type) in enumerate(queries):
            tmp_dir = tempfile.mkdtemp(prefix='inv_photo_ddg_')
            try:
                with DDGS() as ddgs:
                    ddg_results = list(ddgs.images(query, max_results=max_images, safesearch='on'))
                
                # Download URLs
                for idx, r in enumerate(ddg_results):
                    img_url = r.get('image')
                    if not img_url:
                        continue
                    try:
                        resp = requests.get(img_url, timeout=5, stream=True)
                        resp.raise_for_status()
                        
                        # Validate and convert to RGB JPEG using PIL
                        from PIL import Image
                        from io import BytesIO
                        
                        img = Image.open(BytesIO(resp.content))
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                            
                        fpath = os.path.join(tmp_dir, f"{idx:06d}.jpg")
                        img.save(fpath, format='JPEG', quality=90)
                    except Exception as e:
                        log.debug("Failed to download DDG image %s: %s", img_url, e)
                
                for fpath in glob.glob(os.path.join(tmp_dir, '*')):
                    results.append({
                        'url': fpath,
                        'query': query,
                        'query_type': query_type,
                        'query_rank': rank,
                    })
                    
                if results:
                    _temp_dirs.append(tmp_dir)
                    log.info("Found %d images for '%s' via DDG %s", len(results), name, query_type)
                    return results
                else:
                    shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception as e:
                shutil.rmtree(tmp_dir, ignore_errors=True)
                log.warning("DDG search failed for '%s': %s", query, e)
            time.sleep(sleep)
            
    elif search_engine.lower() == 'baidu':
        from icrawler.builtin import BaiduImageCrawler
        for rank, (query, query_type) in enumerate(queries):
            tmp_dir = tempfile.mkdtemp(prefix='inv_photo_')
            try:
                crawler = BaiduImageCrawler(
                    storage={'root_dir': tmp_dir},
                    log_level=logging.WARNING,
                )
                crawler.crawl(
                    keyword=query,
                    max_num=max_images,
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
                    _temp_dirs.append(tmp_dir)
                    log.info("Found %d images for '%s' via Baidu %s", len(results), name, query_type)
                    return results
                else:
                    shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception as e:
                shutil.rmtree(tmp_dir, ignore_errors=True)
                log.warning("Baidu search failed for '%s': %s", query, e)
            time.sleep(sleep)

    else:  # bing
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
                    filters={'safe': 'strict'},
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
