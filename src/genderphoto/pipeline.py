"""
Main orchestrator: classify a single inventor.

Runs the full pipeline:
1. Parse name -> classify_name
2. If unambiguous -> return name-based result
3. If ambiguous -> search photos -> run ensemble -> return result
"""

from __future__ import annotations

import logging

from genderphoto.constants import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_MAX_IMAGES,
    DEFAULT_VLM,
    OLLAMA_URL,
)
from genderphoto.ensemble import run_ensemble
from genderphoto.name_classifier import classify_name
from genderphoto.photo_search import search_photos
from genderphoto.utils import extract_first_name, save_photo

log = logging.getLogger(__name__)


def classify_inventor(
    name: str,
    affiliation: str = None,
    country_code: str = None,
    max_images: int = DEFAULT_MAX_IMAGES,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    save_photo_flag: bool = False,
    photo_dir: str = './inventor_photos',
    vlm_model: str = DEFAULT_VLM,
    ollama_url: str = OLLAMA_URL,
) -> dict:
    """
    Full classification pipeline for a single inventor.

    Parameters
    ----------
    name : str
        Full name of the inventor (e.g. "Andrea Cavalleri").
    affiliation : str, optional
        Institutional affiliation (improves photo search).
    country_code : str, optional
        ISO 2-letter country code (for cross-cultural name checks).
    max_images : int
        Maximum images to download and analyze.
    confidence_threshold : float
        Minimum confidence (%) to accept a photo-based classification.
    save_photo_flag : bool
        If True, save the best photo to photo_dir.
    photo_dir : str
        Directory for saved photos.
    vlm_model : str
        Ollama VLM model name.
    ollama_url : str
        Ollama API endpoint URL.

    Returns
    -------
    dict
        {
            'inventor_name': str,
            'gender': 'M' | 'F' | 'UNKNOWN',
            'confidence': float | None,
            'method': str,
            'is_ambiguous': bool,
            'photo_url': str | None,
            'photo_saved_path': str | None,
            'images_tried': int,
            'error': str | None,
        }
    """
    first_name = extract_first_name(name)

    # Stage 1: name-based classification
    name_result = classify_name(first_name, country_code)

    base = {
        'inventor_name': name,
        'gender': name_result['gender'] or 'UNKNOWN',
        'confidence': None,
        'method': name_result['method'],
        'is_ambiguous': name_result['is_ambiguous'],
        'gender_raw': name_result['gender_raw'],
        'ambiguity_reason': name_result['ambiguity_reason'],
        'photo_url': None,
        'photo_saved_path': None,
        'images_tried': 0,
        'error': None,
    }

    # If name is unambiguous, return immediately
    if not name_result['is_ambiguous'] and name_result['gender'] is not None:
        log.info(
            "%s -> %s (name_based, %s)",
            name, name_result['gender'], name_result['gender_raw'],
        )
        return base

    # Stage 2: photo-based classification
    log.info("%s -> ambiguous (%s), searching photos...", name, name_result['ambiguity_reason'])

    try:
        photos = search_photos(name, affiliation, max_images)
    except Exception as e:
        log.warning("Photo search error for %s: %s", name, e)
        base['error'] = f'photo_search_error: {str(e)[:100]}'
        return base

    if not photos:
        base['error'] = 'no_images_found'
        log.warning("  No images for %s", name)
        return base

    # Stage 3: ensemble classification
    result, tried, best_img = run_ensemble(
        photos, max_images, vlm_model=vlm_model, ollama_url=ollama_url,
    )
    photo_meta = result.pop('_photo', {})
    base['images_tried'] = tried

    if result.get('gender') and (result.get('confidence') or 0) >= confidence_threshold:
        saved = None
        if save_photo_flag and best_img:
            saved = save_photo(best_img, name, photo_dir)

        base['gender'] = result['gender']
        base['confidence'] = result['confidence']
        base['method'] = result.get('classifier', 'photo')
        base['photo_url'] = photo_meta.get('url', '')
        base['photo_saved_path'] = saved
        base['error'] = result.get('error')
        log.info(
            "  -> %s (%s%%, %s, %d imgs)",
            result['gender'], result['confidence'],
            result.get('classifier'), tried,
        )
    else:
        base['gender'] = 'UNKNOWN'
        base['method'] = 'unresolved'
        base['error'] = result.get('error', 'no_confident_classification')
        log.warning("  Could not classify %s (%d imgs)", name, tried)

    return base
