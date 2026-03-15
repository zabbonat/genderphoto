"""
Ensemble logic: DeepFace consensus across ALL images + VLM fallback.

Decision tree:
1. Run DeepFace on ALL images, collect valid results
2. Count consensus: n_M males, n_F females
3. If ALL agree AND avg confidence >= 90% -> ACCEPT (deepface_consensus)
4. If DISAGREE or low confidence -> ask VLM on best image
   - VLM agrees with majority -> CONFIRMED (ensemble_vlm_majority_agree)
   - VLM disagrees -> trust VLM (ensemble_vlm_override)
   - VLM fails -> fallback to DeepFace majority
"""

from __future__ import annotations

import logging

import numpy as np
from PIL import Image

from genderphoto.constants import DEEPFACE_HIGH_CONF, DEFAULT_VLM, OLLAMA_URL
from genderphoto.face_classifier import classify_face
from genderphoto.utils import load_image
from genderphoto.vlm_classifier import classify_vlm

log = logging.getLogger(__name__)


def run_ensemble(
    photos: list[dict],
    max_images: int = 5,
    vlm_model: str = DEFAULT_VLM,
    ollama_url: str = OLLAMA_URL,
) -> tuple[dict, int, Image.Image | None]:
    """
    Run the ensemble classifier on a set of photos.

    Parameters
    ----------
    photos : list[dict]
        Photo dicts from search_photos(), each with 'url' key.
    max_images : int
        Maximum images to process.
    vlm_model : str
        Ollama VLM model name for fallback.
    ollama_url : str
        Ollama API endpoint URL.

    Returns
    -------
    tuple[dict, int, Image | None]
        (result_dict, images_tried, best_image)
        result_dict has key '_photo' with the best photo metadata.
    """
    images_tried = 0
    valid_results = []

    # === PHASE 1: DeepFace on ALL images ===
    for photo in photos[:max_images]:
        img = load_image(photo['url'])
        if img is None:
            images_tried += 1
            continue
        images_tried += 1

        result = classify_face(img)
        if not result['face_detected'] or result['gender'] is None:
            log.info(
                "    Img %d: %s", images_tried, result.get('error', 'skip'),
            )
            continue

        log.info(
            "    Img %d: %s (%s%%, %df, %s)",
            images_tried, result['gender'], result['confidence'],
            result['n_faces'], result['classifier'],
        )
        valid_results.append((
            result['gender'], result['confidence'], img, photo, result,
        ))

    if not valid_results:
        return {
            'gender': None, 'confidence': None, 'face_detected': False,
            'n_faces': 0, 'classifier': 'all_failed',
            'error': 'no_usable_face', '_photo': {},
        }, images_tried, None

    # === PHASE 2: Consensus ===
    genders = [r[0] for r in valid_results]
    confs = [r[1] for r in valid_results]
    n_M = genders.count('M')
    n_F = genders.count('F')
    best_idx = max(range(len(valid_results)), key=lambda i: valid_results[i][1])
    best_gender, best_conf, best_img, best_photo, best_raw = valid_results[best_idx]

    log.info(
        "    Consensus: %dxM, %dxF across %d images",
        n_M, n_F, len(valid_results),
    )

    # All agree + high avg confidence -> ACCEPT
    if n_M == 0 or n_F == 0:
        agreed = 'M' if n_M > 0 else 'F'
        avg = round(float(np.mean(confs)), 2)
        if avg >= DEEPFACE_HIGH_CONF:
            log.info("    All agree: %s (avg %s%%) -> ACCEPTED", agreed, avg)
            return {
                'gender': agreed,
                'gender_raw': f'consensus_{agreed}_x{len(valid_results)}',
                'confidence': avg,
                'face_detected': True,
                'n_faces': 1,
                'classifier': 'deepface_consensus',
                'error': None,
                '_photo': best_photo,
            }, images_tried, best_img

    # === PHASE 3: Disagreement or low confidence -> VLM ===
    if n_M > 0 and n_F > 0:
        log.info(
            "    DISAGREE (%dM vs %dF) -> asking VLM...", n_M, n_F,
        )
    else:
        log.info(
            "    Low avg conf (%s%%) -> asking VLM...",
            round(float(np.mean(confs)), 1),
        )

    vlm_result = classify_vlm(best_img, model=vlm_model, ollama_url=ollama_url)

    if vlm_result.get('gender') is None:
        log.info("    VLM failed: %s", vlm_result.get('error'))
        majority = 'M' if n_M > n_F else 'F' if n_F > n_M else None
        if majority:
            log.info(
                "    Fallback majority: %s (%dM vs %dF)", majority, n_M, n_F,
            )
            return {
                'gender': majority,
                'gender_raw': f'majority_{n_M}M_{n_F}F_vlm_failed',
                'confidence': round(float(np.mean(confs)), 2),
                'face_detected': True,
                'n_faces': 1,
                'classifier': 'deepface_majority_vlm_failed',
                'error': f'vlm_failed_majority_{n_M}M_{n_F}F',
                '_photo': best_photo,
            }, images_tried, best_img
        best_raw['classifier'] = 'deepface_lowconf_vlm_failed'
        best_raw['_photo'] = best_photo
        return best_raw, images_tried, best_img

    vlm_gender = vlm_result['gender']
    majority = 'M' if n_M > n_F else 'F' if n_F > n_M else None

    if majority and vlm_gender == majority:
        log.info(
            "    VLM (%s) agrees with majority (%dM vs %dF) -> CONFIRMED",
            vlm_gender, n_M, n_F,
        )
        return {
            'gender': vlm_gender,
            'gender_raw': f'vlm={vlm_result.get("gender_raw")}, majority={n_M}M_{n_F}F',
            'confidence': 92.0,
            'face_detected': True,
            'n_faces': 1,
            'classifier': 'ensemble_vlm_majority_agree',
            'error': None,
            '_photo': best_photo,
        }, images_tried, best_img

    log.info(
        "    VLM=%s vs DeepFace=%dM/%dF -> trusting VLM",
        vlm_gender, n_M, n_F,
    )
    return {
        'gender': vlm_gender,
        'gender_raw': f'OVERRIDE: vlm={vlm_result.get("gender_raw")}, df={n_M}M_{n_F}F',
        'confidence': 85.0,
        'face_detected': True,
        'n_faces': 1,
        'classifier': 'ensemble_vlm_override',
        'error': f'vlm_override_{vlm_gender}_vs_{n_M}M_{n_F}F',
        '_photo': best_photo,
    }, images_tried, best_img
