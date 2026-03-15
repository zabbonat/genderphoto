"""
Face-based gender classification using DeepFace with RetinaFace backend.

CRITICAL: enforce_detection=True is MANDATORY. With False, DeepFace
classifies image noise as gender with high confidence.
"""

from __future__ import annotations

import logging

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)


def classify_face(img: Image.Image) -> dict:
    """
    Classify gender from a photo using DeepFace.

    Handles:
    - Single face: return classification
    - Multi-face, same gender: accept (e.g., group of all men)
    - Multi-face, mixed genders: skip (ambiguous group photo)
    - No face detected: skip

    Parameters
    ----------
    img : PIL.Image.Image
        RGB image to analyze.

    Returns
    -------
    dict
        {
            'gender': 'M' | 'F' | None,
            'gender_raw': str,
            'confidence': float | None,
            'n_faces': int,
            'face_detected': bool,
            'classifier': str,
            'error': str | None,
        }
    """
    try:
        from deepface import DeepFace
        result = DeepFace.analyze(
            img_path=np.array(img),
            actions=['gender'],
            enforce_detection=True,
            detector_backend='retinaface',
            silent=True,
        )
        faces = result if isinstance(result, list) else [result]
        n_faces = len(faces)

        if n_faces == 0:
            return {
                'gender': None, 'confidence': None, 'face_detected': False,
                'n_faces': 0, 'classifier': 'deepface', 'error': 'no_face',
            }

        face_genders = []
        face_confs = []
        for f in faces:
            scores = f.get('gender', {})
            dominant = f.get('dominant_gender', None)
            conf = max(scores.values()) if scores else 0
            g = 'M' if dominant == 'Man' else 'F' if dominant == 'Woman' else None
            face_genders.append(g)
            face_confs.append(conf)

        unique = set(g for g in face_genders if g is not None)

        if n_faces == 1:
            return {
                'gender': face_genders[0],
                'gender_raw': faces[0].get('dominant_gender'),
                'confidence': round(face_confs[0], 2),
                'face_detected': True,
                'n_faces': 1,
                'classifier': 'deepface',
                'error': None,
            }
        elif len(unique) == 1:
            g = unique.pop()
            avg = round(float(np.mean(face_confs)), 2)
            log.info(
                "    Multi-face (%d): all %s, avg conf %s%%", n_faces, g, avg,
            )
            return {
                'gender': g,
                'gender_raw': f'all_{g}_x{n_faces}',
                'confidence': avg,
                'face_detected': True,
                'n_faces': n_faces,
                'classifier': 'deepface_multiface_agree',
                'error': None,
            }
        else:
            log.info(
                "    Multi-face (%d): mixed %s, skipping", n_faces, face_genders,
            )
            return {
                'gender': None,
                'gender_raw': f'mixed_{face_genders}',
                'confidence': None,
                'face_detected': True,
                'n_faces': n_faces,
                'classifier': 'deepface_multiface_mixed',
                'error': 'mixed_genders_in_group_photo',
            }

    except ValueError:
        return {
            'gender': None, 'confidence': None, 'face_detected': False,
            'n_faces': 0, 'classifier': 'deepface',
            'error': 'no_face_detected',
        }
    except Exception as e:
        return {
            'gender': None, 'confidence': None, 'face_detected': False,
            'n_faces': 0, 'classifier': 'deepface',
            'error': str(e)[:100],
        }
