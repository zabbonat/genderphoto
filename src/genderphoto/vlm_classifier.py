"""
Vision-language model (VLM) gender classifier using Ollama API.

Uses Qwen2.5-VL run locally via Ollama. No cloud API calls.
Images are resized to max 800px before sending to save memory/time.
"""

from __future__ import annotations

import base64
import logging
from io import BytesIO

import requests
from PIL import Image

from genderphoto.constants import DEFAULT_VLM, OLLAMA_URL

log = logging.getLogger(__name__)


def classify_vlm(
    img: Image.Image,
    model: str = DEFAULT_VLM,
    ollama_url: str = OLLAMA_URL,
) -> dict:
    """
    Classify gender using a vision-language model via Ollama.

    Parameters
    ----------
    img : PIL.Image.Image
        RGB image to analyze.
    model : str
        Ollama model name (default: qwen2.5vl:7b).
    ollama_url : str
        Ollama API endpoint URL.

    Returns
    -------
    dict
        {
            'gender': 'M' | 'F' | None,
            'gender_raw': str,
            'confidence': float | None,
            'face_detected': bool,
            'n_faces': int,
            'classifier': str,
            'error': str | None,
        }
    """
    try:
        # Resize to max 800px to save memory/time
        img_r = img.copy()
        if max(img_r.size) > 800:
            img_r.thumbnail((800, 800), Image.LANCZOS)

        buf = BytesIO()
        img_r.save(buf, format='JPEG', quality=85)
        img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        resp = requests.post(
            ollama_url,
            json={
                'model': model,
                'prompt': (
                    'What is the gender of the main person in this photo? '
                    'Answer with ONLY one word: male or female'
                ),
                'images': [img_b64],
                'stream': False,
            },
            timeout=120,
        )
        resp.raise_for_status()

        answer = resp.json().get('response', '').strip().lower()
        log.info("    VLM raw answer: '%s'", answer)

        import re
        if 'female' in answer or re.search(r'\bwoman\b', answer):
            gender = 'F'
        elif re.search(r'\bmale\b', answer) or re.search(r'\bman\b', answer):
            gender = 'M'
        else:
            return {
                'gender': None,
                'gender_raw': answer,
                'confidence': None,
                'face_detected': False,
                'n_faces': 0,
                'classifier': f'vlm_{model}',
                'error': f'ambiguous: {answer[:80]}',
            }

        return {
            'gender': gender,
            'gender_raw': answer,
            'confidence': 90.0,
            'face_detected': True,
            'n_faces': 1,
            'classifier': f'vlm_{model}',
            'error': None,
        }

    except requests.ConnectionError:
        log.error("    Ollama not running! Start: ollama serve")
        return {
            'gender': None, 'confidence': None, 'face_detected': False,
            'n_faces': 0, 'classifier': f'vlm_{model}',
            'error': 'OLLAMA_NOT_RUNNING',
        }
    except Exception as e:
        log.warning("    VLM error: %s", e)
        return {
            'gender': None, 'confidence': None, 'face_detected': False,
            'n_faces': 0, 'classifier': f'vlm_{model}',
            'error': str(e)[:100],
        }
