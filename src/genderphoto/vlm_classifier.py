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
    name: str = None,
    affiliation: str = None,
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
    name : str, optional
        The person's name for context.
    affiliation : str, optional
        The person's affiliation for context.

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

        prompt = (
            'What is the gender of the main person in this photo? '
            'Answer with ONLY one word: male or female. '
            'IMPORTANT: Base your decision STRICTLY on the visual features of the person in the image. '
            'Do NOT guess based on the cultural or gender associations of the name.'
        )
        if name:
            ctx = f"This is a photo of {name}"
            if affiliation:
                ctx += f" from {affiliation}"
            prompt = f"{ctx}. {prompt}"

        resp = requests.post(
            ollama_url,
            json={
                'model': model,
                'prompt': prompt,
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


def list_available_vlm_models(ollama_url: str = OLLAMA_URL) -> list[dict]:
    """
    Query Ollama for installed models and identify vision-capable ones.

    Parameters
    ----------
    ollama_url : str
        Ollama API endpoint URL. The '/api/tags' endpoint is used.

    Returns
    -------
    list[dict]
        Each dict: {'name': str, 'size_gb': float, 'is_vision': bool}.
        Vision models are sorted first, then alphabetically.
        Returns empty list if Ollama is not running.

    Example
    -------
    >>> from genderphoto.vlm_classifier import list_available_vlm_models
    >>> models = list_available_vlm_models()
    >>> vision = [m for m in models if m['is_vision']]
    >>> print(vision)
    [{'name': 'qwen2.5vl:7b', 'size_gb': 4.7, 'is_vision': True}]
    """
    from genderphoto.constants import VISION_MODEL_PATTERNS

    # Derive the base URL from the generate endpoint
    base_url = ollama_url.replace('/api/generate', '')
    tags_url = f"{base_url}/api/tags"

    try:
        resp = requests.get(tags_url, timeout=5)
        resp.raise_for_status()
        models = resp.json().get('models', [])
    except requests.ConnectionError:
        log.warning("Ollama not running at %s", base_url)
        return []
    except Exception as e:
        log.warning("Error querying Ollama models: %s", e)
        return []

    result = []
    for m in models:
        name = m.get('name', '')
        name_lower = name.lower()
        is_vision = any(p in name_lower for p in VISION_MODEL_PATTERNS)
        size_bytes = m.get('size', 0)
        size_gb = round(size_bytes / (1024**3), 1) if size_bytes else None

        result.append({
            'name': name,
            'size_gb': size_gb,
            'is_vision': is_vision,
        })

    # Vision models first, then alphabetically
    result.sort(key=lambda x: (not x['is_vision'], x['name']))
    return result
