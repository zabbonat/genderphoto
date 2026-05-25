"""
Stage 1: Name-based gender classification using global-gender-predictor.

Uses WGND 2.0 (4.1M names) to predict gender probabilities.
Italian cross-cultural names outside Italy are flagged as ambiguous.
Names with a predicted probability below the threshold are flagged as ambiguous.
"""

from __future__ import annotations

import logging
import json
import os

from global_gender_predictor import GlobalGenderPredictor

from genderphoto.constants import ITALIAN_MALE_NAMES, DEFAULT_NAME_THRESHOLD
from genderphoto.utils import is_asian_name

log = logging.getLogger(__name__)

# Module-level predictor (created once)
_predictor = GlobalGenderPredictor()

_chinese_pinyin_dict = None

def _get_chinese_pinyin_dict() -> dict:
    global _chinese_pinyin_dict
    if _chinese_pinyin_dict is None:
        dict_path = os.path.join(os.path.dirname(__file__), 'data', 'chinese_pinyin.json')
        if os.path.exists(dict_path):
            with open(dict_path, 'r', encoding='utf-8') as f:
                _chinese_pinyin_dict = json.load(f)
        else:
            _chinese_pinyin_dict = {}
    return _chinese_pinyin_dict


def classify_name(
    first_name: str, 
    country_code: str = None, 
    threshold: float = DEFAULT_NAME_THRESHOLD,
    full_name: str = None
) -> dict:
    """
    Classify gender from a first name using global-gender-predictor.

    Parameters
    ----------
    first_name : str
        The first name to classify.
    country_code : str, optional
        ISO 2-letter country code of the person's country of residence.
    threshold : float
        Probability threshold below which the name is considered ambiguous.
    full_name : str, optional
        The full name (used for Asian surname heuristics).

    Returns
    -------
    dict
        {
            'gender': 'M' | 'F' | None,
            'gender_raw': str (e.g. 'Male', 'Female', 'Unknown'),
            'name_probability': float,
            'is_ambiguous': bool,
            'ambiguity_reason': str,
            'method': 'name_based'
        }
    """
    fn = first_name.strip()
    fn_lower = fn.lower()
    
    if threshold is None:
        threshold = DEFAULT_NAME_THRESHOLD
        
    # Check Chinese Pinyin Dictionary for unambiguous Asian names
    is_asian_country = country_code and country_code.upper() in ['CN', 'TW', 'HK', 'SG', 'KR', 'KP', 'VN']
    if is_asian_country or (full_name and is_asian_name(full_name)):
        pinyin_dict = _get_chinese_pinyin_dict()
        if fn_lower in pinyin_dict:
            return {
                'gender': pinyin_dict[fn_lower],
                'gender_raw': 'Male' if pinyin_dict[fn_lower] == 'M' else 'Female',
                'name_probability': 1.0,
                'is_ambiguous': False,
                'ambiguity_reason': 'unambiguous_chinese_pinyin',
                'method': 'name_based',
            }

    # Get probability from global-gender-predictor
    result_gender, weight = _predictor.predict_gender_probability(fn)

    # Override: Italian male names IN Italy are always male
    is_italian_in_italy = (
        fn_lower in ITALIAN_MALE_NAMES
        and country_code is not None
        and country_code.upper() == 'IT'
    )
    if is_italian_in_italy:
        return {
            'gender': 'M',
            'gender_raw': 'Male',
            'name_probability': 1.0,
            'is_ambiguous': False,
            'ambiguity_reason': f'italian_male_in_italy_{fn_lower}',
            'method': 'name_based',
        }

    # Cross-cultural check: Italian male names used outside Italy
    is_cross_cultural = (
        fn_lower in ITALIAN_MALE_NAMES
        and country_code is not None
        and country_code.upper() != 'IT'
    )

    is_ambiguous = (
        result_gender == 'Unknown'
        or weight < threshold
        or is_cross_cultural
    )

    reason = result_gender
    if is_cross_cultural:
        reason = f'{result_gender}_but_cross_cultural_{fn_lower}_in_{country_code}'
    elif result_gender != 'Unknown' and weight < threshold:
        reason = f'{result_gender}_low_probability_{weight:.2f}'

    # Map result to M/F/None
    gender = None
    if not is_ambiguous:
        if result_gender == 'Male':
            gender = 'M'
        elif result_gender == 'Female':
            gender = 'F'

    log.debug(
        "classify_name('%s', country=%s) -> %s (prob=%.2f, ambiguous=%s)",
        first_name, country_code, gender, weight, is_ambiguous,
    )

    return {
        'gender': gender,
        'gender_raw': result_gender,
        'name_probability': weight,
        'is_ambiguous': is_ambiguous,
        'ambiguity_reason': reason,
        'method': 'name_based',
    }
