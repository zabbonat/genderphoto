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

import gender_guesser.detector

from global_gender_predictor import GlobalGenderPredictor

from genderphoto.constants import ISO_TO_GENDER_GUESSER, DEFAULT_NAME_THRESHOLD
from genderphoto.utils import is_asian_name

log = logging.getLogger(__name__)

# Module-level predictors (created once)
_predictor = GlobalGenderPredictor()
_gg_detector = gender_guesser.detector.Detector()

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

    # Get global probability from global-gender-predictor
    result_gender, weight = _predictor.predict_gender_probability(fn)

    # Use gender-guesser to determine if the name is strictly cross-cultural worldwide
    # A name is cross-cultural if it has both a male and female context in the WGND dataset.
    name_cap = fn_lower.capitalize()
    gg_record = _gg_detector.names.get(name_cap)
    
    is_globally_cross_cultural = False
    if gg_record:
        has_male = 'male' in gg_record or 'mostly_male' in gg_record
        has_female = 'female' in gg_record or 'mostly_female' in gg_record
        is_globally_cross_cultural = has_male and has_female

    is_ambiguous = False
    reason = result_gender

    if is_globally_cross_cultural:
        mapped_country = None
        if country_code:
            mapped_country = ISO_TO_GENDER_GUESSER.get(country_code.upper())
            
        if not mapped_country:
            # Cross-cultural name but country is unknown or unsupported
            is_ambiguous = True
            reason = f'cross_cultural_name_{fn_lower}_country_unknown'
        else:
            # We know the country, let's ask gender-guesser for this specific country
            local_gender = _gg_detector.get_gender(name_cap, mapped_country)
            if local_gender in ['male', 'female']:
                result_gender = 'Male' if local_gender == 'male' else 'Female'
                weight = 1.0  # We are confident for this specific country
            else:
                # Still ambiguous even in this country (e.g., 'mostly_male', 'unknown', 'andy')
                is_ambiguous = True
                reason = f'cross_cultural_name_{fn_lower}_ambiguous_in_{mapped_country}'
    else:
        # Not cross-cultural globally, rely on the global probability threshold
        if result_gender == 'Unknown' or weight < threshold:
            is_ambiguous = True
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
