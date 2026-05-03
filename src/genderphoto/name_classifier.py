"""
Stage 1: Name-based gender classification using global-gender-predictor.

Uses WGND 2.0 (4.1M names) to predict gender probabilities.
Italian cross-cultural names outside Italy are flagged as ambiguous.
Names with a predicted probability below the threshold are flagged as ambiguous.
"""

from __future__ import annotations

import logging

from global_gender_predictor import GlobalGenderPredictor

from genderphoto.constants import ITALIAN_MALE_NAMES, DEFAULT_NAME_THRESHOLD

log = logging.getLogger(__name__)

# Module-level predictor (created once)
_predictor = GlobalGenderPredictor()


def classify_name(
    first_name: str, 
    country_code: str = None, 
    threshold: float = DEFAULT_NAME_THRESHOLD
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
