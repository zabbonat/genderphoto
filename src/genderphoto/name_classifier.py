"""
Stage 1: Name-based gender classification using gender_guesser.

V6 CHANGE: mostly_male and mostly_female are flagged as ambiguous,
not just 'andy' and 'unknown'. Cross-cultural Italian names outside
Italy are also flagged as ambiguous.
"""

from __future__ import annotations

import logging

import gender_guesser.detector as gg

from genderphoto.constants import ITALIAN_MALE_NAMES

log = logging.getLogger(__name__)

# Module-level detector (created once)
_detector = gg.Detector()


def classify_name(first_name: str, country_code: str = None) -> dict:
    """
    Classify gender from a first name using gender_guesser.

    Parameters
    ----------
    first_name : str
        The first name to classify.
    country_code : str, optional
        ISO 2-letter country code of the person's country of residence.

    Returns
    -------
    dict
        {
            'gender': 'M' | 'F' | None,
            'gender_raw': str (gender_guesser output),
            'is_ambiguous': bool,
            'ambiguity_reason': str,
            'method': 'name_based'
        }
    """
    fn = first_name.strip()
    fn_lower = fn.lower()
    result = _detector.get_gender(fn)

    # Override: Italian male names IN Italy are always male
    # (gender_guesser misclassifies them as female globally)
    is_italian_in_italy = (
        fn_lower in ITALIAN_MALE_NAMES
        and country_code is not None
        and country_code.upper() == 'IT'
    )
    if is_italian_in_italy:
        return {
            'gender': 'M',
            'gender_raw': f'{result}_override_italian_male',
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

    # V6: mostly_male and mostly_female are also ambiguous
    is_ambiguous = (
        result in ('andy', 'unknown', 'mostly_male', 'mostly_female')
        or is_cross_cultural
    )

    reason = result
    if is_cross_cultural:
        reason = f'{result}_but_cross_cultural_{fn_lower}_in_{country_code}'

    # Map gender_guesser result to M/F/None
    gender = None
    if not is_ambiguous:
        if result == 'male':
            gender = 'M'
        elif result == 'female':
            gender = 'F'

    log.debug(
        "classify_name('%s', country=%s) -> %s (raw=%s, ambiguous=%s)",
        first_name, country_code, gender, result, is_ambiguous,
    )

    return {
        'gender': gender,
        'gender_raw': result,
        'is_ambiguous': is_ambiguous,
        'ambiguity_reason': reason,
        'method': 'name_based',
    }
