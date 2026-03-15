"""
Unit tests for name_classifier.py.

Tests the V6 logic where mostly_male and mostly_female are flagged
as ambiguous, and cross-cultural Italian names are detected.
"""

import pytest
from genderphoto.name_classifier import classify_name


class TestClearGenders:
    """Names that should be unambiguously classified."""

    def test_james_us_male(self):
        r = classify_name("James", "US")
        assert r['gender'] == 'M'
        assert r['is_ambiguous'] is False

    def test_jennifer_us_female(self):
        r = classify_name("Jennifer", "US")
        assert r['gender'] == 'F'
        assert r['is_ambiguous'] is False

    def test_andrea_italy_male(self):
        """Andrea in Italy: override -> male (Italian male name in Italy)."""
        r = classify_name("Andrea", "IT")
        assert r['gender'] == 'M'
        assert r['is_ambiguous'] is False


class TestAmbiguousNames:
    """Names that should be flagged as ambiguous."""

    def test_andrea_us_cross_cultural(self):
        """Andrea outside Italy -> ambiguous (cross-cultural)."""
        r = classify_name("Andrea", "US")
        assert r['is_ambiguous'] is True
        assert 'cross_cultural' in r['ambiguity_reason']

    def test_wei_cn_andy(self):
        """Wei in China -> andy/unknown, ambiguous."""
        r = classify_name("Wei", "CN")
        assert r['is_ambiguous'] is True

    def test_robin_us_mostly_male_v6(self):
        """Robin in US -> mostly_male, ambiguous (V6 fix!)."""
        r = classify_name("Robin", "US")
        assert r['is_ambiguous'] is True
        assert r['gender_raw'] == 'mostly_male'

    def test_kim_gb_mostly_female_v6(self):
        """Kim in GB -> mostly_female, ambiguous (V6 fix!)."""
        r = classify_name("Kim", "GB")
        assert r['is_ambiguous'] is True
        assert r['gender_raw'] == 'mostly_female'


class TestItalianNamesAbroad:
    """Italian male names used outside Italy should be flagged."""

    @pytest.mark.parametrize("name,country", [
        ("Simone", "US"),
        ("Nicola", "GB"),
        ("Michele", "CH"),
        ("Gabriele", "US"),
        ("Luca", "DE"),
        ("Daniele", "NL"),
    ])
    def test_italian_male_names_abroad(self, name, country):
        r = classify_name(name, country)
        assert r['is_ambiguous'] is True
        assert 'cross_cultural' in r['ambiguity_reason']

    def test_luca_in_italy(self):
        """Luca in Italy: gender_guesser says 'male', not cross-cultural."""
        r = classify_name("Luca", "IT")
        assert r['is_ambiguous'] is False
        assert r['gender'] == 'M'

    @pytest.mark.parametrize("name", [
        "Simone", "Nicola", "Michele", "Gabriele",
    ])
    def test_italian_names_in_italy_override_male(self, name):
        """In Italy -> override to male (Italian male name in Italy)."""
        r = classify_name(name, "IT")
        assert r['is_ambiguous'] is False
        assert r['gender'] == 'M'


class TestEdgeCases:
    """Edge cases and clean input handling."""

    def test_whitespace_handling(self):
        r = classify_name("  James  ", "US")
        assert r['gender'] == 'M'

    def test_no_country(self):
        """Without country code, no cross-cultural flagging."""
        r = classify_name("Andrea")
        # gender_guesser returns 'female' for Andrea without country
        # not in the ambiguous set -> classified as female
        assert r['gender'] == 'F'
        assert r['is_ambiguous'] is False

    def test_method_field(self):
        r = classify_name("James", "US")
        assert r['method'] == 'name_based'
