"""
Unit tests for name_classifier.py.

Tests the new global-gender-predictor logic.
Checks probability thresholds and cross-cultural Italian names.
"""

import pytest
from genderphoto.name_classifier import classify_name


class TestClearGenders:
    """Names that should be unambiguously classified."""

    def test_james_us_male(self):
        r = classify_name("James", "US")
        assert r['gender'] == 'M'
        assert r['is_ambiguous'] is False
        assert r['name_probability'] > 0.9

    def test_jennifer_us_female(self):
        r = classify_name("Jennifer", "US")
        assert r['gender'] == 'F'
        assert r['is_ambiguous'] is False
        assert r['name_probability'] > 0.9

    def test_andrea_italy_male(self):
        """Andrea in Italy: override -> male (Italian male name in Italy)."""
        r = classify_name("Andrea", "IT")
        assert r['gender'] == 'M'
        assert r['is_ambiguous'] is False
        assert r['name_probability'] == 1.0


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

    def test_robin_us_ambiguous(self):
        """Robin in US -> likely low probability male/female -> ambiguous."""
        r = classify_name("Robin", "US", threshold=0.99) # high threshold guarantees ambiguity
        assert r['is_ambiguous'] is True

    def test_kim_gb_ambiguous(self):
        """Kim in GB -> likely ambiguous depending on threshold."""
        r = classify_name("Kim", "GB", threshold=0.99)
        assert r['is_ambiguous'] is True


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
        """Luca in Italy: global-gender-predictor says 'male', not cross-cultural."""
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
        # global-gender-predictor predicts female globally for Andrea
        # so without country, it should be F (or ambiguous if probability is below threshold)
        assert 'cross_cultural' not in str(r.get('ambiguity_reason', ''))

    def test_method_field(self):
        r = classify_name("James", "US")
        assert r['method'] == 'name_based'
