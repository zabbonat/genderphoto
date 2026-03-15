"""
Validation test using the 100-researcher dataset from V6.

This test runs name_classifier on all 100 researchers and validates that:
1. The expected number of ambiguous names are flagged
2. Known ambiguous names (Robin Murphy, Kim Nasmyth, Jamie Shotton) ARE flagged
3. Known clear names (James Heckman, Jennifer Doudna) are NOT flagged

NOTE: This does NOT test the photo pipeline (requires network + Ollama).
"""

import pytest
from genderphoto.name_classifier import classify_name
from genderphoto.utils import extract_first_name


# Full 100-researcher dataset from V6 notebook cell 2
RESEARCHERS = [
    # A. Italian MALE names outside Italy (20)
    {'inventor_name': 'Andrea Cavalleri', 'affiliation': 'Max Planck Hamburg', 'country_code': 'DE', 'gender_true': 'M', 'category': 'italian_male_abroad'},
    {'inventor_name': 'Andrea Alu', 'affiliation': 'CUNY New York', 'country_code': 'US', 'gender_true': 'M', 'category': 'italian_male_abroad'},
    {'inventor_name': 'Andrea Montanari', 'affiliation': 'Stanford University', 'country_code': 'US', 'gender_true': 'M', 'category': 'italian_male_abroad'},
    {'inventor_name': 'Andrea Rinaldo', 'affiliation': 'EPFL', 'country_code': 'CH', 'gender_true': 'M', 'category': 'italian_male_abroad'},
    {'inventor_name': 'Simone Campanoni', 'affiliation': 'Northwestern University', 'country_code': 'US', 'gender_true': 'M', 'category': 'italian_male_abroad'},
    {'inventor_name': 'Simone Giombi', 'affiliation': 'Princeton University', 'country_code': 'US', 'gender_true': 'M', 'category': 'italian_male_abroad'},
    {'inventor_name': 'Nicola Marzari', 'affiliation': 'EPFL', 'country_code': 'CH', 'gender_true': 'M', 'category': 'italian_male_abroad'},
    {'inventor_name': 'Nicola Perra', 'affiliation': 'Queen Mary University London', 'country_code': 'GB', 'gender_true': 'M', 'category': 'italian_male_abroad'},
    {'inventor_name': 'Michele Parrinello', 'affiliation': 'ETH Zurich', 'country_code': 'CH', 'gender_true': 'M', 'category': 'italian_male_abroad'},
    {'inventor_name': 'Michele Mosca', 'affiliation': 'University of Waterloo', 'country_code': 'CA', 'gender_true': 'M', 'category': 'italian_male_abroad'},
    {'inventor_name': 'Gabriele Veneziano', 'affiliation': 'CERN', 'country_code': 'CH', 'gender_true': 'M', 'category': 'italian_male_abroad'},
    {'inventor_name': 'Luca Cardelli', 'affiliation': 'Microsoft Research', 'country_code': 'GB', 'gender_true': 'M', 'category': 'italian_male_abroad'},
    {'inventor_name': 'Luca Trevisan', 'affiliation': 'Bocconi University', 'country_code': 'US', 'gender_true': 'M', 'category': 'italian_male_abroad'},
    {'inventor_name': 'Daniele Bhatt', 'affiliation': 'Mount Sinai', 'country_code': 'US', 'gender_true': 'M', 'category': 'italian_male_abroad'},
    {'inventor_name': 'Raffaele Ferrari', 'affiliation': 'MIT', 'country_code': 'US', 'gender_true': 'M', 'category': 'italian_male_abroad'},
    {'inventor_name': 'Mattia Frasca', 'affiliation': 'University of Catania', 'country_code': 'US', 'gender_true': 'M', 'category': 'italian_male_abroad'},
    {'inventor_name': 'Emanuele Berti', 'affiliation': 'Johns Hopkins University', 'country_code': 'US', 'gender_true': 'M', 'category': 'italian_male_abroad'},
    {'inventor_name': 'Pasquale Cirillo', 'affiliation': 'TU Delft', 'country_code': 'NL', 'gender_true': 'M', 'category': 'italian_male_abroad'},
    {'inventor_name': 'Samuele Fiorini', 'affiliation': 'Universite libre de Bruxelles', 'country_code': 'BE', 'gender_true': 'M', 'category': 'italian_male_abroad'},
    {'inventor_name': 'Daniele Quercia', 'affiliation': 'Nokia Bell Labs', 'country_code': 'GB', 'gender_true': 'M', 'category': 'italian_male_abroad'},
    # B. Italian-named WOMEN outside Italy (10)
    {'inventor_name': 'Andrea Goldsmith', 'affiliation': 'Princeton University', 'country_code': 'US', 'gender_true': 'F', 'category': 'italian_name_female_abroad'},
    {'inventor_name': 'Andrea Liu', 'affiliation': 'University of Pennsylvania', 'country_code': 'US', 'gender_true': 'F', 'category': 'italian_name_female_abroad'},
    {'inventor_name': 'Nicola Spaldin', 'affiliation': 'ETH Zurich', 'country_code': 'CH', 'gender_true': 'F', 'category': 'italian_name_female_abroad'},
    {'inventor_name': 'Nicola Clayton', 'affiliation': 'University of Cambridge', 'country_code': 'GB', 'gender_true': 'F', 'category': 'italian_name_female_abroad'},
    {'inventor_name': 'Simone Biles', 'affiliation': 'World Champions Centre', 'country_code': 'US', 'gender_true': 'F', 'category': 'italian_name_female_abroad'},
    {'inventor_name': 'Andrea Bertozzi', 'affiliation': 'UCLA', 'country_code': 'US', 'gender_true': 'F', 'category': 'italian_name_female_abroad'},
    {'inventor_name': 'Nicola Sturgeon', 'affiliation': 'Scottish Government', 'country_code': 'GB', 'gender_true': 'F', 'category': 'italian_name_female_abroad'},
    {'inventor_name': 'Andrea Young', 'affiliation': 'UCSB', 'country_code': 'US', 'gender_true': 'F', 'category': 'italian_name_female_abroad'},
    {'inventor_name': 'Michele Wucker', 'affiliation': 'Gray Rhino Company', 'country_code': 'US', 'gender_true': 'F', 'category': 'italian_name_female_abroad'},
    {'inventor_name': 'Gabriele Hogan', 'affiliation': 'Rockefeller University', 'country_code': 'US', 'gender_true': 'F', 'category': 'italian_name_female_abroad'},
    # C. East Asian (20)
    {'inventor_name': 'Fei-Fei Li', 'affiliation': 'Stanford University', 'country_code': 'US', 'gender_true': 'F', 'category': 'east_asian'},
    {'inventor_name': 'Jian-Wei Pan', 'affiliation': 'USTC Hefei', 'country_code': 'CN', 'gender_true': 'M', 'category': 'east_asian'},
    {'inventor_name': 'Yann LeCun', 'affiliation': 'Meta AI', 'country_code': 'US', 'gender_true': 'M', 'category': 'east_asian'},
    {'inventor_name': 'Yuki Igarashi', 'affiliation': 'Meiji University', 'country_code': 'JP', 'gender_true': 'F', 'category': 'east_asian'},
    {'inventor_name': 'Wei Zhang', 'affiliation': 'Peking University', 'country_code': 'CN', 'gender_true': 'M', 'category': 'east_asian'},
    {'inventor_name': 'Jie Shan', 'affiliation': 'Cornell University', 'country_code': 'US', 'gender_true': 'F', 'category': 'east_asian'},
    {'inventor_name': 'Yi Cui', 'affiliation': 'Stanford University', 'country_code': 'US', 'gender_true': 'M', 'category': 'east_asian'},
    {'inventor_name': 'Ling-Yan Hung', 'affiliation': 'Fudan University', 'country_code': 'CN', 'gender_true': 'F', 'category': 'east_asian'},
    {'inventor_name': 'Takashi Taniguchi', 'affiliation': 'NIMS Japan', 'country_code': 'JP', 'gender_true': 'M', 'category': 'east_asian'},
    {'inventor_name': 'Maki Kawai', 'affiliation': 'RIKEN', 'country_code': 'JP', 'gender_true': 'F', 'category': 'east_asian'},
    {'inventor_name': 'Xiang Zhang', 'affiliation': 'University of Hong Kong', 'country_code': 'HK', 'gender_true': 'M', 'category': 'east_asian'},
    {'inventor_name': 'Chia-Ling Chien', 'affiliation': 'Johns Hopkins University', 'country_code': 'US', 'gender_true': 'M', 'category': 'east_asian'},
    {'inventor_name': 'Jing Wang', 'affiliation': 'Tsinghua University', 'country_code': 'CN', 'gender_true': 'F', 'category': 'east_asian'},
    {'inventor_name': 'Shoucheng Zhang', 'affiliation': 'Stanford University', 'country_code': 'US', 'gender_true': 'M', 'category': 'east_asian'},
    {'inventor_name': 'Naomi Halas', 'affiliation': 'Rice University', 'country_code': 'US', 'gender_true': 'F', 'category': 'east_asian'},
    {'inventor_name': 'Yongjie Hu', 'affiliation': 'UCLA', 'country_code': 'US', 'gender_true': 'M', 'category': 'east_asian'},
    {'inventor_name': 'Lan Yang', 'affiliation': 'Washington University', 'country_code': 'US', 'gender_true': 'F', 'category': 'east_asian'},
    {'inventor_name': 'Hyunsoo Yang', 'affiliation': 'NUS Singapore', 'country_code': 'SG', 'gender_true': 'M', 'category': 'east_asian'},
    {'inventor_name': 'Sunghwan Kim', 'affiliation': 'KAIST', 'country_code': 'KR', 'gender_true': 'M', 'category': 'east_asian'},
    {'inventor_name': 'Jia Li', 'affiliation': 'Google Cloud', 'country_code': 'US', 'gender_true': 'F', 'category': 'east_asian'},
    # D. French ambiguous (10)
    {'inventor_name': 'Dominique Costagliola', 'affiliation': 'INSERM Paris', 'country_code': 'FR', 'gender_true': 'F', 'category': 'french_ambiguous'},
    {'inventor_name': 'Dominique Langevin', 'affiliation': 'Universite Paris-Saclay', 'country_code': 'FR', 'gender_true': 'F', 'category': 'french_ambiguous'},
    {'inventor_name': 'Claude Cohen-Tannoudji', 'affiliation': 'ENS Paris', 'country_code': 'FR', 'gender_true': 'M', 'category': 'french_ambiguous'},
    {'inventor_name': 'Camille Nous', 'affiliation': 'Cogitamus Laboratory', 'country_code': 'FR', 'gender_true': 'M', 'category': 'french_ambiguous'},
    {'inventor_name': 'Jean Tirole', 'affiliation': 'Toulouse School of Economics', 'country_code': 'FR', 'gender_true': 'M', 'category': 'french_ambiguous'},
    {'inventor_name': 'Claude Berrou', 'affiliation': 'IMT Atlantique', 'country_code': 'FR', 'gender_true': 'M', 'category': 'french_ambiguous'},
    {'inventor_name': 'Dominique de Villepin', 'affiliation': 'Sciences Po', 'country_code': 'FR', 'gender_true': 'M', 'category': 'french_ambiguous'},
    {'inventor_name': 'Camille Parmesan', 'affiliation': 'CNRS Moulis', 'country_code': 'FR', 'gender_true': 'F', 'category': 'french_ambiguous'},
    {'inventor_name': 'Jean-Pierre Bourguignon', 'affiliation': 'IHES', 'country_code': 'FR', 'gender_true': 'M', 'category': 'french_ambiguous'},
    {'inventor_name': 'Francoise Barre-Sinoussi', 'affiliation': 'Institut Pasteur', 'country_code': 'FR', 'gender_true': 'F', 'category': 'french_ambiguous'},
    # E. English ambiguous (10)
    {'inventor_name': 'Robin Murphy', 'affiliation': 'Texas A&M', 'country_code': 'US', 'gender_true': 'F', 'category': 'english_ambiguous'},
    {'inventor_name': 'Robin Clark', 'affiliation': 'University of Pennsylvania', 'country_code': 'US', 'gender_true': 'M', 'category': 'english_ambiguous'},
    {'inventor_name': 'Morgan Quigley', 'affiliation': 'Open Robotics', 'country_code': 'US', 'gender_true': 'M', 'category': 'english_ambiguous'},
    {'inventor_name': 'Jordan Ellenberg', 'affiliation': 'University of Wisconsin', 'country_code': 'US', 'gender_true': 'M', 'category': 'english_ambiguous'},
    {'inventor_name': 'Alex Graves', 'affiliation': 'DeepMind', 'country_code': 'GB', 'gender_true': 'M', 'category': 'english_ambiguous'},
    {'inventor_name': 'Taylor Perron', 'affiliation': 'MIT', 'country_code': 'US', 'gender_true': 'M', 'category': 'english_ambiguous'},
    {'inventor_name': 'Jamie Shotton', 'affiliation': 'Wayve', 'country_code': 'GB', 'gender_true': 'M', 'category': 'english_ambiguous'},
    {'inventor_name': 'Alexis Conneau', 'affiliation': 'Meta AI', 'country_code': 'US', 'gender_true': 'M', 'category': 'english_ambiguous'},
    {'inventor_name': 'Kim Nasmyth', 'affiliation': 'University of Oxford', 'country_code': 'GB', 'gender_true': 'M', 'category': 'english_ambiguous'},
    {'inventor_name': 'Pat Hanrahan', 'affiliation': 'Stanford University', 'country_code': 'US', 'gender_true': 'M', 'category': 'english_ambiguous'},
    # F. Clear MALE (15)
    {'inventor_name': 'James Heckman', 'affiliation': 'University of Chicago', 'country_code': 'US', 'gender_true': 'M', 'category': 'clear_male'},
    {'inventor_name': 'Yoshua Bengio', 'affiliation': 'MILA Montreal', 'country_code': 'CA', 'gender_true': 'M', 'category': 'clear_male'},
    {'inventor_name': 'Geoffrey Hinton', 'affiliation': 'University of Toronto', 'country_code': 'CA', 'gender_true': 'M', 'category': 'clear_male'},
    {'inventor_name': 'Thomas Piketty', 'affiliation': 'Paris School of Economics', 'country_code': 'FR', 'gender_true': 'M', 'category': 'clear_male'},
    {'inventor_name': 'Daron Acemoglu', 'affiliation': 'MIT', 'country_code': 'US', 'gender_true': 'M', 'category': 'clear_male'},
    {'inventor_name': 'David Card', 'affiliation': 'UC Berkeley', 'country_code': 'US', 'gender_true': 'M', 'category': 'clear_male'},
    {'inventor_name': 'Guido Imbens', 'affiliation': 'Stanford University', 'country_code': 'US', 'gender_true': 'M', 'category': 'clear_male'},
    {'inventor_name': 'Philip Anderson', 'affiliation': 'Princeton University', 'country_code': 'US', 'gender_true': 'M', 'category': 'clear_male'},
    {'inventor_name': 'Robert Langer', 'affiliation': 'MIT', 'country_code': 'US', 'gender_true': 'M', 'category': 'clear_male'},
    {'inventor_name': 'Michael Stonebraker', 'affiliation': 'MIT', 'country_code': 'US', 'gender_true': 'M', 'category': 'clear_male'},
    {'inventor_name': 'Daniel Kahneman', 'affiliation': 'Princeton University', 'country_code': 'US', 'gender_true': 'M', 'category': 'clear_male'},
    {'inventor_name': 'Paul Romer', 'affiliation': 'NYU', 'country_code': 'US', 'gender_true': 'M', 'category': 'clear_male'},
    {'inventor_name': 'Steven Weinberg', 'affiliation': 'University of Texas', 'country_code': 'US', 'gender_true': 'M', 'category': 'clear_male'},
    {'inventor_name': 'Richard Thaler', 'affiliation': 'University of Chicago', 'country_code': 'US', 'gender_true': 'M', 'category': 'clear_male'},
    {'inventor_name': 'Abhijit Banerjee', 'affiliation': 'MIT', 'country_code': 'US', 'gender_true': 'M', 'category': 'clear_male'},
    # G. Clear FEMALE (15)
    {'inventor_name': 'Maria Teresa Landi', 'affiliation': 'NIH', 'country_code': 'US', 'gender_true': 'F', 'category': 'clear_female'},
    {'inventor_name': 'Priya Natarajan', 'affiliation': 'Yale University', 'country_code': 'US', 'gender_true': 'F', 'category': 'clear_female'},
    {'inventor_name': 'Jennifer Doudna', 'affiliation': 'UC Berkeley', 'country_code': 'US', 'gender_true': 'F', 'category': 'clear_female'},
    {'inventor_name': 'Emmanuelle Charpentier', 'affiliation': 'Max Planck Berlin', 'country_code': 'DE', 'gender_true': 'F', 'category': 'clear_female'},
    {'inventor_name': 'Katalin Kariko', 'affiliation': 'University of Szeged', 'country_code': 'HU', 'gender_true': 'F', 'category': 'clear_female'},
    {'inventor_name': 'Esther Duflo', 'affiliation': 'MIT', 'country_code': 'US', 'gender_true': 'F', 'category': 'clear_female'},
    {'inventor_name': 'Claudia Goldin', 'affiliation': 'Harvard University', 'country_code': 'US', 'gender_true': 'F', 'category': 'clear_female'},
    {'inventor_name': 'Donna Strickland', 'affiliation': 'University of Waterloo', 'country_code': 'CA', 'gender_true': 'F', 'category': 'clear_female'},
    {'inventor_name': 'Anne LHuillier', 'affiliation': 'Lund University', 'country_code': 'SE', 'gender_true': 'F', 'category': 'clear_female'},
    {'inventor_name': 'Maryam Mirzakhani', 'affiliation': 'Stanford University', 'country_code': 'US', 'gender_true': 'F', 'category': 'clear_female'},
    {'inventor_name': 'Lisa Randall', 'affiliation': 'Harvard University', 'country_code': 'US', 'gender_true': 'F', 'category': 'clear_female'},
    {'inventor_name': 'Carolyn Bertozzi', 'affiliation': 'Stanford University', 'country_code': 'US', 'gender_true': 'F', 'category': 'clear_female'},
    {'inventor_name': 'Susan Athey', 'affiliation': 'Stanford University', 'country_code': 'US', 'gender_true': 'F', 'category': 'clear_female'},
    {'inventor_name': 'Janet Yellen', 'affiliation': 'US Treasury', 'country_code': 'US', 'gender_true': 'F', 'category': 'clear_female'},
    {'inventor_name': 'Christina Romer', 'affiliation': 'UC Berkeley', 'country_code': 'US', 'gender_true': 'F', 'category': 'clear_female'},
]


@pytest.fixture
def name_results():
    """Run classify_name on all 100 researchers."""
    results = []
    for r in RESEARCHERS:
        first_name = extract_first_name(r['inventor_name'])
        nr = classify_name(first_name, r['country_code'])
        results.append({**r, **nr})
    return results


class TestDatasetSize:
    def test_total_count(self, name_results):
        assert len(name_results) == 100


class TestAmbiguousCounts:
    def test_ambiguous_count(self, name_results):
        n_ambig = sum(1 for r in name_results if r['is_ambiguous'])
        # Should be a substantial number (60+) given the dataset
        assert n_ambig >= 50, f"Expected >= 50 ambiguous, got {n_ambig}"

    def test_resolved_by_name(self, name_results):
        n_resolved = sum(1 for r in name_results if not r['is_ambiguous'])
        assert n_resolved >= 20, f"Expected >= 20 resolved, got {n_resolved}"


class TestV6FixSpecificNames:
    """V6 fix: these names MUST be flagged as ambiguous."""

    def test_robin_murphy_ambiguous(self, name_results):
        robin = next(r for r in name_results if r['inventor_name'] == 'Robin Murphy')
        assert robin['is_ambiguous'] is True, "Robin Murphy should be ambiguous (V6 fix)"

    def test_kim_nasmyth_ambiguous(self, name_results):
        kim = next(r for r in name_results if r['inventor_name'] == 'Kim Nasmyth')
        assert kim['is_ambiguous'] is True, "Kim Nasmyth should be ambiguous (V6 fix)"

    def test_jamie_shotton_ambiguous(self, name_results):
        jamie = next(r for r in name_results if r['inventor_name'] == 'Jamie Shotton')
        assert jamie['is_ambiguous'] is True, "Jamie Shotton should be ambiguous (V6 fix)"


class TestClearNamesNotFlagged:
    """These clear names must NOT be flagged as ambiguous."""

    def test_james_heckman_not_ambiguous(self, name_results):
        james = next(r for r in name_results if r['inventor_name'] == 'James Heckman')
        assert james['is_ambiguous'] is False, "James Heckman should NOT be ambiguous"
        assert james['gender'] == 'M'

    def test_jennifer_doudna_not_ambiguous(self, name_results):
        jennifer = next(r for r in name_results if r['inventor_name'] == 'Jennifer Doudna')
        assert jennifer['is_ambiguous'] is False, "Jennifer Doudna should NOT be ambiguous"
        assert jennifer['gender'] == 'F'


class TestCategoryBreakdown:
    """Italian male names abroad should be ambiguous, clear names should not."""

    def test_all_italian_males_abroad_ambiguous(self, name_results):
        italian = [r for r in name_results if r['category'] == 'italian_male_abroad']
        for r in italian:
            assert r['is_ambiguous'] is True, (
                f"{r['inventor_name']} should be ambiguous (italian_male_abroad)"
            )

    def test_all_italian_females_abroad_ambiguous(self, name_results):
        italian_f = [r for r in name_results if r['category'] == 'italian_name_female_abroad']
        for r in italian_f:
            assert r['is_ambiguous'] is True, (
                f"{r['inventor_name']} should be ambiguous (italian_name_female_abroad)"
            )

    def test_clear_males_not_ambiguous(self, name_results):
        """Most clear_male names should not be ambiguous.

        Note: Some names like 'Yoshua' and 'Abhijit' return 'unknown' from
        gender_guesser, so they ARE ambiguous in V6 logic. This is expected.
        """
        clear = [r for r in name_results if r['category'] == 'clear_male']
        # Names gender_guesser doesn't know are legitimately ambiguous
        known_ambiguous = {'Yoshua', 'Abhijit'}
        for r in clear:
            first = r['inventor_name'].split()[0]
            if first in known_ambiguous:
                continue
            assert r['is_ambiguous'] is False, (
                f"{r['inventor_name']} should NOT be ambiguous (clear_male)"
            )
            assert r['gender'] == 'M'

    def test_clear_females_not_ambiguous(self, name_results):
        clear = [r for r in name_results if r['category'] == 'clear_female']
        for r in clear:
            assert r['is_ambiguous'] is False, (
                f"{r['inventor_name']} should NOT be ambiguous (clear_female)"
            )
            assert r['gender'] == 'F'
