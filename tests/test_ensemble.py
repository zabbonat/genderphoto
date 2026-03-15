"""
Unit tests for ensemble.py.

Uses mocked DeepFace and VLM results to test the consensus logic
without requiring network access or Ollama.
"""

import pytest
from unittest.mock import patch, MagicMock
from PIL import Image

from genderphoto.ensemble import run_ensemble


def _make_photo(url="fake.jpg"):
    """Create a fake photo dict."""
    return {'url': url, 'query': 'test', 'query_type': 'test', 'query_rank': 0}


def _make_img():
    """Create a small test image."""
    return Image.new('RGB', (100, 100), color='red')


class TestAllAgreeHighConfidence:
    """All DeepFace results agree with avg confidence >= 90%."""

    @patch('genderphoto.ensemble.classify_vlm')
    @patch('genderphoto.ensemble.classify_face')
    @patch('genderphoto.ensemble.load_image')
    def test_consensus_male(self, mock_load, mock_face, mock_vlm):
        mock_load.return_value = _make_img()
        mock_face.return_value = {
            'gender': 'M', 'confidence': 99.5, 'face_detected': True,
            'n_faces': 1, 'classifier': 'deepface', 'error': None,
        }
        photos = [_make_photo() for _ in range(3)]
        result, tried, best = run_ensemble(photos, max_images=3)

        assert result['gender'] == 'M'
        assert result['classifier'] == 'deepface_consensus'
        assert result['confidence'] >= 90.0
        assert tried == 3
        mock_vlm.assert_not_called()

    @patch('genderphoto.ensemble.classify_vlm')
    @patch('genderphoto.ensemble.classify_face')
    @patch('genderphoto.ensemble.load_image')
    def test_consensus_female(self, mock_load, mock_face, mock_vlm):
        mock_load.return_value = _make_img()
        mock_face.return_value = {
            'gender': 'F', 'confidence': 95.0, 'face_detected': True,
            'n_faces': 1, 'classifier': 'deepface', 'error': None,
        }
        photos = [_make_photo() for _ in range(4)]
        result, tried, best = run_ensemble(photos, max_images=4)

        assert result['gender'] == 'F'
        assert result['classifier'] == 'deepface_consensus'
        mock_vlm.assert_not_called()


class TestDisagreementVLMAgreesWithMajority:
    """DeepFace results disagree, VLM agrees with majority."""

    @patch('genderphoto.ensemble.classify_vlm')
    @patch('genderphoto.ensemble.classify_face')
    @patch('genderphoto.ensemble.load_image')
    def test_vlm_confirms_majority(self, mock_load, mock_face, mock_vlm):
        mock_load.return_value = _make_img()
        # 3 images: 2 male, 1 female
        face_results = [
            {'gender': 'M', 'confidence': 95.0, 'face_detected': True,
             'n_faces': 1, 'classifier': 'deepface', 'error': None},
            {'gender': 'M', 'confidence': 92.0, 'face_detected': True,
             'n_faces': 1, 'classifier': 'deepface', 'error': None},
            {'gender': 'F', 'confidence': 88.0, 'face_detected': True,
             'n_faces': 1, 'classifier': 'deepface', 'error': None},
        ]
        mock_face.side_effect = face_results
        mock_vlm.return_value = {
            'gender': 'M', 'gender_raw': 'male', 'confidence': 90.0,
            'face_detected': True, 'n_faces': 1,
            'classifier': 'vlm_qwen2.5vl:7b', 'error': None,
        }
        photos = [_make_photo() for _ in range(3)]
        result, tried, best = run_ensemble(photos, max_images=3)

        assert result['gender'] == 'M'
        assert result['classifier'] == 'ensemble_vlm_majority_agree'


class TestVLMOverride:
    """VLM disagrees with DeepFace -> trust VLM."""

    @patch('genderphoto.ensemble.classify_vlm')
    @patch('genderphoto.ensemble.classify_face')
    @patch('genderphoto.ensemble.load_image')
    def test_vlm_overrides_deepface(self, mock_load, mock_face, mock_vlm):
        mock_load.return_value = _make_img()
        # DeepFace says male (2x)
        mock_face.return_value = {
            'gender': 'M', 'confidence': 85.0, 'face_detected': True,
            'n_faces': 1, 'classifier': 'deepface', 'error': None,
        }
        # VLM says female
        mock_vlm.return_value = {
            'gender': 'F', 'gender_raw': 'female', 'confidence': 90.0,
            'face_detected': True, 'n_faces': 1,
            'classifier': 'vlm_qwen2.5vl:7b', 'error': None,
        }
        photos = [_make_photo() for _ in range(2)]
        result, tried, best = run_ensemble(photos, max_images=2)

        assert result['gender'] == 'F'
        assert result['classifier'] == 'ensemble_vlm_override'


class TestVLMFailureFallback:
    """VLM fails -> fallback to DeepFace majority."""

    @patch('genderphoto.ensemble.classify_vlm')
    @patch('genderphoto.ensemble.classify_face')
    @patch('genderphoto.ensemble.load_image')
    def test_vlm_fails_majority_fallback(self, mock_load, mock_face, mock_vlm):
        mock_load.return_value = _make_img()
        face_results = [
            {'gender': 'M', 'confidence': 80.0, 'face_detected': True,
             'n_faces': 1, 'classifier': 'deepface', 'error': None},
            {'gender': 'M', 'confidence': 82.0, 'face_detected': True,
             'n_faces': 1, 'classifier': 'deepface', 'error': None},
            {'gender': 'F', 'confidence': 75.0, 'face_detected': True,
             'n_faces': 1, 'classifier': 'deepface', 'error': None},
        ]
        mock_face.side_effect = face_results
        mock_vlm.return_value = {
            'gender': None, 'confidence': None, 'face_detected': False,
            'n_faces': 0, 'classifier': 'vlm_qwen2.5vl:7b',
            'error': 'OLLAMA_NOT_RUNNING',
        }
        photos = [_make_photo() for _ in range(3)]
        result, tried, best = run_ensemble(photos, max_images=3)

        assert result['gender'] == 'M'
        assert 'vlm_failed' in result['classifier']


class TestNoFacesFound:
    """All images fail face detection."""

    @patch('genderphoto.ensemble.classify_face')
    @patch('genderphoto.ensemble.load_image')
    def test_no_usable_faces(self, mock_load, mock_face):
        mock_load.return_value = _make_img()
        mock_face.return_value = {
            'gender': None, 'confidence': None, 'face_detected': False,
            'n_faces': 0, 'classifier': 'deepface', 'error': 'no_face_detected',
        }
        photos = [_make_photo() for _ in range(3)]
        result, tried, best = run_ensemble(photos, max_images=3)

        assert result['gender'] is None
        assert result['classifier'] == 'all_failed'
        assert best is None

    @patch('genderphoto.ensemble.load_image')
    def test_all_images_fail_to_load(self, mock_load):
        mock_load.return_value = None
        photos = [_make_photo() for _ in range(3)]
        result, tried, best = run_ensemble(photos, max_images=3)

        assert result['gender'] is None
        assert tried == 3
        assert best is None
