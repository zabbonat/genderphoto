"""
Constants and default configuration for genderphoto.
"""

# Italian first names that are male in Italy but often female elsewhere
ITALIAN_MALE_NAMES = {
    'andrea', 'simone', 'nicola', 'gabriele', 'michele', 'daniele',
    'raffaele', 'samuele', 'emanuele', 'pasquale', 'luca', 'mattia',
}

# DeepFace confidence threshold for accepting consensus without VLM
DEEPFACE_HIGH_CONF = 90.0

# Ollama API endpoint (local, no cloud)
OLLAMA_URL = 'http://localhost:11434/api/generate'

# Default vision-language model
DEFAULT_VLM = 'qwen2.5vl:7b'

# Default maximum images to download per inventor
DEFAULT_MAX_IMAGES = 5

# Default sleep between inventors (seconds)
DEFAULT_SLEEP = 2.5

# Default confidence threshold for accepting a photo classification
DEFAULT_CONFIDENCE_THRESHOLD = 75.0
