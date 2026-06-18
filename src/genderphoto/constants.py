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
DEFAULT_MAX_IMAGES = 3

# Default search engine for photos ('auto', 'duckduckgo', 'bing', 'baidu')
DEFAULT_SEARCH_ENGINE = 'auto'

# Default sleep between inventors (seconds)
DEFAULT_SLEEP = 0.5

# Default confidence threshold for accepting a photo classification
DEFAULT_CONFIDENCE_THRESHOLD = 75.0

# Default probability threshold for accepting a name classification
DEFAULT_NAME_THRESHOLD = 0.75

# Known vision-capable model name patterns (for Ollama auto-detection)
VISION_MODEL_PATTERNS = [
    'qwen2.5vl', 'qwen2-vl', 'qwenvl',
    'llava', 'bakllava',
    'minicpm-v', 'minicpm_v',
    'moondream',
    'llama3.2-vision', 'llama-vision',
    'internvl', 'cogvlm',
]

# Suggested VLM models (for documentation / user guidance)
SUGGESTED_VLM_MODELS = [
    'qwen2.5vl:7b',      # Default – good balance of speed and accuracy
    'qwen2.5vl:3b',      # Faster, slightly less accurate
    'qwen2.5vl:72b',     # Most accurate, requires significant GPU RAM
    'llava:7b',           # Alternative VLM
    'llava:13b',          # Larger LLaVA
    'minicpm-v:8b',       # MiniCPM-V
    'moondream:1.8b',     # Lightweight, very fast
]
