# genderphoto

Gender classification of patent inventors (or any person) from their name, publicly available photos, and — when needed — a local vision-language model.

The package was developed for bibliometric research on inventor gender gaps, where relying on names alone fails systematically: "Andrea" is male in Italy and female in the US, most East-Asian given names are unisex to name classifiers, and so on. Rather than discarding these cases, `genderphoto` downloads photos via Bing, runs DeepFace on every image it finds, and calls a local VLM through Ollama when DeepFace results disagree or are uncertain.

No data ever leaves your machine — the VLM runs entirely on localhost.

## Installation

```bash
pip install git+https://github.com/zabbonat/genderphoto.git
```

Development install (editable, with test dependencies):

```bash
git clone https://github.com/zabbonat/genderphoto.git
cd genderphoto
pip install -e ".[dev]"
```

If you want VLM fallback support, also install the optional `ollama` client:

```bash
pip install genderphoto[vlm]
```

---

## Quick start

### Classify a single inventor

```python
from genderphoto import classify_inventor

result = classify_inventor(
    name="Andrea Cavalleri",
    affiliation="Max Planck Hamburg",
    country_code="DE",
)
print(result['gender'], result['method'], result['confidence'])
# M  deepface_consensus  98.5
```

### Classify a batch from a DataFrame

```python
import pandas as pd
from genderphoto import classify_batch

df = pd.DataFrame([
    {'inventor_name': 'Andrea Cavalleri', 'affiliation': 'Max Planck Hamburg', 'country_code': 'DE'},
    {'inventor_name': 'Jennifer Doudna', 'affiliation': 'UC Berkeley', 'country_code': 'US'},
    {'inventor_name': 'Fei-Fei Li', 'affiliation': 'Stanford University', 'country_code': 'US'},
])

result_df = classify_batch(df, save_photos=True, photo_dir='./inventor_photos')
print(result_df[['inventor_name', 'gender_final', 'gender_method']])
```

### Check name ambiguity only (no photos)

```python
from genderphoto import classify_name

classify_name("Andrea", "IT")   # → {'gender': 'M', 'is_ambiguous': False, ...}
classify_name("Andrea", "US")   # → {'gender': None, 'is_ambiguous': True, ...}
classify_name("Wei", "CN")      # → {'gender': None, 'is_ambiguous': True, ...}
```

---

## How the pipeline works

Each inventor goes through up to four stages. The pipeline stops as soon as one stage produces a confident result.

1. **Name-based classification** — uses `global-gender-predictor` (backed by the WGND 2.0 dataset with 4.1M names) to resolve unambiguous names instantly. Names with a predicted probability below the `name_threshold` (default 0.75) are flagged as ambiguous. Italian male names (Andrea, Simone, Nicola, …) used outside Italy are also flagged.

2. **Photo search** — For ambiguous names, Bing image search (via `icrawler`) downloads up to `max_images` photos. The search tries three queries in order: `"{name} {affiliation}"`, `"{name} researcher"`, and `"{name}"`, stopping at the first query that returns results.

3. **DeepFace consensus** — DeepFace (`retinaface` backend, `enforce_detection=True`) runs on every downloaded photo. If all images agree on the same gender with an average confidence ≥ 90%, the result is accepted without calling the VLM.

4. **VLM fallback** — When DeepFace results disagree across images, or the average confidence is below 90%, the best photo (highest DeepFace confidence) is sent to a local VLM via Ollama. If the VLM agrees with the DeepFace majority, confidence is set to 92%. If the VLM overrides DeepFace, confidence is 85%. If Ollama is not running, the pipeline falls back to DeepFace majority vote.

---

## API reference

### `classify_inventor`

```python
from genderphoto import classify_inventor

result = classify_inventor(
    name="Andrea Cavalleri",          # full name (required)
    affiliation="Max Planck Hamburg",  # improves photo search quality
    country_code="DE",                # ISO 2-letter code, for cross-cultural name checks
    max_images=5,                     # photos to download per inventor (default: 5)
    confidence_threshold=75.0,        # minimum confidence % to accept (default: 75.0)
    save_photo_flag=False,            # save the best photo to disk (default: False)
    photo_dir="./inventor_photos",    # where to save photos
    vlm_model="qwen2.5vl:7b",        # any Ollama vision model (see below)
    ollama_url="http://localhost:11434/api/generate",  # Ollama endpoint
    verbose=False,                    # True for step-by-step log output (default: False)
    name_threshold=0.75,              # Min probability to accept name-only classification (default: 0.75)
)
```

**Returns** a dict:

| Key | Type | Description |
|-----|------|-------------|
| `inventor_name` | str | Input name |
| `gender` | `'M'` / `'F'` / `'UNKNOWN'` | Final classification |
| `confidence` | float or None | Confidence %, None for name-based |
| `name_probability` | float or None | Name probability from WGND 2.0 |
| `method` | str | Which stage resolved it (e.g. `name_based`, `deepface_consensus`, `ensemble_vlm_override`) |
| `is_ambiguous` | bool | Whether the name was flagged ambiguous |
| `gender_raw` | str | Raw output from global-gender-predictor or consensus details |
| `ambiguity_reason` | str | Why the name was (or wasn't) considered ambiguous |
| `photo_url` | str or None | URL/path of the photo used for classification |
| `photo_saved_path` | str or None | Local path if `save_photo_flag=True` |
| `images_tried` | int | Number of images processed |
| `error` | str or None | Error description if something went wrong |

---

### `classify_batch`

```python
from genderphoto import classify_batch

result_df = classify_batch(
    df,                                # pandas DataFrame (required)
    name_col="inventor_name",          # column with full names (default: 'inventor_name')
    affiliation_col="affiliation",     # column with affiliations (default: 'affiliation')
    country_col="country_code",        # column with ISO country codes (default: 'country_code')
    max_images=5,                      # photos per inventor (default: 5)
    confidence_threshold=75.0,         # min confidence % (default: 75.0)
    sleep=2.5,                         # seconds between inventors, for rate limiting (default: 2.5)
    save_photos=True,                  # save best photos to disk (default: True)
    photo_dir="./inventor_photos",     # photo output directory
    vlm_model="qwen2.5vl:7b",         # Ollama vision model
    ollama_url="http://localhost:11434/api/generate",  # Ollama endpoint
    checkpoint_path="./checkpoint.csv",  # auto-save partial results (default: None)
    checkpoint_every=10,               # save checkpoint every N inventors (default: 10)
    verbose=False,                     # True for detailed per-inventor logging (default: False)
    name_threshold=0.75,               # Min probability to accept name-only classification
)
```

The batch function first runs name-based classification on every row. Only the ambiguous names go through the photo pipeline, which saves considerable time on large datasets.

**Columns added to the DataFrame:**

| Column | Description |
|--------|-------------|
| `gender_name` | Name-based classification (`'M'`, `'F'`, or None) |
| `gender_name_raw` | Raw `global-gender-predictor` output (`Male`, `Female`, `Unknown`) |
| `name_probability` | Probability score from WGND 2.0 (e.g., 0.85) |
| `is_ambiguous` | Whether photo pipeline was needed |
| `ambiguity_reason` | Reason for ambiguity flag |
| `gender_photo` | Photo-based classification (None if not needed) |
| `photo_confidence` | DeepFace/ensemble confidence % |
| `photo_url` | Photo used |
| `photo_saved_path` | Local path of saved photo |
| `photo_images_tried` | Number of images analyzed |
| `photo_classifier` | Which classifier resolved it |
| `photo_error` | Error description, if any |
| `gender_final` | Consolidated result: name-based if unambiguous, photo-based otherwise |
| `gender_method` | Which method produced `gender_final` |

---

### `classify_name`

Name-only classification, no photos, no network calls. Useful for quick pre-screening.

```python
from genderphoto import classify_name

result = classify_name(
    first_name="Andrea",    # first name only (required)
    country_code="IT",      # ISO 2-letter code (default: None)
)
```

**Returns** a dict with keys: `gender`, `gender_raw`, `is_ambiguous`, `ambiguity_reason`, `method`.

The `country_code` matters because Italian male names (Andrea, Simone, Nicola, Gabriele, Michele, Daniele, Raffaele, Samuele, Emanuele, Pasquale, Luca, Mattia) are classified as unambiguously male when `country_code="IT"`, but flagged ambiguous for any other country.

Names classified with a probability below `name_threshold` (default 0.75) are treated as ambiguous to avoid silent misclassifications on large datasets.

---

### `list_available_vlm_models`

Queries your local Ollama instance and returns which models are installed, flagging the ones that support vision.

```python
from genderphoto import list_available_vlm_models

models = list_available_vlm_models(
    ollama_url="http://localhost:11434/api/generate",  # default Ollama endpoint
)

# Each entry: {'name': 'qwen2.5vl:7b', 'size_gb': 4.7, 'is_vision': True}
for m in models:
    if m['is_vision']:
        print(f"  {m['name']}  ({m['size_gb']} GB)  ← vision model")
```

Returns an empty list if Ollama is not running.

---

## Verbose mode

By default both `classify_inventor` and `classify_batch` run silently. In batch mode you get a progress bar and a short summary:

```
✓ Name-based: 120/150 resolved, 30 need photo analysis
Andrea Cavalleri       | 12/30 [00:45<01:30, 4.0s/inv] → M (deepface_consensus)
✓ Complete: 145/150 classified
```

Set `verbose=True` to see exactly what happens at each step — which images were downloaded, what DeepFace returned for each one, whether the VLM was called, and what it answered:

```python
result = classify_inventor("Andrea Cavalleri", affiliation="Max Planck", verbose=True)
result_df = classify_batch(df, verbose=True)
```

---

## Setting up the VLM

The VLM stage is optional. Without it, the pipeline falls back to DeepFace majority vote when images disagree.

To enable VLM fallback, install [Ollama](https://ollama.com/) and pull a vision-capable model:

```bash
ollama pull qwen2.5vl:7b
ollama serve
```

### Using a different model

Pass `vlm_model` to use any vision model you have installed in Ollama:

```python
# Lighter model, faster inference
result = classify_inventor("Wei Zhang", vlm_model="qwen2.5vl:3b")

# Use LLaVA instead of Qwen
result = classify_inventor("Wei Zhang", vlm_model="llava:7b")

# Same in batch mode
result_df = classify_batch(df, vlm_model="moondream:1.8b")
```

### Recommended models

Any vision-capable model that runs on Ollama will work. Some options, roughly ordered by resource requirements:

| Model | VRAM needed | Notes |
|-------|-------------|-------|
| `moondream:1.8b` | ~2 GB | Very fast, lowest accuracy |
| `qwen2.5vl:3b` | ~3 GB | Decent accuracy, fast |
| `qwen2.5vl:7b` | ~6 GB | **Default.** Best tradeoff between speed and accuracy |
| `llava:7b` | ~6 GB | Good alternative to Qwen |
| `minicpm-v:8b` | ~7 GB | Another solid option |
| `llava:13b` | ~10 GB | Higher accuracy, slower |
| `qwen2.5vl:72b` | ~48 GB | Best accuracy, needs serious hardware |

Install any of them with `ollama pull <model_name>`.

### Custom Ollama endpoint

If Ollama is running on a different machine or port:

```python
result = classify_inventor(
    "Wei Zhang",
    ollama_url="http://192.168.1.100:11434/api/generate",
)
```

---

## Accuracy

~93% on a hand-curated validation set of 100 researchers with known gender, covering:

- Italian names used cross-culturally (Andrea, Simone, Nicola — inside and outside Italy)
- East Asian names (Wei, Jie, Fei-Fei, Yuki)
- French ambiguous names (Dominique, Claude, Camille)
- English ambiguous names (Robin, Kim, Jamie, Morgan)
- Unambiguous male and female controls

The full validation dataset is in `tests/test_validation_100.py`.

## Limitations

- **Photo noise.** Bing sometimes returns photos of the wrong person, especially for common names. Adding an affiliation helps.
- **DeepFace on Asian faces.** DeepFace has a documented bias toward classifying East-Asian women as male. The VLM fallback partially corrects this.
- **Binary only.** The classifier outputs M or F. Non-binary gender identities are not represented.
- **Public photos required.** Inventors with no photos indexed by Bing cannot be classified beyond the name stage.
- **Rate limiting.** Bing may throttle after many consecutive requests. Increase the `sleep` parameter for large batches.

## Dependencies

Core: `pandas`, `numpy`, `Pillow`, `requests`, `deepface`, `retina-face`, `icrawler`, `global-gender-predictor`, `tqdm`.

Optional: `ollama` (Python client, for VLM support).

## Citation

If you use this software, please cite it:

```bibtex
@software{genderphoto,
  author = {Abbonato, Diletta},
  title  = {genderphoto: Gender classification of patent inventors using name + photo + VLM ensemble},
  year   = {2025},
  url    = {https://github.com/zabbonat/genderphoto}
}
```

The name classification relies on the World Gender Name Dictionary (WGND 2.0). If your work benefits from the name-based stage, please also cite the WGND dataset:

```bibtex
@data{DVN/MSEGSJ_2021,
  author = {Raffo, Julio},
  publisher = {Harvard Dataverse},
  title = {{WGND 2.0}},
  year = {2021},
  doi = {10.7910/DVN/MSEGSJ},
  url = {https://doi.org/10.7910/DVN/MSEGSJ}
}
```

## License

MIT. See [LICENSE](LICENSE).
