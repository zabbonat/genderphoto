# genderphoto — *Andrea Is a Man*

Gender classification of patent inventors (or any person) from their name, publicly available photos, and — when needed — a local vision-language model.

The package was developed for bibliometric research on inventor gender gaps, where relying on names alone fails systematically. The name **Andrea** exemplifies the core problem: it is unambiguously male in Italy, but classified as female in Germany, Austria, Spain, and most English-speaking countries by standard name-based tools (e.g., `gender-guesser`, WGND 2.0). An Italian physicist working at Max Planck in Hamburg would be silently misclassified as a woman — with 96% confidence — by any conventional name-gender pipeline that trusts the local convention of the country of residence. The same problem affects Simone (male in Italy, female in France), Nicola (male in Italy, female in the UK), Dominique (male in France, female in the US), and hundreds of East-Asian given names that are unisex to name classifiers.

Rather than discarding these ambiguous cases, `genderphoto` flags them and downloads publicly available photos, runs DeepFace on every image it finds, and calls a local VLM through Ollama when DeepFace results disagree or are uncertain. **No data ever leaves your machine** — the VLM runs entirely on localhost.

## Installation

```bash
pip install git+https://github.com/zabbonat/genderphoto.git
```

> [!IMPORTANT]
> **Python version:** We strongly recommend using **Python 3.9, 3.10, or 3.11**. Python 3.12+ might have compatibility issues with `tensorflow` and `deepface`. A virtual environment (like `conda` or `venv`) is highly recommended to avoid dependency conflicts with other machine learning libraries.

Development install (editable, with test dependencies):

```bash
git clone https://github.com/zabbonat/genderphoto.git
cd genderphoto
pip install -e ".[dev]"
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
# → M  deepface_consensus  98.5
# Andrea in DE is cross-cultural → flagged ambiguous → resolved via photo
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

classify_name("Andrea", "IT")   # → {'gender': 'M', 'is_ambiguous': False}   Andrea is a man in Italy
classify_name("Andrea", "DE")   # → {'gender': None, 'is_ambiguous': True}   ambiguous outside Italy → photo
classify_name("Andrea", None)   # → {'gender': None, 'is_ambiguous': True}   unknown country → photo
classify_name("Wei", "CN")      # → {'gender': None, 'is_ambiguous': True}   East Asian → photo
classify_name("Dominique", "FR") # → {'gender': None, 'is_ambiguous': True}  curated cross-cultural → photo
classify_name("Jennifer", "US") # → {'gender': 'F', 'is_ambiguous': False}   unambiguous
```

---

## How the pipeline works

Each inventor goes through up to four stages. The pipeline stops as soon as one stage produces a confident result.

### Stage 1: Name-based screening & Cross-cultural conflict filter

Uses `global-gender-predictor` (backed by WIPO WGND 2.0 with 4.1M names) combined with a dedicated Pinyin dictionary for Chinese given names. Crucially, it incorporates a **cross-cultural conflict filter** using `gender-guesser` to prevent silent misclassifications on names whose gender meaning changes across countries.

The filter operates through three mechanisms:

1. **Italian male name override** — A curated set of names that are unambiguously male in Italy but female or ambiguous elsewhere (Andrea, Simone, Nicola, Gabriele, Michele, Daniele, Raffaele, Samuele, Emanuele, Pasquale, Luca, Mattia). When `country_code="IT"`, these are classified as **male with probability 1.0** without photo verification. For **any other country** (or missing country code), they are flagged as `is_ambiguous=True` and routed to photo analysis. This is the "Andrea is a man" rule: an Andrea residing in Italy is classified as male; an Andrea residing in Germany, the US, or with unknown country is treated as ambiguous and verified via photo.

2. **Curated cross-cultural names** — Names like Dominique, Claude, Camille, Robin, Kim, and others that carry opposite gender connotations in different cultures are **always flagged as ambiguous** regardless of country, bypassing any probabilistic score.

3. **Low-probability threshold** — Any name with a probability score below `name_threshold` (default 0.75) is flagged as ambiguous.

The `country_code` parameter must be the **inventor's country of residence** (from patent address metadata), while institutional `affiliation` is reserved exclusively for Stage 2 (photo search).

### Stage 2: Photo search

For ambiguous names, image search downloads up to `max_images` photos. The default `search_engine="auto"` enables a cascading fallback (`bing` → `duckduckgo`, plus `baidu` for East Asian names) to maximize retrieval rates. When an affiliation is provided, the search queries `"{name} {affiliation}"`; otherwise it falls back to `"{name} researcher"` and then `"{name}"`. The **inventor's institutional affiliation** (`affiliation`) helps resolve homonyms and ensure high precision.

### Stage 3: DeepFace consensus

DeepFace (`retinaface` backend, `enforce_detection=True`) analyzes every downloaded photo. If all detected faces across all valid images agree on the same gender with an average confidence ≥ 90%, the consensus result is accepted (`deepface_consensus`) without calling the VLM.

### Stage 4: VLM tiebreaker / override

When DeepFace produces conflicting results across images or yields uncertain confidence (< 90%), the best-quality image is passed to a local Vision-Language Model (`Qwen2.5-VL` via Ollama). Acting as an algorithmic **tiebreaker**, the VLM interprets the photograph holistically (evaluating attire, presentation, and context rather than localized facial geometry alone). If the VLM agrees with the DeepFace majority, the result is confirmed with 92% confidence (`ensemble_vlm_majority_agree`). If the VLM disagrees, its assessment overrides DeepFace (`ensemble_vlm_override` with 95% confidence), mitigating well-documented algorithmic biases of face detectors against East Asian women.


---

## API reference

### `classify_inventor`

```python
from genderphoto import classify_inventor

result = classify_inventor(
    name="Andrea Cavalleri",          # full name (required)
    affiliation="Max Planck Hamburg",  # improves photo search quality
    country_code="DE",                # ISO 2-letter code, for cross-cultural name checks
    max_images=3,                     # photos to download per inventor (default: 3)
    confidence_threshold=75.0,        # minimum confidence % to accept (default: 75.0)
    save_photo_flag=False,            # save the best photo to disk (default: False)
    photo_dir="./inventor_photos",    # where to save photos
    search_engine="auto",             # 'auto', 'bing', 'duckduckgo', or 'baidu' (default: 'auto')
    detector_backend="retinaface",    # face detector engine (default: 'retinaface')
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
    max_images=3,                      # photos per inventor (default: 3)
    confidence_threshold=75.0,         # min confidence % (default: 75.0)
    sleep=0.5,                         # seconds between inventors, for rate limiting (default: 0.5)
    save_photos=True,                  # save best photos to disk (default: True)
    photo_dir="./inventor_photos",     # photo output directory
    search_engine="auto",              # 'auto', 'bing', 'duckduckgo', or 'baidu' (default: 'auto')
    detector_backend="retinaface",     # face detector engine (default: 'retinaface')
    vlm_model="qwen2.5vl:7b",         # Ollama vision model
    ollama_url="http://localhost:11434/api/generate",  # Ollama endpoint
    checkpoint_path="checkpoint.csv",  # auto-save partial results (default: 'checkpoint.csv')
    checkpoint_every=500,              # save checkpoint every N inventors (default: 500)
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

The `country_code` matters because Italian male names (Andrea, Simone, Nicola, Gabriele, Michele, Daniele, Raffaele, Samuele, Emanuele, Pasquale, Luca, Mattia) are classified as unambiguously male when `country_code="IT"`, but flagged ambiguous (`is_ambiguous=True`) for any other country or missing country code (`DE`, `US`, `FR`, `UK`, `None`), routing them to photo search.

Names classified with a probability below `name_threshold` (default 0.75) as well as curated cross-cultural names (`Dominique`, `Claude`, `Camille`, `Robin`, etc.) are treated as ambiguous to avoid silent misclassifications on large datasets.

---

### `compute_partial_identification_bounds`

Computes Manski bounds (1989) for the female share across a population where some inventors remain unclassified (`UNKNOWN` or `None`), separates name-resolved vs photo-resolved shares ($p_N$ vs $p_P$), and calculates the **country-matched plausible scenario** ($p_{matched}$).

```python
from genderphoto import compute_partial_identification_bounds

# Pass the final classified DataFrame (including method and country columns if available)
bounds = compute_partial_identification_bounds(
    result_df, 
    gender_col="gender_final",
    method_col="gender_method",
    country_col="country_code"
)

print(f"Observed F share overall: {bounds['observed_female_share']}%")
print(f"  - Name-resolved share (p_N): {bounds['observed_female_share_name_resolved']}%")
print(f"  - Photo/VLM-resolved share (p_P): {bounds['observed_female_share_photo_resolved']}%")
print(f"Plausible Manski bounds for total population: [{bounds['lower_bound']}%, {bounds['upper_bound']}%]")
print(f"Country-matched plausible scenario (p_matched): {bounds['country_matched_share']}%")
```

**Returns** a dict containing `total_population`, `classified_count`, `unknown_count`, `female_count`, `male_count`, `observed_female_share`, `observed_female_share_name_resolved` ($p_N$), `observed_female_share_photo_resolved` ($p_P$), `lower_bound` (% female assuming all unknowns are male), `upper_bound` (% female assuming all unknowns are female), and `country_matched_share` (% female where unknowns are imputed using country-specific classified female shares).

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

Core: `pandas`, `numpy`, `Pillow`, `requests`, `deepface`, `retina-face`, `tf-keras`, `icrawler`, `global-gender-predictor`, `gender-guesser`, `ddgs`, `tqdm`.

(The VLM fallback connects to the local Ollama API via HTTP requests, so no additional Python packages are required, only the Ollama application itself).

## Citation

If you use this software, please cite it:

```bibtex
@software{genderphoto,
  author = {Abbonato, Diletta and Maronero, Cecilia},
  title  = {genderphoto: Andrea Is a Man},
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
