# genderphoto

**Gender classification of patent inventors** (or any person) using an ensemble of name-based inference, photo-based face classification (DeepFace), and vision-language model fallback (Qwen2.5-VL via Ollama).

Designed for bibliometric research on inventor gender gaps, where name-only approaches fail on cross-cultural names (e.g., "Andrea" is male in Italy, female in the US) and East Asian names.

## Installation

```bash
pip install git+https://github.com/zabbonat/genderphoto.git
```

For development:

```bash
git clone https://github.com/zabbonat/genderphoto.git
cd genderphoto
pip install -e ".[dev]"
```

## Quick Start

### Single inventor

```python
from genderphoto import classify_inventor

result = classify_inventor(
    name="Andrea Cavalleri",
    affiliation="Max Planck Hamburg",
    country_code="DE",
)
print(result['gender'], result['method'], result['confidence'])
# M  deepface_consensus  100.0
```

### Batch DataFrame

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

## VLM Setup (Optional)

The vision-language model fallback requires [Ollama](https://ollama.com/) running locally:

```bash
ollama pull qwen2.5vl:7b
ollama serve
```

If Ollama is not running, the pipeline gracefully falls back to DeepFace majority vote.

## Pipeline

1. **Name-based** (gender_guesser): classifies unambiguous names instantly
2. **Photo search** (Bing via icrawler): downloads up to 5 photos per ambiguous inventor
3. **DeepFace consensus**: analyzes ALL photos; accepts if all agree with ≥90% avg confidence
4. **VLM fallback** (Qwen2.5-VL): resolves disagreements or low-confidence cases

## Accuracy

~93% on a validation dataset of 100 researchers with known gender, spanning:

- Cross-cultural Italian names (Andrea, Simone, Nicola in/outside Italy)
- East Asian names (Wei, Jie, Fei-Fei, Yuki)
- French ambiguous names (Dominique, Claude, Camille)
- English ambiguous names (Robin, Kim, Jamie, Morgan)
- Clear male/female controls

See `tests/test_validation_100.py` for the full dataset.

## Known Limitations

- **Bing search noise**: may return photos of a different person with the same common name
- **DeepFace bias**: systematic misclassification of Asian women as male (partially mitigated by VLM fallback)
- **Binary gender only**: classifies as M/F; non-binary identities are not represented
- **Public photos required**: cannot classify inventors with no publicly available photos
- **Rate limiting**: Bing may throttle after many requests; use the `sleep` parameter

## Citation

```bibtex
@article{abbonato2026genderphoto,
  title={Gender Classification of Patent Inventors Using an Ensemble of
         Name-Based Inference and Photo-Based Face Analysis},
  author={Abbonato, Diletta},
  journal={TBD},
  year={2026},
  doi={PLACEHOLDER}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
