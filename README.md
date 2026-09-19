# AI Meeting Action-Item Extractor

A complete, medium-level, **fully offline-capable** project that converts meeting transcripts into structured action items.

**Output schema per action item:**
- `task` – short description of the work
- `owner` – person responsible
- `deadline` – date or relative deadline (ISO or natural language)
- `status` – `pending` | `completed`
- `confidence` – float 0–1

## Features

1. **20 realistic annotated transcripts** (105 gold action items)
2. Transcript cleaning & speaker/sentence segmentation
3. Hybrid extraction pipeline:
   - Rule-based + spaCy NER for people & dates
   - Zero-shot classification (Hugging Face `facebook/bart-large-mnli`)
   - Tunable hyperparameters (threshold, owner/deadline/rule boosts)
   - Optional pure rule-based mode (no model download)
4. Validation rules (missing owner, invalid/missing date, duplicates, low confidence)
5. Evaluation (exact + fuzzy match on task+owner)
6. Polished Streamlit UI:
   - Upload / Paste / Sample tabs
   - Sidebar hyperparameter controls
   - Color-coded table + metrics + CSV/JSON/Markdown export
7. Fully modular code, free & open-source

## Project Structure

```
ai-meeting-action-extractor/
├── data/
│   ├── sample_transcripts/          # 20 .txt transcripts
│   └── annotations.json             # Gold action items for evaluation
├── src/
│   ├── __init__.py
│   ├── cleaner.py                   # Cleaning + segmentation
│   ├── extractor.py                 # Core extraction pipeline
│   ├── validator.py                 # Validation rules
│   ├── evaluator.py                 # Accuracy metrics
│   └── utils.py                     # Helpers (dates, fuzzy, etc.)
├── evaluation/
│   └── run_eval.py                  # CLI evaluation script
├── app.py                           # Streamlit interface
├── requirements.txt
└── README.md
```

## Quick Start

```bash
# 1. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download spaCy English model
python -m spacy download en_core_web_sm

# 4. Run the Streamlit app
streamlit run app.py
```

The first run will download the zero-shot model (`facebook/bart-large-mnli` ≈ 1.6 GB). Subsequent runs are offline.

## Using the App

1. Open the browser tab that Streamlit opens.
2. Either **upload a .txt transcript** or **paste text**.
3. Click **Extract Action Items**.
4. Review the table. Rows with low confidence or missing fields are highlighted.
5. Download results as CSV if desired.

## Evaluation

```bash
python evaluation/run_eval.py
```

This loads the 20 sample transcripts + gold annotations and reports:
- Exact match accuracy (task + owner)
- Fuzzy match accuracy (rapidfuzz / Levenshtein)
- Precision / Recall / F1 at different confidence thresholds

## Design Decisions (Offline-First)

| Component              | Choice                                      | Why |
|------------------------|---------------------------------------------|-----|
| NER / people & dates   | spaCy `en_core.web_sm`                      | Fast, offline, good enough |
| Action-item detection  | Zero-shot BART-MNLI                         | Strong zero-shot, no fine-tuning needed |
| Task phrasing          | Rule-based + pattern extraction             | Reliable & deterministic |
| Dates                  | `dateparser` + heuristics                   | Handles relative dates (“next Friday”) |
| Validation             | Pure Python rules                           | Transparent & fast |
| UI                     | Streamlit                                   | Minimal code, good UX |

## Extending the Project

- Add more sample transcripts → drop `.txt` files + update `annotations.json`
- Improve model → swap zero-shot model or fine-tune a small classifier
- Add speaker diarization → integrate pyannote (still offline after download)
- Export to Jira / Notion / CSV → already supported via pandas
- Hyperparameter tuning → see `src/extractor.py` (thresholds are configurable)

## License

MIT – free for any use.
