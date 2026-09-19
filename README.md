# AI Meeting Action-Item Extractor

A complete **fully offline-capable** project that converts meeting transcripts into structured action items.

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
# 1. Create virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 3. Run the app
streamlit run app.py
The first run will download the zero-shot model (`facebook/bart-large-mnli` ≈ 1.6 GB). Subsequent runs are offline.


##How to Use the App

Choose a sample transcript from the sidebar or
Upload a .txt file or
Paste any meeting transcript
Click Extract Action Items
Review the color-coded table
Download results as CSV / JSON / Markdown


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


##(future improvements)
1,Speaker Diarization,Automatically detect “who spoke when” from audio (using pyannote.audio) so you don’t need pre-written speaker names
2,Audio → Text Support,Upload .mp3 / .wav meeting recordings and convert them to transcript using Whisper (offline)
3,Better Date Understanding,"Improve relative dates like “next Friday”, “end of next week”, “in 3 days”"
4,Edit & Confirm UI,Let users edit extracted action items directly in the Streamlit table before exporting
5,Save History,Save previous extractions so users can go back and review old meetings


## License
MIT – free for any use.


##Auther
---
Developed by **Kiran patil**


