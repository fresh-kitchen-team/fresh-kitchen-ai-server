# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`fresh-kitchen-ai-server` is the AI backend for the fresh-kitchen application. It provides three independent ML pipelines:

1. **Food Classifier** — EfficientNet V2-M identifies food items from images; falls back to Gemini Vision if confidence < 75%
2. **Receipt OCR** — Google Document AI extracts text from receipt images; Gemini filters results to food items only
3. **Object Detection** — YOLOv8n detects items in refrigerator photos

## Setup

```bash
# Requires Python 3.10.19 (conda environment "project")
conda activate project
pip install -r requirements.txt

# API keys (required for OCR and Gemini fallback)
cp .env.example .env
# Fill in: GOOGLE_APPLICATION_CREDENTIALS, PROJECT_ID, PROCESSOR_ID, LOCATION, GEMINI_API_KEY

# Model weights (not in git — obtain from team drive)
# Place best_food_model_v2_m_ver2.pth in the project root
```

## Running the Models

```bash
# Food classification (single image prediction)
python models/food_classifier/predict_V2-M.py

# Receipt OCR
python models/receipt_ocr/receipt_ocr.py

# Refrigerator object detection
python models/object_detection/yolo_predict.py

# Evaluate classifier on test set
python models/food_classifier/test_V2-M.py

# Train food classifier (30 epochs max, early stopping at patience=5)
python training/train_EfficientNet_V2-M.py
```

## Data Pipeline Scripts

Run in this order when building a new dataset:

```bash
python scripts/data_crawl.py    # Download images via Bing
python scripts/data_clean.py    # Remove duplicates
python scripts/data_split.py    # Create train/val splits
python scripts/data_diet.py     # Cap dataset size per class
python scripts/data_blurred.py  # Generate blurred validation images
python scripts/data_len.py      # Print class distribution stats
```

## Architecture

### Confidence-Based Fallback (Food Classifier)
`models/food_classifier/predict_V2-M.py` implements a two-stage prediction:
- EfficientNet V2-M is the primary model (loaded once via `load_food_model()`)
- If softmax confidence < 75%, `gemini_predict()` is called with the full class list
- Gemini predictions are auto-saved to `dataset/auto_labeled/{label}/` to improve future training

### Two-Stage OCR (Receipt)
`models/receipt_ocr/receipt_ocr.py`:
- Stage 1: `process_receipt_raw()` calls Google Document AI for text extraction
- Stage 2: `filter_with_gemini()` sends extracted text to Gemini to keep only food items; forces JSON response via `response_mime_type="application/json"`

### Hardware Acceleration
All model scripts detect and prefer `mps` (Apple Silicon) → `cuda` → `cpu`. Batch size is capped at 8 for MPS stability.

### Dataset Structure
```
dataset/
├── train/, val/, test/   # Class-separated folders (45 food classes)
├── crawldata/            # Raw Bing downloads
└── auto_labeled/         # Gemini-labeled images (future training data)
```

Training uses `WeightedRandomSampler` to handle class imbalance.

## Key Files

| File | Purpose |
|------|---------|
| `models/food_classifier/predict_V2-M.py` | Main classifier with Gemini fallback |
| `models/food_classifier/test_V2-M.py` | Per-class accuracy evaluation |
| `models/receipt_ocr/receipt_ocr.py` | Document AI + Gemini OCR pipeline |
| `models/object_detection/yolo_predict.py` | YOLOv8n inference |
| `training/train_EfficientNet_V2-M.py` | Model training script |
| `requirements.txt` | All Python dependencies |
| `.env` | API credentials (not in git) |

## Credentials & Secrets

Never commit:
- `.env` — contains `GEMINI_API_KEY` and Google Cloud project details
- `receipt-app-*.json` — Google Cloud service account key
- `*.pth` / `*.pt` — model weight files (200 MB+)
- `dataset/` — training images

`GOOGLE_APPLICATION_CREDENTIALS` in `.env` supports relative paths; the code expands them to absolute paths at runtime.

# FreshKitchenAIServer AI Working Rules

This file is the entrypoint for AI agents working in this repository.
Keep detailed workflows in their source documents and use this file as the
project-level architecture guide and index.

## Workflow Index

Use this table for routing only. After choosing a workflow skill, let that skill
own the detailed file-reading order. Do not pre-read downstream documents from
this table unless the task is specifically about that document.

| Task | Source of truth |
| --- | --- |
| Commit analysis and commit creation | `.claude/skills/smart-commit/SKILL.md` |
| PR title/body writing | `.claude/skills/pr-writer/SKILL.md` |
| Commit and branch convention maintenance | `docs/git-convention.md` |

## Commit Workflow

- Always route git-related work to the `smart-commit` skill.
- The skill owns diff inspection, selective staging, commit splitting, and commit
  message rules.

## Pull Request Workflow

- Always route PR title/body creation or editing to the `pr-writer` skill.

## Automation Boundary

Allowed without extra user confirmation:

- git diff/status analysis
- selective staging
- commit creation

Not allowed unless the user explicitly asks:

- git push
- PR creation
- merge

## Default Git Behavior

When changes exist and the user asks for implementation or cleanup:

1. inspect branch and working tree
2. analyze the diff
3. create a commit plan
4. stage selectively
5. create smart commits
6. stop before push

