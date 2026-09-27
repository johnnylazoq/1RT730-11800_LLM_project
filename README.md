# 1RT730-11800 LLM Project

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Structure

```
src/llm_project/
    data/         # data loading & preprocessing
    models/       # model definitions
    training/      # training loops / scripts
    evaluation/    # evaluation & metrics
    utils/         # shared helpers
notebooks/         # exploratory notebooks
scripts/           # CLI entry points
tests/             # unit tests
configs/           # experiment configs
data/raw/          # raw datasets (gitignored)
data/processed/    # processed datasets (gitignored)
outputs/           # checkpoints & logs (gitignored)
```
