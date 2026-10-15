# Rumor2Reality UI

Gradio interface for the Rumor2Reality project. The app lets a user enter a
rumor or claim, shows pipeline progress, displays an assessment, and lists the
retrieved evidence sources.

## Files

```text
app.py            # Gradio UI
pipeline.py       # Starter backend/data contract
requirements.txt  # Python dependency for the UI
```

## Run Locally

```bash
python -m pip install -r requirements.txt
python app.py
```

Then open the local Gradio URL, usually:

```text
http://127.0.0.1:7860
```

## Current Output

The UI expects the pipeline to return:

- verdict
- confidence
- evidence summary
- assessment text
- limitations
- retrieved evidence sources
- timing metadata

The current `pipeline.py` uses a small starter corpus so the UI can be tested
before the final backend is connected.

## Deploy

For Hugging Face Spaces:

1. Create a new Space.
2. Choose Gradio as the SDK.
3. Upload `app.py`, `pipeline.py`, `requirements.txt`, and this README.
4. Wait for the Space to build.

Do not commit API keys. Add them as Space secrets if the final backend needs
external model access.
