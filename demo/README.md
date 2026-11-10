# LWOW demo

Requires **Python 3.12**, **Git**, and internet access for initial setup.
Run all commands from the project root.

Setup commands:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\scripts\setup_lwow.ps1
.\.venv\Scripts\python.exe -m scripts.setup_wordnet
.\.venv\Scripts\python.exe -m streamlit run app.py
```

After setup, only the last command is needed.
The app will be available at **http://localhost:8501**.

The LWOW repo is downloaded to `lwow/`.
Demo references can be found in `data/reference/`.

Optional scripts

```powershell
.\.venv\Scripts\python.exe -m scripts.inspect_network --model Mistral
.\.venv\Scripts\python.exe -m scripts.evaluate_bias --model Mistral --max-targets 8
```

Use `--max-targets 0` to include all available target words in the walkthrough.

