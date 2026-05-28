# Quick Instructions

1. Create environment:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
```

2. Install:

```bash
pip install -r requirements.txt
```

3. Configure token:

```bash
cp .env.example .env
```

Edit `.env`:

```text
HUGGINGFACEHUB_API_TOKEN=hf_your_token_here
HF_MODEL_ID=Qwen/Qwen2.5-7B-Instruct
```

4. Run:

```bash
streamlit run app.py
```

5. In browser:

- Upload `.csv` or `.xlsx`.
- Click `Build index`.
- Ask your question.

If packages are stale:

```bash
pip install --upgrade --force-reinstall -r requirements.txt
```
