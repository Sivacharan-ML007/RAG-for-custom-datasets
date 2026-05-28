# Hugging Face Dataset RAG

Hugging Face Dataset RAG is a Streamlit application for asking natural-language questions over tabular dataset files. It reads supported dataset formats, converts rows into searchable documents, builds a local FAISS vector index, and generates answers through a hosted open-source chat model using Hugging Face Router.

The application does not require an OpenAI key, Ollama, or a local LLM download.

## Requirements

- Python 3.13
- Hugging Face account
- Hugging Face token with `Inference Providers` access
- Dataset file: `.csv`, `.tsv`, `.txt`, `.xlsx`, `.xls`, `.xlsm`, `.ods`, `.json`, `.jsonl`, `.ndjson`, `.parquet`, or `.feather`

## Setup

Create a virtual environment, install the project dependencies, and copy the environment template:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

Update `.env` with your Hugging Face credentials and model configuration:

```text
HUGGINGFACEHUB_API_TOKEN=hf_your_token_here
HF_MODEL_ID=Qwen/Qwen2.5-7B-Instruct
```

## Run

Activate the virtual environment and start the Streamlit app:

```bash
source .venv/bin/activate
streamlit run app.py
```

Open the local URL printed by Streamlit, usually:

```text
http://localhost:8501
```

## Usage

1. Prepare a supported tabular dataset file.
2. Upload the file in the sidebar.
3. Click `Build index`.
4. Ask questions about the uploaded data.

The answer is generated from retrieved dataset rows. Source references are shown with the file name, table or sheet name, and row number when available.

## Project Structure

- `app.py`: Streamlit UI.
- `rag/config.py`: `.env` settings.
- `rag/spreadsheets.py`: dataset loading.
- `rag/vectorstore.py`: LangChain embeddings + FAISS.
- `rag/qa.py`: retrieval + Hugging Face Router chat call.

## Notes

- The app uses LangChain for document handling, embeddings, splitting, and FAISS retrieval.
- Hugging Face generation goes through `https://router.huggingface.co/v1/chat/completions`.
- If a selected model is unavailable or fails for the current account, try another chat model from the Hugging Face Playground and update `HF_MODEL_ID`.
