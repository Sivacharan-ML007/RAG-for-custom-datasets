from __future__ import annotations

import requests


HF_ROUTER_URL = "https://router.huggingface.co/v1/chat/completions"


def answer_question(
    vectorstore,
    question: str,
    model_id: str,
    hf_token: str,
    top_k: int,
    temperature: float,
    max_new_tokens: int,
) -> dict:

    source_documents = vectorstore.similarity_search(question, k=top_k)
    context = build_context(source_documents)
    answer = call_huggingface_router(
        model_id=model_id,
        hf_token=hf_token,
        question=question,
        context=context,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )
    return {"result": answer, "source_documents": source_documents}


def build_context(documents) -> str:
    chunks = []
    for index, document in enumerate(documents, start=1):
        metadata = document.metadata
        source = metadata.get("source", "file")
        table = metadata.get("table", metadata.get("sheet", "table"))
        row = metadata.get("row", "?")
        chunks.append(
            f"[{index}] Source: {source} / {table} / row {row}\n"
            f"{document.page_content}"
        )
    return "\n\n".join(chunks)


def call_huggingface_router(
    model_id: str,
    hf_token: str,
    question: str,
    context: str,
    temperature: float,
    max_new_tokens: int,
) -> str:
    headers = {
        "Authorization": f"Bearer {hf_token.strip()}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_id.strip(),
        "messages": [
            {
                "role": "system",
                "content": (
                    "Answer only from the dataset context. If the answer is not "
                    "present, say you could not find it in the uploaded dataset."
                ),
            },
            {
                "role": "user",
                "content": f"Dataset context:\n{context}\n\nQuestion: {question}",
            },
        ],
        "temperature": temperature,
        "max_tokens": max_new_tokens,
    }

    try:
        response = requests.post(
            HF_ROUTER_URL,
            headers=headers,
            json=payload,
            timeout=90,
        )
        response.raise_for_status()
        
    except requests.HTTPError as exc:
        raise RuntimeError(explain_huggingface_error(response)) from exc
    except requests.RequestException as exc:
        raise RuntimeError(f"Could not reach Hugging Face: {exc}") from exc

    data = response.json()
    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected Hugging Face response: {data}") from exc


def explain_huggingface_error(response: requests.Response) -> str:
    status = response.status_code

    try:
        detail = response.json()
    except ValueError:
        detail = response.text

    if status == 401:
        return (
            "Your Hugging Face token is invalid or missing. Create a token with "
            "`Inference Providers` access and put it in `.env`."
        )

    if status == 402:
        return (
            "Hugging Face says billing or credits are required for this provider/model. "
            "Try a smaller/free model in the Hugging Face Playground, then paste that model id here."
        )

    if status == 403:
        return (
            "Your token does not have access to this model/provider. Check that your "
            "token has `Inference Providers` permission and that you accepted any model terms."
        )

    if status == 404:
        return "Hugging Face could not find that model id. Check the model name."

    if status == 429:
        return "Hugging Face rate-limited the request. Wait a bit, then try again."

    if status >= 500:
        return "The Hugging Face provider is temporarily unavailable. Try again or switch models."

    return f"Hugging Face request failed with HTTP {status}: {detail}"


def format_sources(answer: dict) -> list[str]:
    sources = []
    for doc in answer.get("source_documents", []):
        metadata = doc.metadata
        location = (
            f"{metadata.get('source', 'file')}"
            f" / {metadata.get('table', metadata.get('sheet', 'table'))}"
            f" / row {metadata.get('row', '?')}"
        )
        if location not in sources:
            sources.append(location)
    return sources
