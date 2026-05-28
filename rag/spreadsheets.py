from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
from langchain.docstore.document import Document


SUPPORTED_DATASET_TYPES = {
    ".csv",
    ".tsv",
    ".txt",
    ".xlsx",
    ".xls",
    ".xlsm",
    ".ods",
    ".json",
    ".jsonl",
    ".ndjson",
    ".parquet",
    ".feather",
}


def read_dataset(uploaded_file) -> dict[str, pd.DataFrame]:
    suffix = Path(uploaded_file.name).suffix.lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = Path(tmp.name)

    try:
        if suffix in {".csv", ".tsv"}:
            separator = "\t" if suffix == ".tsv" else ","
            return {
                Path(uploaded_file.name).stem: read_csv_with_fallbacks(
                    tmp_path, separator
                )
            }

        if suffix == ".txt":
            return {Path(uploaded_file.name).stem: read_delimited_text(tmp_path)}

        if suffix in {".xlsx", ".xls", ".xlsm", ".ods"}:
            return pd.read_excel(tmp_path, sheet_name=None)

        if suffix in {".json", ".jsonl", ".ndjson"}:
            return {Path(uploaded_file.name).stem: read_json_dataset(tmp_path, suffix)}

        if suffix == ".parquet":
            return {Path(uploaded_file.name).stem: pd.read_parquet(tmp_path)}

        if suffix == ".feather":
            return {Path(uploaded_file.name).stem: pd.read_feather(tmp_path)}

        raise ValueError(
            "Upload a supported dataset file: "
            + ", ".join(sorted(SUPPORTED_DATASET_TYPES))
            + "."
        )
    except ImportError as exc:
        raise ValueError(
            f"Reading {suffix} files requires an optional dependency. "
            "Install the packages from requirements.txt, then try again."
        ) from exc
    finally:
        tmp_path.unlink(missing_ok=True)


def read_spreadsheet(uploaded_file) -> dict[str, pd.DataFrame]:
    return read_dataset(uploaded_file)


def read_csv_with_fallbacks(path: Path, separator: str) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "utf-8", "latin1"):
        try:
            return pd.read_csv(path, sep=separator, encoding=encoding)
        except UnicodeDecodeError:
            continue
        except pd.errors.EmptyDataError as exc:
            raise ValueError("That CSV/TSV file is empty.") from exc

    raise ValueError("Could not read the CSV/TSV encoding.")


def read_delimited_text(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "utf-8", "latin1"):
        try:
            return pd.read_csv(path, sep=None, engine="python", encoding=encoding)
        except UnicodeDecodeError:
            continue
        except pd.errors.EmptyDataError as exc:
            raise ValueError("That text file is empty.") from exc

    raise ValueError("Could not read the text file encoding.")


def read_json_dataset(path: Path, suffix: str) -> pd.DataFrame:
    lines = suffix in {".jsonl", ".ndjson"}
    try:
        frame = pd.read_json(path, lines=lines)
    except ValueError as exc:
        raise ValueError("Could not read that JSON dataset file.") from exc

    if isinstance(frame, pd.Series):
        return frame.to_frame(name="value")
    return frame


def documents_from_upload(uploaded_file) -> list[Document]:
    frames = read_dataset(uploaded_file)
    documents: list[Document] = []

    for sheet_name, frame in frames.items():
        if frame.empty:
            continue

        clean_frame = frame.dropna(how="all").copy()
        if clean_frame.empty:
            continue

        clean_frame.columns = [str(column).strip() for column in clean_frame.columns]

        for row_number, (_, row) in enumerate(clean_frame.iterrows(), start=2):
            row_text = row_to_text(row)
            if not row_text:
                continue

            documents.append(
                Document(
                    page_content=row_text,
                    metadata={
                        "source": uploaded_file.name,
                        "table": sheet_name,
                        "sheet": sheet_name,
                        "row": row_number,
                    },
                )
            )

    return documents


def row_to_text(row: pd.Series) -> str:
    parts = []
    for column, value in row.items():
        text = normalize_cell(value)
        if text:
            parts.append(f"{column}: {text}")
    return "\n".join(parts)


def normalize_cell(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()
