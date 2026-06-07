import argparse
import json
import re
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "docs" / "data"
DEFAULT_OUTPUT_FILE = PROJECT_ROOT / "docs" / "documents.json"
HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
REQUIRED_METADATA = {"id", "title"}


def parse_front_matter(text: str, source: Path) -> tuple[dict[str, Any], str]:
    lines = text.splitlines()

    # если нету ---, тогда отменяется
    if not lines or lines[0].strip() != "---":
        raise ValueError(f"{source}: YAML front matter must start with '---'")

    try:
        closing_index = next(
            index
            for index, line in enumerate(lines[1:], start=1)
            if line.strip() == "---"
        )
    except StopIteration as exc:
        raise ValueError(f"{source}: YAML front matter has no closing '---'") from exc

    metadata = yaml.safe_load("\n".join(lines[1:closing_index]))
    if not isinstance(metadata, dict):
        raise ValueError(f"{source}: YAML front matter must contain a mapping")

    missing_fields = REQUIRED_METADATA - metadata.keys()
    if missing_fields:
        missing = ", ".join(sorted(missing_fields))
        raise ValueError(f"{source}: missing required metadata: {missing}")

    body = "\n".join(lines[closing_index + 1 :]).strip()
    if not body:
        raise ValueError(f"{source}: document body is empty")

    return metadata, body


def parse_sections(body: str) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    current_heading = "Без раздела"
    current_parent: str | None = None
    current_level = 0
    current_lines: list[str] = []

    def save_section() -> None:
        content = "\n".join(current_lines).strip()
        if content:
            section_path = (
                f"{current_parent} > {current_heading}"
                if current_parent and current_level > 1
                else current_heading
            )
            sections.append(
                {
                    "section": current_parent or current_heading,
                    "subsection": current_heading if current_level > 1 else None,
                    "section_path": section_path,
                    "heading_level": current_level,
                    "content": content,
                }
            )

    for line in body.splitlines():
        heading_match = HEADING_PATTERN.match(line)
        if not heading_match:
            current_lines.append(line)
            continue

        save_section()
        current_lines = []
        current_level = len(heading_match.group(1))
        current_heading = heading_match.group(2).strip()
        if current_level == 1:
            current_parent = current_heading

    save_section()
    return sections


def split_text(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    if max_chars < 1:
        raise ValueError("max_chars must be greater than zero")
    if overlap_chars < 0 or overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be between zero and max_chars - 1")

    normalized = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(normalized) <= max_chars:
        return [normalized]

    chunks: list[str] = []
    start = 0

    while start < len(normalized):
        end = min(start + max_chars, len(normalized))
        if end < len(normalized):
            search_start = start + max_chars // 2
            boundary = max(
                normalized.rfind("\n\n", search_start, end),
                normalized.rfind(". ", search_start, end),
                normalized.rfind(" ", search_start, end),
            )
            if boundary > start:
                end = boundary + 1

        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(normalized):
            break
        start = max(end - overlap_chars, start + 1)

    return chunks


def build_document_chunks(
    source: Path,
    max_chars: int,
    overlap_chars: int,
) -> list[dict[str, Any]]:
    metadata, body = parse_front_matter(source.read_text(encoding="utf-8"), source)
    sections = parse_sections(body)
    chunks: list[dict[str, Any]] = []
    document_id = str(metadata["id"])
    document_chunk_index = 0

    for section in sections:
        section_chunks = split_text(
            section["content"],
            max_chars=max_chars,
            overlap_chars=overlap_chars,
        )
        for section_chunk_index, text in enumerate(section_chunks, start=1):
            document_chunk_index += 1
            chunk_metadata = {
                **metadata,
                "document_id": document_id,
                "source_file": source.name,
                "section": section["section"],
                "subsection": section["subsection"],
                "section_path": section["section_path"],
                "heading_level": section["heading_level"],
                "chunk_index": document_chunk_index,
                "section_chunk_index": section_chunk_index,
            }
            chunks.append(
                {
                    "chunk_id": f"{document_id}-{document_chunk_index:04d}",
                    "text": text,
                    "metadata": chunk_metadata,
                }
            )

    if not chunks:
        raise ValueError(f"{source}: no chunks were created")
    return chunks


def preprocess_documents(
    input_dir: Path,
    output_file: Path,
    max_chars: int,
    overlap_chars: int,
) -> list[dict[str, Any]]:
    sources = sorted(input_dir.glob("*.md"))
    if not sources:
        raise ValueError(f"No Markdown documents found in {input_dir}")

    all_chunks: list[dict[str, Any]] = []
    document_ids: set[str] = set()

    for source in sources:
        document_chunks = build_document_chunks(source, max_chars, overlap_chars)
        document_id = document_chunks[0]["metadata"]["document_id"]
        if document_id in document_ids:
            raise ValueError(f"Duplicate document id: {document_id}")
        document_ids.add(document_id)
        all_chunks.extend(document_chunks)

    chunk_ids = [chunk["chunk_id"] for chunk in all_chunks]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("Duplicate chunk ids were generated")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(all_chunks, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return all_chunks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Markdown movie documents into JSON chunks for RAG."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--max-chars", type=int, default=1000)
    parser.add_argument("--overlap-chars", type=int, default=150)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chunks = preprocess_documents(
        input_dir=args.input_dir,
        output_file=args.output_file,
        max_chars=args.max_chars,
        overlap_chars=args.overlap_chars,
    )
    document_count = len({chunk["metadata"]["document_id"] for chunk in chunks})
    print(
        f"Processed {document_count} documents into {len(chunks)} chunks. "
        f"Saved to {args.output_file}"
    )


if __name__ == "__main__":
    main()
