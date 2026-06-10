import yaml
import re
from pathlib import Path
import json

# отделяет метаданные от основного текста
def parse_front_matter(text: str) -> tuple[dict, str]:
    if text.startswith('---') and text.count('---') >= 2:
        parts = text.split('---', maxsplit = 2)
        body = parts[2].strip()
        if not body:
            raise  ValueError('The Body is empty!')
        try:
            metadata = yaml.safe_load(parts[1])
        except yaml.YAMLError:
            raise ValueError('The yaml is broken!')

        return metadata, body

# разделяет текст над подзаголовки и заголовки
def parse_sections(body:str) -> list[dict]:
    sections = []

    current_section = None
    current_subsection = None
    current_lines = []

    lines = body.splitlines()

    for line in lines:
        if line.startswith('## '):
            text = '\n'.join(current_lines).strip()

            if text:
                sections.append({
                    'section': current_section,
                    'subsection': current_subsection,
                    'text':text
                })

            current_subsection = line.removeprefix("## ").strip()
            current_lines = []
        elif line.startswith('# '):
            text = '/n'.join(current_lines).strip()

            if text:
                sections.append({
                    'section': current_section,
                    'subsection': current_subsection,
                    'text': text
                })

            current_section = line.removeprefix("# ").strip()
            current_subsection = None
            current_lines = []
        else:
            current_lines.append(line)
    text = '/n'.join(current_lines).strip()

    if text:
        sections.append({
            'section': current_section,
            'subsection': current_subsection,
            'text': text
        })
    return sections

# режет текст для RAG
def split_text(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    if overlap_chars >= max_chars:
        raise ValueError("overlap_chars должен быть меньше max_chars")

    paragraphs = [p.strip() for p in text.split("/n/n") if p.strip()]
    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        # Слишком длинный абзац делим по предложениям
        if len(paragraph) > max_chars:
            sentences = re.split(r"(?<=[.!?])\s+", paragraph)
        else:
            sentences = [paragraph]

        for sentence in sentences:
            candidate = (
                f"{current_chunk}\n\n{sentence}"
                if current_chunk
                else sentence
            )

            if len(candidate) <= max_chars:
                current_chunk = candidate
                continue

            if current_chunk:
                chunks.append(current_chunk)

                overlap = current_chunk[-overlap_chars:].strip()
                # Не начинаем overlap посередине слова
                if " " in overlap:
                    overlap = overlap.split(" ", 1)[1]

                candidate = f"{overlap} {sentence}".strip()

            current_chunk = candidate if len(candidate) <= max_chars else sentence

    if current_chunk:
        chunks.append(current_chunk)

    return chunks

# создаёт отдельный json документ для одного фильма
def build_document_chunks(source, max_charc, overlap_charc):
    final_result = []

    metadata, body = parse_front_matter(Path(source).read_text(encoding='utf-8'))

    sections_body = parse_sections(body)

    chunk_index = 0

    for section in sections_body:
        section_chunks = split_text(
            section["text"],
            max_chars=max_charc,
            overlap_chars=overlap_charc,
        )

        for chunk_text in section_chunks:
            chunk_index += 1
            chunk_id = f"{metadata['id']}-{chunk_index:04d}"

            final_result.append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "metadata": {
                    **metadata,
                    "document_id": metadata["id"],
                    "section": section["section"],
                    "subsection": section["subsection"],
                    "chunk_index": chunk_index,
                },
            })
    return final_result

# Объединяет весь пайплайн для создания комплекта документов
def preprocess_documents(
    input_dir: str,
    output_file: str,
    max_char: int,
    overlap_char: int,
) -> list[dict]:

    all_documents = []
    sources = sorted(Path(input_dir).glob("*.md"))

    for source in sources:
        documents = build_document_chunks(
            source,
            max_char,
            overlap_char,
        )
        all_documents.extend(documents)

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(all_documents, file, ensure_ascii=False, indent=2)

    return all_documents

# if __name__ == "__main__":
#     project_root = Path(__file__).resolve().parents[1]
#
#     input_dir = project_root / "docs" / "data"
#     output_file = project_root / "docs" / "documents.json"
#
#     documents = preprocess_documents(
#         input_dir=str(input_dir),
#         output_file=str(output_file),
#         max_char=384,
#         overlap_char=50,
#     )
#
#     print(f"Обработано chunks: {len(documents)}")
#     print(f"Результат сохранён: {output_file}")