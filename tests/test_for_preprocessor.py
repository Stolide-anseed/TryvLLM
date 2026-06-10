from pathlib import Path
import json
from Rag.preprocessor import parse_front_matter, parse_sections, build_document_chunks


sorse  = r'C:\Users\stoli\PycharmProjects\TryvLLM\docs\data\knives_out_1_rag_short.md'

# metadata, body = parse_front_matter(text)
#
# section = parse_sections(body)

document = build_document_chunks(sorse, max_charc=500, overlap_charc=50)

with open("test.json", "w", encoding="utf-8") as file:
    json.dump(document, file, ensure_ascii=False, indent=2)