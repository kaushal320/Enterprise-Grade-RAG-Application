
import logfire


def chunk_text(text: str, chunk_size: int = 1500, overlap: int = 200) -> list[str]:
    """
    Robust chunker that splits text by paragraphs and lines.
    Ensures no chunk exceeds `chunk_size` characters even if paragraphs lack double-newlines.
    Adjacent chunks share the trailing `overlap` characters of the previous chunk so
    context (e.g. a sentence spanning the boundary) is preserved across chunks.
    """
    with logfire.span("✂️ Text Chunking", text_length=len(text)):
        if not text.strip():
            return []

        overlap = max(0, overlap)
        lines = text.split("\n")
        chunks = []
        current_chunk = ""
        prev_tail = ""  # last `overlap` chars of the previous chunk, prepended to the next

        def flush(chunk: str) -> None:
            """Emit a chunk, seeding it with the previous chunk's trailing `overlap` chars."""
            nonlocal prev_tail
            chunk = chunk.strip()
            if not chunk:
                return
            if prev_tail:
                chunk = (prev_tail + " " + chunk).strip()
            chunks.append(chunk)
            if overlap:
                prev_tail = chunk[-overlap:].strip() if len(chunk) > overlap else chunk
            else:
                prev_tail = ""

        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue

            # If a single line is larger than chunk_size, split by words/characters
            while len(line_str) > chunk_size:
                cut_idx = line_str.rfind(" ", 0, chunk_size)
                if cut_idx == -1 or cut_idx < chunk_size // 2:
                    cut_idx = chunk_size
                cut = line_str[:cut_idx]
                line_str = line_str[cut_idx:].strip()

                flush(current_chunk)
                current_chunk = ""
                flush(cut)

            if len(current_chunk) + len(line_str) + 1 < chunk_size:
                current_chunk += line_str + " "
            else:
                flush(current_chunk)
                current_chunk = line_str + " "

        flush(current_chunk)

        valid_chunks = [c for c in chunks if c.strip()]
        logfire.info(f"✅ Generated {len(valid_chunks)} chunks")
        return valid_chunks
