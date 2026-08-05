from typing import List
import logfire

def chunk_text(text: str, chunk_size: int = 1500, overlap: int = 200) -> List[str]:
    """
    Robust chunker that splits text by paragraphs and lines.
    Ensures no chunk exceeds `chunk_size` characters even if paragraphs lack double-newlines.
    """
    with logfire.span("✂️ Text Chunking", text_length=len(text)):
        if not text.strip(): 
            return []
            
        lines = text.split("\n")
        chunks = []
        current_chunk = ""
        
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
                
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                chunks.append(cut.strip())

            if len(current_chunk) + len(line_str) + 1 < chunk_size:
                current_chunk += line_str + " "
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = line_str + " "
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
            
        valid_chunks = [c for c in chunks if c.strip()]
        logfire.info(f"✅ Generated {len(valid_chunks)} chunks")
        return valid_chunks
