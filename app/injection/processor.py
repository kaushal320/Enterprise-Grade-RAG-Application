import json
import os
import pickle
import sys
import uuid

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import logfire
from qdrant_client import QdrantClient
from qdrant_client.http import models
from rank_bm25 import BM25Okapi

from app.config import settings
from app.injection.chunking.splitter import chunk_text
from app.injection.loaders.html import parse_html
from app.injection.loaders.pdf import parse_pdf
from app.injection.loaders.text import parse_text
from app.services.retrieval.jina_embedding import embed_texts, get_embedding_dim
from app.services.retrieval.qdrant_service import BM25_INDEX_PATH, tokenize

logfire.configure(service_name="enterprise-ingestion-service")

# Local folder where parsed + chunked JSON metadata is saved
PROCESSED_DATA_DIR = "processed_data"

# Initialize Qdrant Client
qdrant_client = QdrantClient(
    url=settings.QDRANT_URL,
    api_key=settings.QDRANT_API_KEY,
)


def save_processed_locally(data: dict, source_type: str, filename: str) -> str:
    """Save parsed chunk metadata as JSON in processed_data/<source_type>/."""
    folder = os.path.join(PROCESSED_DATA_DIR, source_type)
    os.makedirs(folder, exist_ok=True)
    dest = os.path.join(folder, f"{filename}.json")
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return dest


def build_bm25_index():
    """
    Rebuild a persistent BM25 lexical index over every locally saved chunk and
    pickle it to BM25_INDEX_PATH, keeping it in sync with the Qdrant collection
    after ingestion. Hybrid retrieval fuses these hits with vector hits via RRF.
    """
    with logfire.span("Building BM25 Index"):
        if not os.path.isdir(PROCESSED_DATA_DIR):
            logfire.warning("No processed data found; skipping BM25 index build.")
            return

        corpus = []
        for root, _, files in os.walk(PROCESSED_DATA_DIR):
            for name in sorted(files):
                if not name.endswith(".json"):
                    continue
                path = os.path.join(root, name)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except (json.JSONDecodeError, OSError) as e:
                    logfire.warning(f"Could not read {path}: {e}")
                    continue
                source = data.get("filename", name)
                source_type = data.get("source_type", "general")
                for chunk in data.get("chunks", []):
                    if chunk and chunk.strip():
                        corpus.append({
                            "text": chunk,
                            "source": source,
                            "source_type": source_type,
                        })

        if not corpus:
            logfire.warning("No chunks to index; skipping BM25 index build.")
            return

        tokenized_corpus = [tokenize(chunk["text"]) for chunk in corpus]
        bm25 = BM25Okapi(tokenized_corpus)

        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        with open(BM25_INDEX_PATH, "wb") as f:
            pickle.dump(
                {"index": bm25, "chunks": corpus},
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        logfire.info(f"Built BM25 index over {len(corpus)} chunks -> {BM25_INDEX_PATH}")


def process_file(file_path: str, filename: str, source_type: str):
    """Parse -> chunk -> save locally -> embed -> index in Qdrant."""
    with logfire.span("Processing File", file=filename, source=source_type):
        try:
            # 1. Extract text based on file extension
            ext = filename.lower().rsplit(".", 1)[-1]
            if ext == "pdf":
                full_text = parse_pdf(file_path)
            elif ext in ("html", "htm"):
                full_text = parse_html(file_path)
            elif ext == "txt":
                full_text = parse_text(file_path)
            elif ext in ("docx", "pptx"):
                from app.injection.loaders.office import parse_office
                full_text = parse_office(file_path)
            else:
                logfire.warning(f"Skipping unsupported file type: {filename}")
                return

            if not full_text or not full_text.strip():
                logfire.warning(f"No text extracted from {filename} - skipping.")
                return

            # 2. Chunk text
            chunks = chunk_text(full_text)
            if not chunks:
                return

            # 3. Save processed metadata locally
            processed_data = {
                "filename": filename,
                "source_type": source_type,
                "chunks": chunks,
            }
            local_path = save_processed_locally(processed_data, source_type, filename)
            logfire.info(f"Saved processed data -> {local_path}")

            # 4. Embed and index in Qdrant
            with logfire.span("Vectorizing & Indexing"):
                embeddings = embed_texts(chunks)
                points = [
                    models.PointStruct(
                        id=str(uuid.uuid4()),
                        vector=vector,
                        payload={
                            "text": chunk,
                            "source": filename,
                            "source_type": source_type,
                        },
                    )
                    for chunk, vector in zip(chunks, embeddings)
                ]

                qdrant_client.upsert(
                    collection_name=settings.QDRANT_COLLECTION_NAME,
                    points=points,
                )
                logfire.info(f"Indexed {len(points)} points to Qdrant from {filename}.")

        except Exception as e:
            logfire.error(f"Failed to process {filename}: {e}")


def process_directory(dir_path: str, source_type: str):
    """Process every file in a directory."""
    with logfire.span("Scanning Directory", path=dir_path, source=source_type):
        files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
        logfire.info(f"Found {len(files)} files in {dir_path}.")
        for filename in files:
            process_file(os.path.join(dir_path, filename), filename, source_type)


def run_universal_ingestion(base_dir: str, explicit_source_type: str = None, wipe: bool = False):
    """
    Scan base_dir, map sub-folders to source types, and ingest all documents.
    Pass --wipe to drop and recreate the Qdrant collection before ingestion.
    """
    with logfire.span("Universal Ingestion Started", base_directory=base_dir):

        # Wipe collection if requested
        if wipe:
            with logfire.span("Wiping Collection"):
                if qdrant_client.collection_exists(settings.QDRANT_COLLECTION_NAME):
                    qdrant_client.delete_collection(settings.QDRANT_COLLECTION_NAME)
                    logfire.info(f"Collection '{settings.QDRANT_COLLECTION_NAME}' deleted.")

        # Recreate collection - dimension resolved at runtime after embedding model probe
        if not qdrant_client.collection_exists(settings.QDRANT_COLLECTION_NAME):
            dim = get_embedding_dim()
            qdrant_client.create_collection(
                collection_name=settings.QDRANT_COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=dim,
                    distance=models.Distance.COSINE,
                ),
            )
            logfire.info(
                f"Created collection '{settings.QDRANT_COLLECTION_NAME}' "
                f"({dim}-dim, Cosine)."
            )

        # Route to sub-folders or treat the whole dir as one source
        subdirs = [
            d for d in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, d))
        ]

        if not subdirs:
            if explicit_source_type:
                source_type = explicit_source_type
            else:
                base_name = os.path.basename(os.path.normpath(base_dir)).lower()
                source_type = (
                    "true" if "true" in base_name
                    else "noisy" if "noisy" in base_name
                    else "general"
                )
            logfire.info(f"No sub-folders found - processing '{base_dir}' as '{source_type}'.")
            process_directory(base_dir, source_type)
        else:
            for subdir in subdirs:
                source_type = (
                    "true" if "true" in subdir.lower()
                    else "noisy" if "noisy" in subdir.lower()
                    else subdir
                )
                process_directory(os.path.join(base_dir, subdir), source_type)

        # Keep the lexical index in sync with the freshly upserted vectors
        build_bm25_index()


if __name__ == "__main__":
    # Usage:
    #   python -m app.injection.processor DATA --wipe
    #   python -m app.injection.processor DATA/true_data true
    wipe_requested = "--wipe" in sys.argv
    clean_args = [a for a in sys.argv if a != "--wipe"]
 
    target_dir = clean_args[1] if len(clean_args) > 1 else "DATA"
    explicit_type = clean_args[2] if len(clean_args) > 2 else None

    if not os.path.exists(target_dir):
        print(f"Error: path '{target_dir}' does not exist.")
        sys.exit(1)

    run_universal_ingestion(target_dir, explicit_source_type=explicit_type, wipe=wipe_requested)
    try:
        qdrant_client.close()
    except Exception:
        pass
    logfire.info("Ingestion job completed.")
