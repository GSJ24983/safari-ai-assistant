# ingest.py — NO CHROMADB VERSION
# ============================================================
# Reads your PDF manuals, breaks them into chunks,
# creates embeddings using sentence-transformers,
# and saves everything to a simple .pkl file.
#
# RUN THIS ONCE before starting the app.
# How to run:
#   python ingest.py
#
# Creates: safari_db.pkl  (replaces chroma_db/ folder)
# ============================================================

import os
import pickle
from pathlib import Path
from dotenv import load_dotenv
import PyPDF2
import numpy as np
from sentence_transformers import SentenceTransformer

load_dotenv()

# ── Settings ──────────────────────────────────────────────────
CHUNK_SIZE    = 400   # words per chunk
CHUNK_OVERLAP = 40    # overlapping words between chunks
DB_PATH       = "./safari_db.pkl"   # single file replaces chroma_db/

# ── PDF files to process ──────────────────────────────────────
PDF_FILES = [
    {"file": "service_manual.pdf",            "label": "Service Manual"},
    {"file": "infotainment_manual.pdf",        "label": "Infotainment Manual"},
    {"file": "Tata Safari - ready reckoner.pdf", "label": "Ready Reckoner by users"},
]

# ── Extract text from PDF ─────────────────────────────────────
def extract_text(pdf_path):
    print(f"📖 Reading {pdf_path}...")
    pages = []
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            total = len(reader.pages)
            print(f"   Found {total} pages")
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text and text.strip():
                    pages.append({"text": text.strip(), "page": i + 1})
        print(f"   ✅ Read {len(pages)} pages with text")
        return pages
    except FileNotFoundError:
        print(f"   ❌ File not found: {pdf_path} — skipping")
        return []

# ── Split text into chunks ────────────────────────────────────
def make_chunks(pages, label):
    chunks = []
    buffer = []
    start_page = 1

    for page_data in pages:
        words = page_data["text"].split()
        for word in words:
            buffer.append(word)
            if len(buffer) >= CHUNK_SIZE:
                chunks.append({
                    "text": " ".join(buffer),
                    "source": label,
                    "page": start_page,
                })
                buffer = buffer[-CHUNK_OVERLAP:]
                start_page = page_data["page"]

    if len(buffer) > 20:
        chunks.append({
            "text": " ".join(buffer),
            "source": label,
            "page": start_page,
        })

    print(f"   📦 Created {len(chunks)} chunks")
    return chunks

# ── MAIN ──────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("🚗 TATA SAFARI ASSISTANT — PDF INGESTION (numpy version)")
    print("=" * 60)

    # Collect chunks from available PDFs
    all_chunks = []
    for pdf_info in PDF_FILES:
        if not Path(pdf_info["file"]).exists():
            print(f"⚠️  Skipping missing file: {pdf_info['file']}")
            continue
        print(f"\n{'─' * 40}")
        print(f"Processing: {pdf_info['label']}")
        pages = extract_text(pdf_info["file"])
        if pages:
            chunks = make_chunks(pages, pdf_info["label"])
            all_chunks.extend(chunks)

    if not all_chunks:
        print("\n❌ No chunks created. Check your PDF files.")
        return

    print(f"\n🤖 Creating embeddings for {len(all_chunks)} chunks...")
    print("   (downloading model on first run — ~90MB, one time only)")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [c["text"] for c in all_chunks]
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)

    # Save everything to a single pickle file
    db = {
        "chunks": all_chunks,
        "embeddings": embeddings,   # numpy array, shape (N, 384)
        "model_name": "all-MiniLM-L6-v2"
    }
    with open(DB_PATH, "wb") as f:
        pickle.dump(db, f)

    print(f"\n{'=' * 60}")
    print(f"✅ INGESTION COMPLETE!")
    print(f"   Total chunks stored: {len(all_chunks)}")
    print(f"   Saved to: {DB_PATH}")
    print(f"\n👉 Next step — run the app:")
    print(f"   streamlit run app.py")
    print("=" * 60)

if __name__ == "__main__":
    main()
