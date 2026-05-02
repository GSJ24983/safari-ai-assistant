# ingest.py
# ============================================================
# Reads your PDF manuals, breaks them into small chunks,
# and stores them locally using ChromaDB.
#
# RUN THIS ONCE before starting the app.
# How to run: open Terminal in VS Code, type:
#   python ingest.py
#
# A folder called chroma_db will be created automatically.
# ============================================================

import os
import shutil
from pathlib import Path
from dotenv import load_dotenv
import PyPDF2
import chromadb
from chromadb.utils import embedding_functions

load_dotenv()

# ── Settings ──────────────────────────────────────────────────
CHUNK_SIZE    = 400   # words per chunk
CHUNK_OVERLAP = 40    # overlapping words between chunks
DB_PATH       = "./chroma_db"   # folder where data is stored

# ── PDF files to process ──────────────────────────────────────
PDF_FILES = [
    {"file": "service_manual.pdf",      "label": "Service Manual",            "type": "pdf"},
    {"file": "infotainment_manual.pdf", "label": "Infotainment Manual",       "type": "pdf"},
    {"file": "ready_reckoner.txt",      "label": "Ready Reckoner by users",   "type": "txt"},
]

# ── Connect to ChromaDB ───────────────────────────────────────
def get_collection(reset=False):
    """
    Opens (or creates) the local database.
    Like opening a filing cabinet on your laptop.
    """
    if reset and Path(DB_PATH).exists():
        print("🗑️  Clearing old database...")
        shutil.rmtree(DB_PATH)
        print("   ✅ Cleared\n")

    client = chromadb.PersistentClient(path=DB_PATH)

    # Use a simple sentence-transformer model for embeddings
    # This runs locally — no API calls, no cost
    ef = embedding_functions.DefaultEmbeddingFunction()

    collection = client.get_or_create_collection(
        name="safari_manual",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"}
    )
    return collection

# ── Extract text from PDF ─────────────────────────────────────
def extract_text(pdf_path):
    """
    Reads a PDF and returns text page by page.
    Like a person flipping through the manual and typing out each page.
    """
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
        print(f"   ❌ File not found: {pdf_path}")
        print(f"      Make sure the PDF is in your safari-assistant folder")
        print(f"      and named exactly: {pdf_path}")
        return []
def extract_text_file(txt_path):
    print(f"📖 Reading {txt_path}...")
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            content = f.read()
        # Split on --- dividers so each warning light is its own "page"
        sections = [s.strip() for s in content.split("---") if s.strip()]
        pages = [{"text": s, "page": i+1} for i, s in enumerate(sections)]
        print(f"   ✅ Read {len(pages)} sections")
        return pages
    except FileNotFoundError:
        print(f"   ❌ File not found: {txt_path}")
        return []
# ── Split text into chunks ────────────────────────────────────
def make_chunks(pages, label):
    """
    Breaks pages into bite-sized chunks of ~400 words.
    Like cutting a long article into index cards.
    The AI can only read a few cards at a time,
    so smaller cards = more precise answers.
    """
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
                    "chunk_id": len(chunks)
                })
                # Keep last N words for overlap (context continuity)
                buffer = buffer[-CHUNK_OVERLAP:]
                start_page = page_data["page"]

    # Don't lose the final remaining words
    if len(buffer) > 20:
        chunks.append({
            "text": " ".join(buffer),
            "source": label,
            "page": start_page,
            "chunk_id": len(chunks)
        })

    print(f"   📦 Created {len(chunks)} chunks")
    return chunks

# ── Store chunks in ChromaDB ──────────────────────────────────
def store_chunks(collection, chunks, label):
    """
    Saves each chunk into your local database.
    ChromaDB automatically creates a 'fingerprint' for each chunk
    so it can find relevant ones later when someone asks a question.
    """
    print(f"💾 Storing {len(chunks)} chunks...")

    # ChromaDB works best in batches of 50
    batch_size = 50
    stored = 0

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]

        ids       = [f"{label}_{c['chunk_id']}" for c in batch]
        texts     = [c["text"] for c in batch]
        metadatas = [{"source": c["source"], "page": c["page"]} for c in batch]

        collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas
        )

        stored += len(batch)
        progress = int((stored / len(chunks)) * 30)
        bar = "█" * progress + "░" * (30 - progress)
        print(f"   [{bar}] {stored}/{len(chunks)}", end="\r")

    print(f"\n   ✅ Stored {stored} chunks from {label}")
    return stored

# ── MAIN ──────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("🚗 TATA SAFARI ASSISTANT — PDF INGESTION")
    print("   (ChromaDB local storage)")
    print("=" * 60)

    # Check PDFs exist before doing anything
    missing = [p["file"] for p in PDF_FILES if not Path(p["file"]).exists()]
    if missing:
        print("\n❌ These PDF files are missing from your folder:")
        for f in missing:
            print(f"   → {f}")
        print("\nPlease add them and try again.")
        print("The files must be in the same folder as ingest.py")
        return

    print(f"\n✅ Found all PDF files\n")

    # Ask if they want to start fresh
    ans = input("Start fresh (delete old data)? y/n: ").strip().lower()
    collection = get_collection(reset=(ans == "y"))

    total_stored = 0

    for pdf_info in PDF_FILES:
        print(f"\n{'─' * 40}")
        print(f"Processing: {pdf_info['label']}")
        print(f"{'─' * 40}")

        if pdf_info.get("type") == "txt":
            pages = extract_text_file(pdf_info["file"])
        else:
            pages = extract_text(pdf_info["file"])
        if not pages:
            continue

        chunks = make_chunks(pages, pdf_info["label"])
        stored = store_chunks(collection, chunks, pdf_info["label"])
        total_stored += stored

    # Final summary
    print("\n" + "=" * 60)
    print(f"✅ INGESTION COMPLETE!")
    print(f"   Total chunks stored: {total_stored}")
    print(f"   Location: {DB_PATH}/ folder")
    print(f"\n👉 Next step — run the app:")
    print(f"   streamlit run app.py")
    print("=" * 60)

if __name__ == "__main__":
    main()