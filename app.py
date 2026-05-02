# app.py — GEMINI VERSION
# ============================================================
# Tata Safari AI Assistant
# Uses: ChromaDB (local search) + Google Gemini (free AI brain)
# ============================================================

import os
from dotenv import load_dotenv
import streamlit as st
from google import genai
from google.genai import types
import chromadb
from chromadb.utils import embedding_functions

load_dotenv()

# ── Page setup ────────────────────────────────────────────────
st.set_page_config(
    page_title="Safari AI Assistant",
    page_icon="🚗",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ── Connect to services (runs once per session) ───────────────
@st.cache_resource
def load_services():
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return None, None, "no_key"

    if not os.path.exists("./chroma_db"):
        return None, None, "no_db"

    try:
        gemini_client = genai.Client(api_key=api_key)
        ef  = embedding_functions.DefaultEmbeddingFunction()
        db  = chromadb.PersistentClient(path="./chroma_db")
        col = db.get_collection(name="safari_manual", embedding_function=ef)
        return gemini_client, col, "ok"
    except Exception as e:
        return None, None, f"error: {e}"

gemini_client, collection, status = load_services()

# ── Error guards ──────────────────────────────────────────────
if status == "no_key":
    st.error("⚠️ Gemini API key not found.")
    st.info("Add GEMINI_API_KEY=your-key to your .env file and restart.")
    st.stop()
if status == "no_db":
    st.error("⚠️ Manual database not found.")
    st.info("Please run `python ingest.py` first.")
    st.stop()
if status.startswith("error"):
    st.error(f"⚠️ Startup error: {status}")
    st.stop()

# ── Session state ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Search ChromaDB for relevant manual pages ─────────────────
def search_manual(query: str) -> str:
    try:
        results = collection.query(query_texts=[query], n_results=5)
        if not results["documents"] or not results["documents"][0]:
            return ""
        parts = []
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            src  = meta.get("source", "Manual")
            page = meta.get("page", "?")
            parts.append(f"[{src} — Page {page}]\n{doc}")
        return "\n\n---\n\n".join(parts)
    except Exception:
        return ""

# ── Ask Gemini ────────────────────────────────────────────────
def ask_gemini(question: str, context: str,
               img_bytes: bytes = None, img_type: str = None) -> str:

    system_instruction = """You are the unofficial Tata Safari AI Assistant.
Answer questions using the content provided. As of Apr 2026, this content is based on the official Tata Safari service manual and infotainment manual.

Rules:
- If a dashboard image is provided: identify ALL visible warning lights and error indicators
  in the image. List EVERY error or warning you can see — do not stop after the first one.
- Do not consider normal indicators like turn signals, high beam, or parking brake, auto hold, etc. Focus ONLY on warning lights and error indicators that suggest a problem or required action.
- For EACH warning light found, explain: what it means AND what the owner should do.
- Use bullet points for each warning light. Complete every bullet fully before moving on.
- Put **safety warnings in bold**.
- If a specific light's meaning is not in the context provided, say for that item:
  "Meaning not found in manual. Please contact your Tata authorised service centre."
- Never guess or make up information.
- Always complete your full response. Never stop mid-sentence or mid-list.
- Be clear, practical and helpful."""

    if img_bytes:
        text_part  = types.Part(text=
            f'A Tata Safari owner uploaded this dashboard image and asks: "{question}"\n\n'
            f'Relevant manual sections:\n\n{context or "None found."}\n\n'
            f'Please identify what you see in the image (warning light, error indicator, error) '
            f'and advise the owner based on the manual content.'
            f'IMPORTANT: Identify and explain ALL warning lights and error indicators visible in the image. '
            f'Complete your full response — do not stop partway through the list.'
        )
        image_part = types.Part.from_bytes(data=img_bytes, mime_type=img_type)
        contents   = [text_part, image_part]
    else:
        contents = (
            f'Question from a Tata Safari owner: "{question}"\n\n'
            f'Relevant manual sections:\n\n{context or "None found."}\n\n'
            f'Please answer based on the manual content above.'
        )

    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            max_output_tokens=800 if not img_bytes else 4000,
            temperature=0.2
        ),
        contents=contents
    )

    return response.text

# ── Page header ───────────────────────────────────────────────
st.title("🚗 Tata Safari AI Assistant")
st.caption(
    "Ask anything about your Safari — powered by the official "
    "service manual and infotainment manual."
)
st.divider()

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 💡 Sample questions")
    for q in [
        "What does the orange triangle warning mean?",
        "How do I reset the TPMS?",
        "What is the engine oil change interval?",
        "How to connect Android Auto?",
        "What does the check engine light mean?",
        "How do I activate cruise control?",
    ]:
        st.caption(f"→ {q}")
    st.divider()
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ── Image upload ──────────────────────────────────────────────
uploaded = st.file_uploader(
    "📷 Upload a dashboard photo (optional)",
    type=["jpg", "jpeg", "png", "webp"]
)
if uploaded:
    st.image(uploaded, caption="Your uploaded image", width=250)

# ── Show existing chat history ────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat input ────────────────────────────────────────────────
question = st.chat_input("Ask a question about your Tata Safari...")

# ── Process question ──────────────────────────────────────────
if question:

    # Read image bytes if uploaded
    img_bytes, img_type = None, None
    if uploaded is not None:
        try:
            img_bytes = uploaded.getvalue()
            ext_map   = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
                         "png": "image/png",  "webp": "image/webp"}
            ext       = uploaded.name.rsplit(".", 1)[-1].lower()
            img_type  = ext_map.get(ext, "image/jpeg")
        except Exception as e:
            st.warning(f"Could not read image: {e}")
            img_bytes = None

    # Display text for user bubble
    display = f"📷 *[Image attached]*\n\n{question}" if uploaded else question

    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": display})
    with st.chat_message("user"):
        st.markdown(display)

    # Get answer from Gemini
    with st.chat_message("assistant"):
        with st.spinner("Searching manual..."):
            try:
                if img_bytes:
                    # Broaden search for image queries to catch warning light sections
                    image_search_query = question + " warning light indicator dashboard tell tales error"
                    context = search_manual(image_search_query)
                else:
                    context = search_manual(question)
                answer  = ask_gemini(question, context, img_bytes, img_type)
                st.markdown(answer)
                st.caption("📖 Source: Tata Safari Official Manual · Beta")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })
            except Exception as e:
                st.error(f"Something went wrong: {e}")
                st.info("Check the VS Code terminal for details.")
