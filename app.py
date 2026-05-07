# app.py — GEMINI VERSION (numpy search + Gemini answers)
# ============================================================
# Tata Safari AI Assistant
# Fixes applied:
#   1. Retrieval accuracy — multi-query expansion + n_results=8 + dedup
#   2. Gemini fallback — retry with gemini-2.0-flash-001 on failure
#   3. Demo limit caption fixed to 3 (not 999)
#   4. Token display removed from UI (still logged to Make.com)
#   5. Data sources shown in sidebar
#   6. Make.com webhook logging preserved (question + answer + feedback)
# ============================================================

import os
import subprocess
import base64
import pickle
import numpy as np
from dotenv import load_dotenv
import threading
import requests

load_dotenv()

# ── Feedback logging (Make.com webhook) ──────────────────────
def log_to_webhook(question: str, answer: str, had_image: bool,
                   user: str = "unknown", input_tokens: int = 0,
                   output_tokens: int = 0, feedback: str = None):
    """Fires and forgets — logs Q&A to Make.com without slowing the app."""
    webhook_url = os.environ.get("N8N_WEBHOOK_URL", "")
    if not webhook_url:
        return  # silently skip if not configured

    payload = {
        "timestamp": __import__("datetime").datetime.now().isoformat(),
        "user": user,
        "question": question,
        "answer": answer[:500],        # first 500 chars — enough for analysis
        "had_image": had_image,
        "answer_length": len(answer),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "feedback": feedback           # None for normal logs, "helpful"/"unhelpful" for thumbs
    }

    try:
        threading.Thread(
            target=requests.post,
            args=(webhook_url,),
            kwargs={"json": payload, "timeout": 5}
        ).start()
    except Exception:
        pass  # never crash the app over logging

# ── Auto-ingest on first boot (for Streamlit Cloud) ───────────
if not os.path.exists("./safari_db.pkl"):
    import streamlit as st
    st.info("⏳ First-time setup: building manual database... (takes ~2 min)")
    subprocess.run(["python", "ingest.py"], check=True)
    st.rerun()

import streamlit as st
from google import genai
from google.genai import types

# ── Page setup ────────────────────────────────────────────────
st.set_page_config(
    page_title="Safari AI Assistant",
    page_icon="🚗",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ── Access gate (per-user codes) ──────────────────────────────
def check_access():
    """
    Validates per-user access codes from Streamlit secrets.
    Returns the user's label (e.g. 'beta_01') or stops the app.
    """
    try:
        valid_codes = dict(st.secrets["access_codes"])
    except Exception:
        st.error("Access codes not configured. Contact Gaurav.")
        st.stop()

    if st.session_state.get("authenticated_user"):
        return st.session_state["authenticated_user"]

    st.title("🚗 Tata Safari AI Assistant")
    st.markdown("---")
    code = st.text_input(
        "🔐 Enter your access code",
        type="password",
        placeholder="Enter the code shared with you"
    )

    if st.button("Access Assistant"):
        matched_user = next(
            (label for label, val in valid_codes.items() if val == code),
            None
        )
        if matched_user:
            st.session_state["authenticated_user"] = matched_user
            st.session_state["question_count"] = 0
            st.rerun()
        else:
            st.error("Invalid code. Contact Gaurav on WhatsApp for access.")
            st.stop()
    else:
        st.stop()

current_user = check_access()

# ── Per-user question limits ──────────────────────────────────
QUESTION_LIMITS = {
    "gaurav":  999,   # owner — unlimited
    "default": 3      # all beta users — 3 questions per session
}

def get_limit(user_label: str) -> int:
    return QUESTION_LIMITS.get(user_label, QUESTION_LIMITS["default"])

if "question_count" not in st.session_state:
    st.session_state.question_count = 0

limit = get_limit(current_user)
if st.session_state.question_count >= limit:
    st.warning(f"⚠️ You've reached the {limit}-question demo limit. Contact Gaurav for more access!")
    st.stop()

# ── Load vector DB and Gemini (runs once per session) ─────────
@st.cache_resource
def load_services():
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return None, None, None, "no_key"

    if not os.path.exists("./safari_db.pkl"):
        return None, None, None, "no_db"

    try:
        from sentence_transformers import SentenceTransformer
        gemini_client = genai.Client(api_key=api_key)

        with open("./safari_db.pkl", "rb") as f:
            db = pickle.load(f)

        model = SentenceTransformer("all-MiniLM-L6-v2")
        return gemini_client, db, model, "ok"
    except Exception as e:
        return None, None, None, f"error: {e}"

gemini_client, db, embed_model, status = load_services()

# ── Error guards ──────────────────────────────────────────────
if status == "no_key":
    st.error("⚠️ Gemini API key not found.")
    st.info("Add GEMINI_API_KEY to Streamlit Cloud secrets.")
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

# ── FIX 1: Multi-query search with deduplication ─────────────
# Problem: Single query often misses relevant chunks because
# the user's phrasing doesn't match the manual's exact wording.
# Solution: Expand each question into multiple search queries,
# search for each, then deduplicate and return the best N chunks.

def _expand_queries(question: str, is_image: bool) -> list[str]:
    """
    Returns a list of search queries derived from the user's question.
    More queries = higher chance of finding the right chunk.
    """
    q = question.lower().strip()
    queries = [question]  # always include the original

    # For image uploads, add a broad warning-light sweep
    if is_image:
        queries += [
            question + " warning light indicator dashboard tell tales error",
            "dashboard warning light meaning",
            "instrument cluster warning symbols",
        ]

    # Keyword expansions for common query patterns
    expansions = {
        "cruise":        ["cruise control lamp green", "cruise control indicator dashboard", "cruise control activate procedure", "cruise control symbol meaning"],
        "spanner":       ["spanner sign warning", "service reminder indicator", "maintenance warning lamp",
                          "car with spanner warning light"],
        "service":       ["service reminder", "service due indicator", "maintenance interval"],
        "oil":           ["engine oil warning", "oil pressure indicator", "oil change interval"],
        "tpms":          ["tyre pressure warning", "TPMS reset", "tyre pressure monitoring"],
        "abs":           ["ABS indicator", "anti-lock braking warning"],
        "engine":        ["check engine light", "engine malfunction indicator"],
        "battery":       ["battery warning lamp", "charging system indicator"],
        "brake":         ["brake warning light", "brake fluid indicator"],
        "temperature":   ["coolant temperature warning", "engine overheat indicator"],
        "airbag":        ["airbag warning lamp", "SRS indicator"],
        "android auto":  ["android auto connection", "infotainment connectivity"],
        "apple carplay": ["carplay connection", "infotainment apple"],
        "bluetooth":     ["bluetooth pairing", "phone connection infotainment"],
    }

    for keyword, alts in expansions.items():
        if keyword in q:
            queries += alts
            break  # one expansion set per query is enough

    return queries


def search_manual(query: str, n_results: int = 8,
                  is_image: bool = False) -> str:
    """
    Improved search: runs multiple query variants, merges results,
    deduplicates by chunk index, and returns the top N by score.
    Increased n_results from 5 → 8 to cast a wider net.
    """
    try:
        all_queries   = _expand_queries(query, is_image)
        embeddings    = db["embeddings"]
        chunks        = db["chunks"]

        seen_indices  = {}   # chunk_index → best_score

        for q in all_queries:
            query_vec = embed_model.encode([q])[0]
            norms     = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_vec)
            norms     = np.where(norms == 0, 1e-10, norms)
            scores    = np.dot(embeddings, query_vec) / norms

            top_indices = np.argsort(scores)[::-1][:n_results]
            for idx in top_indices:
                score = float(scores[idx])
                # Keep the best score seen for each chunk across all queries
                if idx not in seen_indices or score > seen_indices[idx]:
                    seen_indices[idx] = score

        # Sort all discovered chunks by best score, take top N
        sorted_chunks = sorted(seen_indices.items(), key=lambda x: x[1], reverse=True)[:n_results]

        parts = []
        for idx, score in sorted_chunks:
            chunk = chunks[idx]
            parts.append(f"[{chunk['source']} — Page {chunk['page']}]\n{chunk['text']}")

        return "\n\n---\n\n".join(parts)

    except Exception:
        return ""


# ── FIX 2: Ask Gemini with fallback model ────────────────────
# Primary model: gemini-2.5-flash (best quality)
# Fallback model: gemini-2.0-flash-001 (used if primary fails)
# This handles quota errors, 429s, and model unavailability.

PRIMARY_MODEL  = "gemini-2.5-flash"
FALLBACK_MODEL = "gemini-2.0-flash-001"

def _call_gemini_model(model_name: str, system_instruction: str,
                        contents, max_tokens: int):
    """Single Gemini call. Raises on failure."""
    response = gemini_client.models.generate_content(
        model=model_name,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            max_output_tokens=max_tokens,
            temperature=0.2
        ),
        contents=contents
    )
    return response


def ask_gemini(question: str, context: str,
               img_bytes: bytes = None, img_type: str = None):
    """
    Returns (answer_text, input_tokens, output_tokens, model_used).
    Tries PRIMARY_MODEL first. On any exception, retries with FALLBACK_MODEL.
    """

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
- IMPORTANT: If the manual does not contain step-by-step procedure for something,
  but DOES contain related information (indicator lamps, warnings, specs),
  share ALL of that related information first. Do not say "not found" when
  partially relevant content exists. Say "The manual does not describe the full
  procedure, but here is what it does say:" and then share everything relevant.
- Be clear, practical and helpful."""

    max_tokens = 3000 if not img_bytes else 4000

    if img_bytes:
        text_part  = types.Part(text=
            f'A Tata Safari owner uploaded this dashboard image and asks: "{question}"\n\n'
            f'Relevant manual sections:\n\n{context or "None found."}\n\n'
            f'Please identify what you see in the image (warning light, error indicator, error) '
            f'and advise the owner based on the manual content. '
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

    model_used = PRIMARY_MODEL
    try:
        response = _call_gemini_model(PRIMARY_MODEL, system_instruction, contents, max_tokens)
    except Exception as primary_err:
        # Fallback to secondary model
        try:
            model_used = FALLBACK_MODEL
            response   = _call_gemini_model(FALLBACK_MODEL, system_instruction, contents, max_tokens)
        except Exception as fallback_err:
            # Both models failed — return a user-friendly error message
            return (
                "⚠️ The AI service is temporarily unavailable. "
                "Please try again in a moment. If the problem persists, "
                "contact Gaurav.",
                0, 0,
                "error"
            )

    try:
        input_tokens  = response.usage_metadata.prompt_token_count or 0
        output_tokens = response.usage_metadata.candidates_token_count or 0
    except Exception:
        input_tokens, output_tokens = 0, 0

    return response.text, input_tokens, output_tokens, model_used


# ── Page header ───────────────────────────────────────────────
st.title("🚗 Tata Safari AI Assistant")

# FIX 3: Caption shows correct limit (pulled from QUESTION_LIMITS, not hardcoded)
questions_left = limit - st.session_state.question_count
st.caption(
    f"Ask anything about your Safari — powered by official manuals. "
    f"Demo: {questions_left} question{'s' if questions_left != 1 else ''} remaining. "
    f"Contact Gaurav for full access."
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

    # FIX 5: Show data sources so users know what the assistant knows
    st.markdown("### 📚 Knowledge Sources")
    st.markdown(
        """
The assistant answers from these official documents:

- 📘 **Tata Safari Service Manual**  
  Engine, dashboard, warning lights, maintenance schedules, fluids

- 📗 **Tata Safari Infotainment Manual**  
  Audio system, Android Auto, Apple CarPlay, Bluetooth, navigation

- 📙 **Tata Safari Ready Reckoner** *(if ingested)*  
  Quick-reference specs and community tips

*Answers are limited to content in these documents. For issues not covered, please visit a Tata authorised service centre.*
"""
    )

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
                context = search_manual(question, is_image=bool(img_bytes))

                answer, input_tokens, output_tokens, model_used = ask_gemini(
                    question, context, img_bytes, img_type
                )

                st.markdown(answer)

                # FIX 4: No token count shown. Clean source caption only.
                fallback_note = " *(fallback model)*" if model_used == FALLBACK_MODEL else ""
                st.caption(f"📖 Source: Tata Safari Official Manual · Beta{fallback_note}")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

                # Store last Q&A for thumbs feedback
                st.session_state["last_question"]     = question
                st.session_state["last_answer"]       = answer
                st.session_state["last_input_tokens"] = input_tokens
                st.session_state["last_output_tokens"]= output_tokens
                st.session_state["last_had_image"]    = bool(img_bytes)
                st.session_state["feedback_given"]    = False

                # Log to Make.com webhook in background
                # Webhook payload includes tokens for your Excel tracking
                # even though they are no longer shown in the UI
                log_to_webhook(
                    question, answer,
                    had_image=bool(img_bytes),
                    user=current_user,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens
                )

                # Increment question counter after successful answer
                st.session_state.question_count += 1

            except Exception as e:
                st.error(f"Something went wrong: {e}")
                st.info("Check the VS Code terminal for details.")

# ── Thumbs feedback (shown after last answer) ─────────────────
if (st.session_state.get("last_question")
        and not st.session_state.get("feedback_given", True)):

    st.markdown("**Was this answer helpful?**")
    col1, col2, col3 = st.columns([1, 1, 8])

    with col1:
        if st.button("👍", key="thumbs_up"):
            log_to_webhook(
                question=st.session_state["last_question"],
                answer=st.session_state["last_answer"],
                had_image=st.session_state["last_had_image"],
                user=current_user,
                input_tokens=st.session_state["last_input_tokens"],
                output_tokens=st.session_state["last_output_tokens"],
                feedback="helpful"
            )
            st.session_state["feedback_given"] = True
            st.toast("Thanks! 👍")
            st.rerun()

    with col2:
        if st.button("👎", key="thumbs_down"):
            log_to_webhook(
                question=st.session_state["last_question"],
                answer=st.session_state["last_answer"],
                had_image=st.session_state["last_had_image"],
                user=current_user,
                input_tokens=st.session_state["last_input_tokens"],
                output_tokens=st.session_state["last_output_tokens"],
                feedback="unhelpful"
            )
            st.session_state["feedback_given"] = True
            st.toast("Noted — we'll improve this. 👎")
            st.rerun()
