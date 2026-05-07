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

    # ── Keyword expansions — full Safari/iRA vocabulary ──────────
    # Multiple keywords can match a single query (no break).
    # Ordered multi-word keys first so "android auto" matches before "auto".
    KEYWORD_EXPANSIONS = {

        # ═══════════════════════════════════════════════════════
        # SECTION 1: WARNING LIGHTS & TELL TALES
        # ═══════════════════════════════════════════════════════

        "cruise":        ["cruise control lamp green", "cruise control indicator dashboard",
                          "cruise control activate procedure", "cruise control symbol meaning",
                          "cruise ON set the speed message", "adaptive cruise control ACC indicator",
                          "ACC follow distance setting", "cruise control not working highway"],

        "spanner":       ["spanner sign warning", "service reminder indicator",
                          "maintenance warning lamp", "car with spanner warning light",
                          "wrench symbol dashboard meaning", "yellow spanner light stays on"],

        "service":       ["service reminder", "service due indicator", "maintenance interval",
                          "service overdue warning", "reset service light procedure",
                          "next service km display"],

        "oil":           ["engine oil warning", "oil pressure indicator", "oil change interval",
                          "check oil pressure voice alert", "low oil level warning red light",
                          "oil pressure lamp engine damage risk"],

        "tpms":          ["tyre pressure warning", "TPMS reset procedure", "tyre pressure monitoring",
                          "TPMS chime 4 seconds meaning", "TPMS fault 20 second chime",
                          "TPMS sensor missing indicator", "TPMS high temperature warning",
                          "TPMS air leakage alert", "check vehicle tyres voice alert",
                          "tyre pressure low all four tyres"],

        "abs":           ["ABS indicator amber light", "anti-lock braking warning",
                          "ABS light stays on amber", "ABS fault normal braking still works",
                          "ABS EBD malfunction together"],

        "engine":        ["check engine light MIL", "engine malfunction indicator lamp",
                          "MIL lamp solid vs flashing", "engine fault code diagnosis",
                          "check engine with oil pressure lamp together"],

        "battery":       ["battery warning lamp", "charging system indicator",
                          "please charge battery voice alert", "12V battery drain issue",
                          "smart key battery low warning", "alternator fault warning"],

        "brake":         ["brake warning light red", "brake fluid level low voice alert",
                          "park brake cum low brake fluid indicator", "EBD malfunction brake warning",
                          "release park brake voice alert", "brake fluid low amber red light"],

        "temperature":   ["coolant temperature warning red blink", "engine overheat indicator",
                          "high coolant temperature chime continuous", "coolant temp hazardous blink",
                          "engine overheat do not remove radiator cap", "coolant temperature amber then red"],

        "airbag":        ["airbag warning lamp SRS", "airbag indicator stays on after 4 seconds",
                          "passenger airbag PAB indicator roof", "SRS warning lamp blinking",
                          "airbag deactivated symbol meaning"],

        "water in fuel": ["water in fuel indicator amber", "water in fuel diesel warning",
                          "check water in fuel voice alert", "drain water fuel filter procedure",
                          "water in fuel injection system damage"],

        "fuel":          ["low fuel warning amber", "check fuel level voice alert",
                          "fuel system fault flashing light", "distance to empty fuel display",
                          "fuel level low refill immediately", "fuel gauge reading incorrect"],

        "epas":          ["EPAS fault indicator amber", "electric power steering warning",
                          "power steering malfunction symbol", "EPAS fault service center",
                          "steering heavy after EPAS warning"],

        "esp":           ["ESP off indicator amber", "electronic stability program warning",
                          "ESP off symbol dashboard meaning", "traction control ESP light",
                          "ESP malfunction indicator"],

        "ebd":           ["EBD malfunction warning", "electronic brake force distribution fault",
                          "EBD fault with brake warning together", "EBD indicator red light meaning"],

        "hhc":           ["HHC fault indicator amber", "hill hold control malfunction",
                          "hill hold not working warning", "HHC system fault stays on"],

        "hdc":           ["HDC ON lamp green meaning", "hill descent control indicator",
                          "hill descent enable disable procedure", "HDC speed setting"],

        "apb":           ["APB malfunction indicator", "automatic parking brake fault",
                          "APB warning stays 4 seconds", "electronic handbrake fault indicator",
                          "auto hold failure amber warning"],

        "auto hold":     ["automatic hold active green", "automatic hold failure amber",
                          "auto hold indicator white meaning", "AVH fault ESP system warning",
                          "auto hold switch location"],

        "drl":           ["DRL daytime running lamp indicator green", "DRL activate deactivate procedure",
                          "park lamp switch twice to toggle DRL", "daytime running light symbol"],

        "door ajar":     ["door ajar lamp red", "one of the doors open voice alert",
                          "driver door open please check voice", "tailgate open voice alert",
                          "door open indicator independent four doors", "boot open warning"],

        "seatbelt":      ["fasten driver seatbelt voice alert", "fasten passenger seatbelt alert",
                          "rear seatbelt reminder 60 seconds", "seatbelt warning 15kmph trigger",
                          "seatbelt telltale 90 second audio warning", "seatbelt buzzer how to stop"],

        "peps":          ["PEPS indication IGN green", "PEPS ACC amber indicator",
                          "smart key not found warning", "key not found voice alert",
                          "PEPS key battery low meaning", "keyless entry not working"],

        "transmission":  ["AT fault indicator amber", "AMT fault warning dashboard",
                          "transmission oil temperature high indicator", "gearbox fault warning light",
                          "transmission failure detected symbol", "limp mode AT fault indicator"],

        "amt":           ["AMT fault warning", "automated manual transmission indicator",
                          "AMT warning light dashboard", "AMT limp mode",
                          "press brake to start AMT reminder"],

        "press brake":   ["press brake clutch amber symbol", "press brake to start engine indicator",
                          "brake clutch press reminder ignition", "press clutch amber light AMT"],

        "urea":          ["low urea level warning amber", "DEF level low refill soon message",
                          "SCR system fault warning", "BSVI chime urea low continuous",
                          "urea refill procedure how to", "DEF refill 7 litres max procedure",
                          "SCR fault level 1 2 3 chime", "urea tank capacity 15 litres",
                          "emission system fault indicator", "AdBlue DEF indicator meaning India"],

        "scr":           ["SCR system fault warning", "urea DEF level low",
                          "BSVI emission system indicator", "SCR fault chime meaning"],

        "high beam assist": ["HBA indicator green dashboard", "high beam assist fault amber",
                             "high beam assist not working", "HBA blink fault frequency",
                             "auto high beam not switching off"],

        "rear fog":      ["rear fog lamp amber indicator", "fog lamp on warning symbol",
                          "rear fog lamp disable procedure"],

        "position lamp": ["position lamp indicator green meaning", "parking lamp on reminder",
                          "park lamp ON battery drain warning", "head lamp forgot to turn off chime"],

        # ═══════════════════════════════════════════════════════
        # SECTION 2: ADAS FEATURES
        # ═══════════════════════════════════════════════════════

        "fcw":           ["FCW forward collision warning indicator", "AEB automatic emergency braking symbol",
                          "forward collision warning false braking", "AEB too sensitive disable",
                          "FCW AEB not working after windshield repair", "front radar blocked warning"],

        "aeb":           ["AEB automatic emergency braking", "forward collision warning FCW",
                          "AEB disable procedure", "collision warning bumper damage recalibration"],

        "ldw":           ["LDW lane departure warning indicator", "lane departure beep while driving",
                          "LDW disable procedure", "lane departure warning sensitivity",
                          "lane departure false alert road markings", "LDW not working faded lane lines"],

        "tsr":           ["TSR traffic sign recognition indicator", "speed limit sign not detected",
                          "traffic sign recognition wrong speed display", "TSR Vienna convention signs only",
                          "TSR not working night rain", "speed limit blink warning meaning",
                          "TSR 10 metre detection zone limitation"],

        "blind spot":    ["BSD LCA blind spot detection indicator", "lane change alert ORVM warning light",
                          "blind spot warning mirror indicator", "BSD not working rear bumper damage",
                          "ORVM indicator failure rear ADAS message", "BSD disable procedure"],

        "bsd":           ["BSD blind spot detection indicator", "ORVM warning light lane change",
                          "blind spot sensor misaligned contact service", "BSD disable procedure"],

        "rcta":          ["RCTA rear cross traffic alert indicator", "reverse parking collision warning",
                          "cross traffic alert not working", "RCTA radar rear bumper",
                          "rear cross traffic alert enable disable"],

        "doa":           ["DOA door open alert indicator", "door open alert while reversing",
                          "door open warning when overtaking"],

        "ddoa":          ["DDOA driver doze off alert", "driver drowsiness detection warning",
                          "have a tea break voice alert meaning", "drowsy driver warning chime",
                          "DDOA level 1 2 3 chime stages", "fatigue alert steering behavior monitor",
                          "driver attention warning not camera based", "DDOA reset after break"],

        "drowsy":        ["DDOA driver drowsiness detection", "have a tea break voice alert",
                          "fatigue warning chime stages", "driver doze off alert meaning"],

        "adas":          ["ADAS sensor blocked warning", "ADAS false braking complex environment",
                          "ADAS not working after windshield replacement", "ADAS calibration after repair",
                          "ADAS bumper clean mud snow ice", "ADAS narrow road limitation",
                          "rear ADAS malfunction contact service"],

        # ═══════════════════════════════════════════════════════
        # SECTION 3: iRA INFOTAINMENT
        # ═══════════════════════════════════════════════════════

        "android auto":  ["android auto connection wireless", "android auto not connecting",
                          "wired android auto USB cable procedure", "android auto bluetooth pair first",
                          "android auto carplay simultaneous not possible", "android auto disconnect auto",
                          "google assistant mic android auto"],

        "apple carplay": ["carplay wireless connection", "carplay not detected iPhone",
                          "wired carplay USB connection procedure", "hey siri carplay activation",
                          "carplay and android auto same time not supported", "carplay disconnecting"],

        "carplay":       ["apple carplay connection iPhone", "carplay USB wireless procedure",
                          "carplay not working infotainment", "hey siri carplay"],

        "bluetooth":     ["bluetooth pairing procedure iRA", "pair new device infotainment",
                          "10 paired devices maximum iRA", "bluetooth audio AVRCP version",
                          "bluetooth auto connect on", "delete paired device infotainment"],

        "voice assistant": ["voice assistant activate procedure", "steering wheel mic button",
                            "native voice assistant commands", "voice assistant not responding",
                            "voice commands list what to say"],

        "alexa":         ["alexa not working network error", "alexa iRA infotainment setup",
                          "alexa connectivity issue vehicle", "alexa active esim required",
                          "built-in alexa car infotainment"],

        "drivenext":     ["DriveNext app fuel efficiency", "DriveNext trip score dashboard",
                          "DriveNext driving safety score", "DriveNext trip history",
                          "DriveNext fuel analysis"],

        "what3words":    ["what3words navigation infotainment", "what3words address car",
                          "three word location address iRA"],

        "aqi":           ["AQI air quality index display", "AQI level poor hazardous meaning",
                          "cabin air quality indicator", "AQI reading infotainment screen",
                          "air purification AQI good moderate bad"],

        "wireless charger": ["wireless charger not working", "wireless charging error icon",
                             "WPC wireless phone charger symbol", "Qi charging pad infotainment",
                             "charging error popup infotainment meaning"],

        "mood light":    ["mood light ambient settings", "mood light color change infotainment",
                          "ambient light enable disable", "mood light not working"],

        "jbl":           ["JBL mode infotainment", "JBL audio preset modes",
                          "JBL sound system modes selection", "JBL mode switch procedure"],

        "media":         ["USB media not playing", "pen drive format compatible infotainment",
                          "video play USB only infotainment", "AVRCP version bluetooth media issue",
                          "media source auto switch", "infotainment no audio source connected"],

        "radio":         ["FM AM radio infotainment", "auto store radio presets",
                          "auto tuning infotainment", "radio preset save procedure"],

        "steering controls": ["steering wheel control volume", "SWC accept call button",
                              "steering scroll next track", "steering wheel mic activate voice"],

        "quick access":  ["QAD quick access drawer infotainment", "swipe down top screen iRA",
                          "brightness control infotainment", "display off sleep mode iRA",
                          "park assist quick access shortcut"],

        "connectivity":  ["Wi-Fi infotainment settings", "connectivity settings iRA",
                          "software update infotainment settings", "phone settings infotainment"],

        # ═══════════════════════════════════════════════════════
        # SECTION 4: CONNECTED CAR & OTA
        # ═══════════════════════════════════════════════════════

        "ira":           ["iRA connected car service", "iRA app features vehicle",
                          "iRA subscription renewal esim", "iRA esim deactivated what to do",
                          "connected car services not working", "iRA 12 month esim activation"],

        "ress":          ["RESS remote engine start stop", "remote start engine iRA app",
                          "remote engine start not working", "RESS feature procedure"],

        "fota":          ["FOTA firmware over the air update", "OTA software update in progress",
                          "infotainment update notification", "system update how long"],

        "ecall":         ["E-call emergency call procedure", "B-call breakdown call feature",
                          "emergency call 5 second cancel", "TATA Asist breakdown call feature",
                          "ecall auto trigger crash", "roadside assistance response time"],

        "bcall":         ["B-call breakdown call feature", "TATA Asist roadside assistance",
                          "B-call switch dashboard location", "breakdown call procedure"],

        # ═══════════════════════════════════════════════════════
        # SECTION 5: DRIVE & TERRAIN MODES
        # ═══════════════════════════════════════════════════════

        "drive mode":    ["auto drive mode activated voice alert", "comfort drive mode indicator",
                          "dynamic drive mode symbol", "rough road mode activated meaning",
                          "city drive mode indicator", "sport drive mode activated",
                          "economy drive mode fuel saving", "drive mode selector location"],

        "terrain mode":  ["wet mode activated indicator", "mud ruts mode activated symbol",
                          "sand mode dashboard indicator", "grass snow mode meaning",
                          "terrain mode voice alert infotainment", "terrain selector procedure"],

        # ═══════════════════════════════════════════════════════
        # SECTION 6: CLIMATE CONTROL
        # ═══════════════════════════════════════════════════════

        "fatc":          ["FATC fully automatic temperature control", "dual zone climate control panel",
                          "FATC auto mode not cooling", "express cooling on off voice alert",
                          "cabin air purification FATC", "rear AC zone climate control"],

        "ac":            ["AC compressor fault warning", "climate control symbol dashboard",
                          "recirculation mode indicator", "auto climate temperature setting",
                          "AC not cooling properly FATC", "express cooling activate procedure"],

        "cabin purifier": ["cabin air purification filter", "PM2.5 cabin filter indicator",
                           "advance filter cabin air quality", "air purifier infotainment AQI link"],

        # ═══════════════════════════════════════════════════════
        # SECTION 7: CAMERA & PARKING
        # ═══════════════════════════════════════════════════════

        "surround view": ["surround view system SVS indicator", "360 camera activation",
                          "bird eye view camera procedure", "SVS calibration after bumper repair",
                          "surround view not working black screen", "SVS camera blocked warning"],

        "rear camera":   ["RVC rear view camera not working", "rear view camera guidelines",
                          "reverse camera delay start", "rear camera image distorted"],

        "parking sensor": ["PDC park distance control beeping", "parking sensor disable quick access",
                           "park assist shortcut QAD", "front rear parking sensor not beeping",
                           "ultrasonic sensor blocked cleaning"],

        # ═══════════════════════════════════════════════════════
        # SECTION 8: KEYS & SECURITY
        # ═══════════════════════════════════════════════════════

        "smart key":     ["smart key PEPS not detected", "key not found voice alert",
                          "smart key battery low replace", "PEPS tailgate switch outside",
                          "keyless entry not working range", "passive entry push start"],

        "immobilizer":   ["anti-theft immobilizer warning", "immobilizer fault not starting",
                          "engine immobilizer indicator", "transponder key immobilizer"],

        "escl":          ["ESCL chime steering column lock", "electronic steering column lock engaged",
                          "ESCL inadvertently engaged warning", "steering lock fault procedure"],

        # ═══════════════════════════════════════════════════════
        # SECTION 9: COMFORT & SEATING
        # ═══════════════════════════════════════════════════════

        "ventilated seat": ["ventilated seat not working", "ventilated seat default setting",
                            "ventilated seat spillage damage warning", "ventilated seat not cooling",
                            "ventilated seat enable infotainment"],

        "seat":          ["power seat memory indicator", "seat heating indicator",
                          "ventilated seat symbol dashboard", "seat memory position recall"],

        "sunroof":       ["power sunroof switch overhead console", "sunroof open reminder",
                          "panoramic roof control procedure", "sunroof tilt vent position"],

        # ═══════════════════════════════════════════════════════
        # SECTION 10: AUDIO REMINDERS & CHIMES
        # ═══════════════════════════════════════════════════════

        "audio reminder": ["key-in reminder buzzer meaning", "park lamp on reminder buzzer",
                           "park brake on reminder 5kmph chime", "reverse gear buzzer meaning",
                           "high coolant temperature continuous chime", "TPMS chime duration meaning",
                           "BSVI urea chime stages meaning", "DDOA chime level change"],

        "chime":         ["audio chime meaning dashboard", "buzzer sound while driving meaning",
                          "continuous chime what does it mean", "chime on ignition on meaning"],

        "reverse":       ["reverse gear chime buzzer", "reverse camera auto activate",
                          "PDC beep reverse procedure", "RCTA activate in reverse"],

        # ═══════════════════════════════════════════════════════
        # SECTION 11: GENERIC CATCH-ALLS
        # ═══════════════════════════════════════════════════════

        "lane":          ["lane departure warning LDW indicator", "lane keep assist symbol",
                          "LDW disable procedure", "lane assist false beeping cause"],

        "forward collision": ["FCW indicator dashboard", "AEB activation too sensitive",
                              "collision alert disable procedure", "forward collision warning radar"],

        "towing":        ["tow mode indicator", "trailer sway warning",
                          "tow haul mode light", "trailer brake indicator"],

        "hill":          ["hill hold HHC fault indicator", "hill start assist lamp",
                          "hill descent HDC ON lamp green", "HDC speed setting procedure",
                          "HHC prevent rollback uphill"],

        "traction":      ["traction control light", "ESC ESP stability indicator",
                          "ESP off indicator amber meaning", "engine drag torque control EDTC",
                          "EDTC brake slip slippery road"],

        "steering":      ["EPAS fault indicator amber", "electric power steering warning",
                          "steering malfunction symbol heavy steering", "ESCL chime steering lock"],
    }

    # Multi-keyword matching — no break, so "ABS and EBD fault together"
    # correctly fires both "abs" and "ebd" expansion sets
    for keyword, alts in KEYWORD_EXPANSIONS.items():
        if keyword in q:
            queries += alts

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
