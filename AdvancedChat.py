#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Advanced Unified Chat — Streamlit

One app, three powers:
1) 🧪 LM Studio / OpenAI-compatible chat (streaming, multi-turn, system prompt, downloads)
2) 🖼️ Vision analysis (Gemini via agno) — general or medical presets, injectable into chat
3) 📊 Data agent (DuckDB/φ for SQL + Together+E2B for auto Python plots) — previews & chat, injectable into chat

───────────────────────────────────────────────────────────────────────────────
Dependencies (install as needed):
  pip install --upgrade streamlit openai tiktoken pillow pandas duckdb
  # Optional (Vision via agno + Gemini):
  pip install --upgrade agno google-generativeai
  # Optional (Data agent: φ, Together, E2B sandbox, plotly for some outputs):
  pip install --upgrade phi together e2b-code-interpreter plotly

Run:
  streamlit run advanced_unified_chat.py
───────────────────────────────────────────────────────────────────────────────
"""

# ─────────────────────────────────────────────────────────────────────────────
# Standard library
# ─────────────────────────────────────────────────────────────────────────────
import os
import io
import re
import csv
import json
import base64
import tempfile
import contextlib
import warnings
from typing import List, Dict, Generator, Optional, Tuple, Any

# ─────────────────────────────────────────────────────────────────────────────
# Third-party
# ─────────────────────────────────────────────────────────────────────────────
import streamlit as st

# OpenAI-compatible SDK (LM Studio adapter)
try:
    from openai import OpenAI
except Exception as e:
    raise RuntimeError("Missing dependency. Install with: pip install --upgrade openai streamlit") from e

# Token counting (best-effort)
try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")
    def count_tokens(text: str) -> int:
        return len(_ENC.encode(text or ""))
except Exception:
    def count_tokens(text: str) -> int:
        return max(1, len((text or "").split()))

# Images
try:
    from PIL import Image as PILImage
except Exception as e:
    raise RuntimeError("Missing dependency. Install with: pip install --upgrade pillow streamlit") from e

# Optional: DICOM support (preview)
try:
    import pydicom
    _HAVE_PYDICOM = True
except Exception:
    _HAVE_PYDICOM = False

# Optional: agno + Gemini (Vision)
try:
    from agno.agent import Agent as AgnoAgent
    from agno.models.google import Gemini
    from agno.media import Image as AgnoImage
    from agno.tools.duckduckgo import DuckDuckGoTools
    _HAVE_AGNO = True
except Exception:
    _HAVE_AGNO = False

# Optional: Data agent stack
try:
    import pandas as pd
except Exception:
    pd = None

try:
    from e2b_code_interpreter import Sandbox
    _HAVE_E2B = True
except Exception:
    _HAVE_E2B = False

try:
    from together import Together
    _HAVE_TOGETHER = True
except Exception:
    _HAVE_TOGETHER = False

try:
    from phi.agent.duckdb import DuckDbAgent
    from phi.model.openai import OpenAIChat as PhiOpenAIChat
    from phi.tools.pandas import PandasTools
    _HAVE_PHI = True
except Exception:
    _HAVE_PHI = False

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

# ─────────────────────────────────────────────────────────────────────────────
# Defaults & configuration
# ─────────────────────────────────────────────────────────────────────────────
APP_TITLE = "🧪📊 Unified Advanced Chat"

DEFAULT_BASE_URL = (
    os.getenv("LMSTUDIO_BASE_URL")
    or os.getenv("OPENAI_BASE_URL")
    or "http://localhost:1234/v1"
)

DEFAULT_API_KEY = (
    os.getenv("LMSTUDIO_API_KEY")
    or os.getenv("OPENAI_API_KEY")
    or "lm-studio"
)

DEFAULT_MODEL          = os.getenv("LMSTUDIO_MODEL", "qwen2.5-7b-instruct")
DEFAULT_TEMPERATURE    = float(os.getenv("LMSTUDIO_TEMPERATURE", "0.2"))
DEFAULT_TOP_P          = float(os.getenv("LMSTUDIO_TOP_P", "0.95"))
DEFAULT_MAX_TOKENS     = int(os.getenv("LMSTUDIO_MAX_TOKENS", "2048"))
DEFAULT_SYSTEM_PROMPT  = os.getenv(
    "LMSTUDIO_SYSTEM_PROMPT",
    "You are a rigorous, practical metallurgy and welding assistant. "
    "Explain clearly and concisely, show calculations step-by-step when relevant, "
    "cite standards/specs by name and number if you mention them, and flag uncertainties. "
    "When users give incomplete parameters, state reasonable assumptions explicitly."
)

# Vision / Data optional keys from env
ENV_GOOGLE_API_KEY   = os.getenv("GOOGLE_API_KEY", "")
ENV_TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "")
ENV_E2B_API_KEY      = os.getenv("E2B_API_KEY", "")
ENV_PHI_OPENAI_KEY   = os.getenv("OPENAI_API_KEY", "")  # for φ data agent


# ─────────────────────────────────────────────────────────────────────────────
# LLM Client wrapper (OpenAI-compatible: LM Studio, OpenAI, vLLM…)
# ─────────────────────────────────────────────────────────────────────────────
class LLMClient:
    def __init__(self, base_url: str, api_key: str):
        self._base_url = base_url.strip()
        self._api_key  = api_key.strip()
        self._client   = OpenAI(base_url=self._base_url, api_key=self._api_key)

    @property
    def base_url(self) -> str:
        return self._base_url

    def set_base_url(self, base_url: str):
        self._base_url = base_url.strip()
        self._client   = OpenAI(base_url=self._base_url, api_key=self._api_key)

    def set_api_key(self, api_key: str):
        self._api_key  = api_key.strip()
        self._client   = OpenAI(base_url=self._base_url, api_key=self._api_key)

    def list_models(self) -> List[str]:
        try:
            models = self._client.models.list()
            return sorted([m.id for m in getattr(models, "data", [])])
        except Exception:
            return []

    def stream_chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> Generator[str, None, None]:
        stream = self._client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=float(temperature),
            top_p=float(top_p),
            max_tokens=int(max_tokens),
            stream=True,
        )
        for event in stream:
            try:
                delta = event.choices[0].delta.content or ""
            except Exception:
                delta = ""
            if delta:
                yield delta


# ─────────────────────────────────────────────────────────────────────────────
# Session bootstrap
# ─────────────────────────────────────────────────────────────────────────────
def init_session_state():
    defaults = {
        # chat core
        "chat_history": [],
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
        "model": DEFAULT_MODEL,
        "temperature": DEFAULT_TEMPERATURE,
        "top_p": DEFAULT_TOP_P,
        "max_tokens": DEFAULT_MAX_TOKENS,
        "base_url": DEFAULT_BASE_URL,
        "api_key": DEFAULT_API_KEY,
        "client": None,

        # tool contexts (to inject into chat)
        "context_image_enabled": True,
        "context_data_enabled": True,
        "context_buffer": [],   # list of dicts: {"source":"vision|data","title":..., "content":...}

        # vision
        "vision_api_key": ENV_GOOGLE_API_KEY,
        "vision_preset": "General Vision",
        "vision_last": None,    # last analysis text
        "vision_last_title": None,

        # data agent
        "tg_key": ENV_TOGETHER_API_KEY,
        "e2b_key": ENV_E2B_API_KEY,
        "phi_openai_key": ENV_PHI_OPENAI_KEY,
        "tg_model_human": "Meta-Llama 3.1-405B",
        "tg_model_id": "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",

        "datasets": [],
        "data_prepared": False,
        "files_hash": None,
        "phi_agent": None,
        "data_chat_history": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    if st.session_state.client is None:
        st.session_state.client = LLMClient(st.session_state.base_url, st.session_state.api_key)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar controls
# ─────────────────────────────────────────────────────────────────────────────
def sidebar_controls():
    with st.sidebar:
        st.header("🔌 Chat Connection")
        base_url = st.text_input(
            "Base URL",
            value=st.session_state.base_url,
            help="LM Studio default: http://localhost:1234/v1. For hosted apps, use your public URL.",
        )
        api_key = st.text_input("API Key", value=st.session_state.api_key, type="password")

        # Model listing (best effort)
        client: LLMClient = st.session_state.client
        models_available = client.list_models()
        model_help = "Pick a model (or type the model ID if not listed)."
        if models_available:
            model = st.selectbox(
                "Model",
                options=models_available,
                index=0 if st.session_state.model not in models_available
                else models_available.index(st.session_state.model),
                help=model_help,
            )
        else:
            model = st.text_input("Model", value=st.session_state.model, help=model_help)

        temperature = st.slider("Temperature", 0.0, 2.0, float(st.session_state.temperature), 0.05)
        top_p       = st.slider("Top-p", 0.0, 1.0, float(st.session_state.top_p), 0.01)
        max_tokens  = st.slider("Max tokens", 64, 8192, int(st.session_state.max_tokens), 32)

        st.markdown("---")
        st.subheader("🧭 System Prompt")
        system_prompt = st.text_area("Instructions", value=st.session_state.system_prompt, height=140)

        st.markdown("---")
        st.subheader("🧩 Tool Context → Chat")
        context_image_enabled = st.toggle("Auto-inject latest Vision analysis into chat", value=st.session_state.context_image_enabled)
        context_data_enabled  = st.toggle("Auto-inject latest Data summary into chat", value=st.session_state.context_data_enabled)

        st.markdown("---")
        st.subheader("🔑 Optional API Keys")
        vision_api_key = st.text_input("Google API Key (Gemini, agno - Vision)", type="password", value=st.session_state.vision_api_key)
        tg_key         = st.text_input("Together API Key (Visual agent)", type="password", value=st.session_state.tg_key)
        e2b_key        = st.text_input("E2B API Key (Sandbox)", type="password", value=st.session_state.e2b_key)
        phi_openai_key = st.text_input("OpenAI API Key (φ / data analyst)", type="password", value=st.session_state.phi_openai_key)

        st.caption("Env fallbacks: GOOGLE_API_KEY, TOGETHER_API_KEY, E2B_API_KEY, OPENAI_API_KEY")

        st.markdown("### 🎛️ Together Model")
        TG_MODELS = {
            "Meta-Llama 3.1-405B": "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
            "DeepSeek V3":         "deepseek-ai/DeepSeek-V3",
            "Qwen 2.5-7B":         "Qwen/Qwen2.5-7B-Instruct-Turbo",
            "Meta-Llama 3.3-70B":  "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        }
        model_human = st.selectbox("Together model", list(TG_MODELS.keys()),
                                   index=list(TG_MODELS.keys()).index(st.session_state.tg_model_human))
        tg_model_id = TG_MODELS[model_human]

        # Apply / Test
        cols = st.columns(2)
        if cols[0].button("Apply"):
            dirty = (base_url.strip() != st.session_state.base_url.strip()) or \
                    (api_key.strip()  != st.session_state.api_key.strip())
            st.session_state.update({
                "base_url": base_url.strip(),
                "api_key": api_key.strip(),
                "model": model.strip(),
                "temperature": float(temperature),
                "top_p": float(top_p),
                "max_tokens": int(max_tokens),
                "system_prompt": system_prompt,
                "context_image_enabled": context_image_enabled,
                "context_data_enabled": context_data_enabled,
                "vision_api_key": vision_api_key,
                "tg_key": tg_key,
                "e2b_key": e2b_key,
                "phi_openai_key": phi_openai_key,
                "tg_model_human": model_human,
                "tg_model_id": tg_model_id,
            })
            if dirty:
                st.session_state.client = LLMClient(st.session_state.base_url, st.session_state.api_key)
            st.success("Settings applied.", icon="✅")

        if cols[1].button("Test connection"):
            try:
                _ = st.session_state.client.list_models()
                st.success("Server reachable.", icon="🟢")
            except Exception as e:
                st.error(f"Connection failed: {e}", icon="🔴")


# ─────────────────────────────────────────────────────────────────────────────
# Chat helpers
# ─────────────────────────────────────────────────────────────────────────────
def build_messages() -> List[Dict[str, str]]:
    msgs = [{"role": "system", "content": st.session_state.system_prompt.strip()}]

    # Inject tool context as additional system note (optional)
    context_lines = []
    if st.session_state.context_image_enabled and st.session_state.vision_last:
        title = st.session_state.vision_last_title or "Vision analysis"
        context_lines.append(f"## Attached vision context: {title}\n{st.session_state.vision_last}")

    # Data summary: pick the last assistant message from data chat (if any)
    if st.session_state.context_data_enabled:
        last_data = None
        for m in reversed(st.session_state.data_chat_history):
            if m.get("role") == "assistant":
                last_data = m.get("content")
                break
        if last_data:
            context_lines.append(f"## Attached data context (latest data-agent answer)\n{last_data}")

    if context_lines:
        msgs.append({
            "role": "system",
            "content": "Additional context from integrated tools:\n\n" + "\n\n---\n\n".join(context_lines)
        })

    for turn in st.session_state.chat_history:
        role = turn.get("role", "user")
        content = turn.get("content", "")
        if content:
            msgs.append({"role": role, "content": content})
    return msgs


def download_buttons():
    hist = st.session_state.chat_history
    json_bytes = json.dumps(hist, ensure_ascii=False, indent=2).encode("utf-8")

    # Markdown transcript
    md_lines = [f"# {APP_TITLE}", ""]
    md_lines += ["## System Prompt", "", "```\n" + st.session_state.system_prompt + "\n```", "", "## Conversation", ""]
    for msg in hist:
        if msg["role"] == "user":
            md_lines.append("**User:**")
        elif msg["role"] == "assistant":
            md_lines.append("**Assistant:**")
        else:
            md_lines.append(f"**{msg['role'].title()}:**")
        md_lines.append("")
        md_lines.append(msg["content"])
        md_lines.append("")

    md_bytes = "\n".join(md_lines).encode("utf-8")

    c1, c2 = st.columns(2)
    c1.download_button("⬇️ Download JSON", json_bytes, file_name="chat_transcript.json", mime="application/json")
    c2.download_button("⬇️ Download Markdown", md_bytes, file_name="chat_transcript.md", mime="text/markdown")


# ─────────────────────────────────────────────────────────────────────────────
# Vision (image analysis) — general or medical preset
# ─────────────────────────────────────────────────────────────────────────────
VISION_GENERAL_PROMPT = """
You are an expert visual analyst. Carefully examine the uploaded image and respond in markdown with:

### 1) Overview
- What is shown? Scene type, objects, composition, quality.

### 2) Key Observations
- Bullet points of the most salient, quantifiable details (colors, counts, text, geometry, defects, etc.).

### 3) Task-Relevant Insights
- If the user mentions a goal (e.g., quality control, documentation, marketing), provide actionable observations tied to that goal.

### 4) Risks / Uncertainties
- Ambiguities or limits of inference from a single image.

### 5) Compact Summary (≤ 5 lines)
- A tight summary suitable for injecting into another LLM as context.
"""

VISION_MEDICAL_PROMPT = """
You are a highly skilled medical imaging expert. Analyze the image and respond with:

### 1) Image Type & Region
- Modality (X-ray/MRI/CT/Ultrasound/Dermatology/Angiogram/etc.)
- Anatomy & positioning
- Image quality/adequacy

### 2) Key Findings
- Primary observations; abnormalities with precise descriptions (location, size, shape, density/echo/signal).
- Severity grading where applicable (Normal/Mild/Moderate/Severe).
- Dermatology: count lesions + approximate size and location mapping.
- Angiography: check for balloon catheter (wire + cylindrical low-attenuation balloon) and for focal/segmental vascular stenosis; avoid false positives (branching taper, overlap, motion, partial filling).

### 3) Diagnostic Assessment
- Primary diagnosis + confidence, differentials with supporting evidence, critical findings.

### 4) Patient-Friendly Explanation
- Clear lay summary, minimal jargon.

### 5) Research Context
- Use web tools to cite 2–3 recent references or protocols if available; otherwise state none found.

### 6) Compact Summary (≤ 5 lines)
- Tight summary suitable for injecting into another LLM as context.

⚠️ Educational use only; not medical advice.
"""

def load_image_any(uploaded_file) -> Optional[PILImage.Image]:
    """Load standard images with PIL, or DICOM via pydicom if available."""
    name = uploaded_file.name.lower()
    try:
        if any(name.endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"]):
            return PILImage.open(uploaded_file).convert("RGB")
        if any(name.endswith(ext) for ext in [".dcm", ".dicom"]) and _HAVE_PYDICOM:
            ds = pydicom.dcmread(io.BytesIO(uploaded_file.getvalue()))
            arr = ds.pixel_array
            # Normalize to 0-255 for preview
            import numpy as np
            arr = arr.astype("float32")
            arr = (255 * (arr - arr.min()) / max(1e-6, (arr.max() - arr.min()))).clip(0, 255).astype("uint8")
            return PILImage.fromarray(arr)
    except Exception as e:
        st.error(f"Image load error: {e}")
    return None


def run_vision_analysis(image: PILImage.Image, preset: str, title_hint: str) -> Optional[str]:
    """Run analysis via agno+Gemini if available & key present, else return None."""
    if not _HAVE_AGNO:
        st.warning("Optional package 'agno' not installed. Install with `pip install agno google-generativeai`.")
        return None
    if not st.session_state.vision_api_key:
        st.warning("Please provide a Google API Key in the sidebar for Gemini analysis.")
        return None

    prompt = VISION_GENERAL_PROMPT if preset == "General Vision" else VISION_MEDICAL_PROMPT

    # Save a temp resized copy for agno Image wrapper
    try:
        width, height = image.size
        max_w = 1200
        if width > max_w:
            new_h = int(height * (max_w / width))
            image = image.resize((max_w, new_h))
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        image.save(tmp.name)
        agno_img = AgnoImage(filepath=tmp.name)
    except Exception as e:
        st.error(f"Temp image error: {e}")
        return None

    try:
        agent = AgnoAgent(
            model=Gemini(id="gemini-2.0-flash", api_key=st.session_state.vision_api_key),
            tools=[DuckDuckGoTools()],
            markdown=True,
        )
        with st.spinner("🔎 Gemini is analyzing the image…"):
            response = agent.run(prompt, images=[agno_img])
        text = getattr(response, "content", None) or str(response)
        # Store in session
        st.session_state.vision_last = text
        st.session_state.vision_last_title = title_hint or "Image analysis"
        return text
    except Exception as e:
        st.error(f"Vision analysis error: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Data agent (φ / DuckDB + Together/E2B Visual) — helpers
# ─────────────────────────────────────────────────────────────────────────────
PYTHON_BLOCK = re.compile(r"```python\n(.*?)\n```", re.DOTALL)
VISUAL_KEYWORDS = {
    "plot","chart","graph","visual","visualise","visualize","scatter","bar",
    "line","hist","histogram","heatmap","pie","boxplot","area","map",
}
def router_agent(user_query: str) -> str:
    return "visual" if any(w in user_query.lower() for w in VISUAL_KEYWORDS) else "analyst"

def preprocess_and_save(file) -> Tuple[Optional[str], Optional[List[str]], Optional["pd.DataFrame"]]:
    """Sanitize uploaded CSV/XLSX → coerce types → persist to quoted temp CSV."""
    if pd is None:
        st.error("pandas is required. Install with: pip install pandas")
        return None, None, None
    try:
        if file.name.lower().endswith(".csv"):
            df = pd.read_csv(file, encoding="utf-8", na_values=["NA","N/A","missing"])
        elif file.name.lower().endswith((".xls", ".xlsx")):
            df = pd.read_excel(file, na_values=["NA","N/A","missing"])
        else:
            st.error("Unsupported file type. Upload CSV or Excel.")
            return None, None, None

        # Escape quotes for CSV safety
        for col in df.select_dtypes(include="object"):
            df[col] = df[col].astype(str).replace({r'"': '""'}, regex=True)

        # Basic type coercion
        for col in df.columns:
            if "date" in col.lower():
                df[col] = pd.to_datetime(df[col], errors="coerce")
            elif df[col].dtype == "object":
                df[col] = pd.to_numeric(df[col], errors="ignore")

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        df.to_csv(tmp.name, index=False, quoting=csv.QUOTE_ALL)
        return tmp.name, df.columns.tolist(), df
    except Exception as exc:
        st.error(f"Pre-processing failed: {exc}")
        return None, None, None

def extract_python(llm_response: str) -> str:
    m = PYTHON_BLOCK.search(llm_response)
    return m.group(1) if m else ""

def upload_to_sandbox(sb: "Sandbox", file) -> str:
    sandbox_path = f"./{file.name}"
    sb.files.write(sandbox_path, file.getvalue())
    file.seek(0)
    return sandbox_path

def execute_in_sandbox(sb: "Sandbox", code: str) -> Tuple[Optional[List[Any]], Optional[str]]:
    with st.spinner("🔧 Executing Python in sandbox…"):
        run = sb.run_code(code)
    return (None, run.error) if run.error else (run.results, None)

def visual_agent(query: str, dataset_infos: List[dict], tg_key: str, tg_model: str, e2b_key: str, max_retries: int = 2) -> Tuple[str, Optional[List[Any]]]:
    if not (_HAVE_TOGETHER and _HAVE_E2B):
        st.error("Together or E2B packages not installed. Install with: pip install together e2b-code-interpreter")
        return "", None
    if not (tg_key and e2b_key):
        st.error("Provide Together and E2B keys in the sidebar.")
        return "", None

    def llm_call(msgs: List[dict]) -> str:
        client = Together(api_key=tg_key)
        resp = client.chat.completions.create(model=tg_model, messages=msgs)
        return resp.choices[0].message.content

    with Sandbox(api_key=e2b_key) as sb:
        # Upload datasets & mapping
        for info in dataset_infos:
            info["sb_path"] = upload_to_sandbox(sb, info["file"])
        mapping_lines = [f"- **{d['name']}** ➜ `{d['sb_path']}`" for d in dataset_infos]

        system_prompt = (
            "You are a senior Python data-scientist and visualization expert.\n"
            "Datasets available inside the sandbox:\n" + "\n".join(mapping_lines) +
            "\n• Think step-by-step.\n"
            "• Return exactly ONE ```python ...``` block that uses the paths above verbatim.\n"
            "• After the code block, add a short explanation."
        )
        msgs = [{"role":"system","content":system_prompt}, {"role":"user","content":query}]
        cumulative = ""

        for attempt in range(max_retries + 1):
            with st.spinner("🤖 LLM (Together) is writing Python…"):
                llm_text = llm_call(msgs)
            cumulative += ("\n\n" + llm_text) if cumulative else llm_text

            code = extract_python(llm_text)
            if not code:
                st.warning("No ```python``` block found.")
                return cumulative, None

            results, err = execute_in_sandbox(sb, code)
            if err is None:
                return cumulative, results

            st.warning(f"Sandbox error:\n\n```\n{err}\n```")
            if attempt == max_retries:
                cumulative += "\n\n**Sandbox execution failed after retries. See error above.**"
                return cumulative, None

            msgs.extend([
                {"role":"assistant","content":llm_text},
                {"role":"user","content":f"Your code errored:\n```\n{err}\n```\nFix and return ONLY a new ```python``` block."},
            ])
    return cumulative, None

def build_phi_agent(csv_infos, openai_key: str) -> Optional["DuckDbAgent"]:
    if not (_HAVE_PHI and pd is not None):
        st.error("phi/pandas not available. Install with: pip install phi pandas")
        return None
    tables_meta = [
        {"name": i["name"], "description": f"Dataset {os.path.basename(i['path'])}.", "path": i["path"]}
        for i in csv_infos
    ]
    llm = PhiOpenAIChat(id="gpt-4o", api_key=openai_key)
    return DuckDbAgent(
        model=llm,
        semantic_model=json.dumps({"tables": tables_meta}, indent=2),
        tools=[PandasTools()],
        markdown=True,
        followups=False,
        system_prompt=(
            "You are an expert data analyst.\n"
            "1) Write ONE SQL query in ```sql```\n"
            "2) Execute it\n"
            "3) Present the result plainly"
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Streamlit App
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="🧪", layout="wide")
    init_session_state()

    st.title(APP_TITLE)
    st.caption(
        "Chat with your LM Studio/OpenAI-compatible model. Add context from Vision and Data tools. "
        "Everything in one place."
    )

    # Sidebar
    sidebar_controls()

    # Tabs
    tab_chat, tab_vision, tab_data = st.tabs(["💬 Chat", "🖼️ Vision", "📊 Data"])

    # ─────────────────────────────────────────────────────────────────────────
    # CHAT TAB
    # ─────────────────────────────────────────────────────────────────────────
    with tab_chat:
        # Context drawer
        with st.expander("📎 Context Drawer (latest tool outputs)", expanded=False):
            colA, colB = st.columns(2)
            # Vision context preview
            with colA:
                st.subheader("Vision")
                if st.session_state.vision_last:
                    st.markdown(f"**{st.session_state.vision_last_title or 'Image analysis'}**")
                    st.markdown(st.session_state.vision_last)
                    if st.button("➕ Inject this vision analysis into chat now"):
                        st.session_state.context_image_enabled = True
                        st.success("Vision analysis will be attached to the next chat turn.")
                else:
                    st.caption("No vision analysis yet.")
            # Data context preview
            with colB:
                st.subheader("Data")
                if st.session_state.data_chat_history:
                    last_data = None
                    for m in reversed(st.session_state.data_chat_history):
                        if m.get("role") == "assistant":
                            last_data = m["content"]; break
                    if last_data:
                        st.markdown(last_data)
                        if st.button("➕ Inject this data answer into chat now"):
                            st.session_state.context_data_enabled = True
                            st.success("Data-summary will be attached to the next chat turn.")
                    else:
                        st.caption("No data-agent answer yet.")
                else:
                    st.caption("No data-agent answer yet.")

        # Render main chat history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Chat input
        user_input = st.chat_input("Ask about metallurgy/welding… or anything you like.")
        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            # Stream reply
            with st.chat_message("assistant"):
                client: LLMClient = st.session_state.client
                messages = build_messages()

                def _generator():
                    try:
                        for chunk in client.stream_chat(
                            model=st.session_state.model,
                            messages=messages,
                            temperature=st.session_state.temperature,
                            top_p=st.session_state.top_p,
                            max_tokens=st.session_state.max_tokens,
                        ):
                            yield chunk
                    except Exception as e:
                        yield f"\n\n**[Error]** {e}"

                full_text = st.write_stream(_generator())

            st.session_state.chat_history.append({"role": "assistant", "content": full_text})

            # Token stats
            joined = [f"{m['role'].upper()}: {m['content']}" for m in st.session_state.chat_history]
            approx_tokens = count_tokens("\n".join(joined))
            st.caption(f"Approx. tokens in this session: **{approx_tokens}**")

        st.divider()
        c1, c2, c3 = st.columns([1, 1, 2])
        if c1.button("🧹 Clear chat"):
            st.session_state.chat_history = []
            st.rerun()
        download_buttons()
        st.info(
            f"Connected to **{st.session_state.base_url}** with model **{st.session_state.model}**.",
            icon="🌐",
        )

    # ─────────────────────────────────────────────────────────────────────────
    # VISION TAB
    # ─────────────────────────────────────────────────────────────────────────
    with tab_vision:
        st.subheader("Upload an image and analyze it (General or Medical preset)")
        col1, col2 = st.columns([1, 2])

        with col1:
            preset = st.radio("Analysis preset", ["General Vision", "Medical Imaging"], index=0)
            st.session_state.vision_preset = preset

            if not _HAVE_AGNO:
                st.warning("Vision agent optional: install `agno` and `google-generativeai`.", icon="ℹ️")

            if not st.session_state.vision_api_key:
                st.info("Provide your Google API key in the sidebar to enable Gemini vision.", icon="🔑")

        with col2:
            uploaded = st.file_uploader(
                "Upload image (JPG/PNG/WebP/BMP/TIFF, or DICOM .dcm/.dicom)",
                type=["jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff", "dcm", "dicom"],
                accept_multiple_files=False,
            )

        image = None
        if uploaded is not None:
            image = load_image_any(uploaded)
            if image is not None:
                st.image(image, caption=f"Preview — {uploaded.name}", use_container_width=True)
                analyze = st.button("🔍 Analyze Image", type="primary")
                if analyze:
                    text = run_vision_analysis(image, st.session_state.vision_preset, title_hint=uploaded.name)
                    if text:
                        st.markdown("### 📋 Analysis Result")
                        st.markdown("---")
                        st.markdown(text)
                        st.markdown("---")
                        st.caption("Note: AI-generated; verify with appropriate domain experts when needed.")
                        if st.button("📎 Send compact summary to Chat"):
                            # Prefer the last 'Compact Summary' if present
                            summary = text
                            # Heuristic: try to extract the compact section
                            if "Compact Summary" in text:
                                summary = text.split("Compact Summary", 1)[-1].strip()
                            st.session_state.vision_last = summary
                            st.session_state.vision_last_title = f"Vision summary: {uploaded.name}"
                            st.session_state.context_image_enabled = True
                            st.success("Summary will be attached to the next chat turn.")
            else:
                st.error("Could not load the uploaded file as an image.")

        if uploaded is None:
            st.info("👆 Upload an image to begin analysis.")

    # ─────────────────────────────────────────────────────────────────────────
    # DATA TAB
    # ─────────────────────────────────────────────────────────────────────────
    with tab_data:
        st.subheader("Upload datasets and chat (SQL via φ / Visuals via Together+E2B)")

        up_files = st.file_uploader(
            "📁 Upload one or more CSV or Excel files",
            type=["csv", "xls", "xlsx"], accept_multiple_files=True
        )

        # Handle uploads & preprocessing
        if up_files and st.session_state.files_hash != hash(tuple(f.name for f in up_files)):
            infos: List[dict] = []
            for f in up_files:
                path, cols, df = preprocess_and_save(f)
                if path:
                    infos.append({
                        "file": f,
                        "path": path,
                        "name": re.sub(r"\W|^(?=\d)", "_", os.path.splitext(f.name)[0]),
                        "columns": cols,
                        "df": df,
                    })
            if infos:
                st.session_state.update({
                    "datasets": infos,
                    "data_prepared": True,
                    "files_hash": hash(tuple(f.name for f in up_files)),
                    "phi_agent": build_phi_agent(infos, st.session_state.phi_openai_key) if st.session_state.phi_openai_key else None,
                })

        # Preview datasets
        if st.session_state.data_prepared and st.session_state.datasets:
            st.markdown("#### Dataset previews (first 8 rows)")
            for d in st.session_state.datasets:
                with st.expander(f"📄 {d['file'].name}  (table `{d['name']}`)"):
                    if pd is not None:
                        st.dataframe(d["df"].head(8), use_container_width=True)
                    st.caption(f"Columns: {', '.join(d['columns'])}")
        else:
            st.info("⬆️ Upload one or more datasets to start data chat.")

        # Data chat area
        if st.session_state.data_prepared:
            # Render data-chat history
            for m in st.session_state.data_chat_history:
                with st.chat_message(m["role"]):
                    st.markdown(m["content"])
                    for r in m.get("vis_results", []):
                        # results: images / matplotlib / plotly / tables
                        typ = r.get("type")
                        if typ == "image":
                            st.image(r["data"])
                        elif typ == "matplotlib":
                            st.pyplot(r["data"])
                        elif typ == "plotly":
                            st.plotly_chart(r["data"])
                        elif typ == "table":
                            st.dataframe(r["data"])
                        else:
                            st.write(r["data"])

            prompt = st.chat_input("Ask a question about your data (say 'plot'/'chart' to trigger visuals)…")
            if prompt:
                st.session_state.data_chat_history.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                agent_type = router_agent(prompt)
                # Visual branch
                if agent_type == "visual":
                    if not (st.session_state.tg_key and st.session_state.e2b_key):
                        st.error("Provide Together and E2B keys in the sidebar.")
                    else:
                        text, results = visual_agent(
                            query=prompt,
                            dataset_infos=st.session_state.datasets,
                            tg_key=st.session_state.tg_key,
                            tg_model=st.session_state.tg_model_id,
                            e2b_key=st.session_state.e2b_key,
                        )
                        msg = {"role": "assistant", "content": text or "(no text)", "vis_results": []}
                        with st.chat_message("assistant"):
                            st.markdown(text or "(no text)")
                            if results:
                                # Interpret typical E2B results contract
                                from io import BytesIO
                                res_count = 0
                                for obj in results:
                                    res_count += 1
                                    # ① images via base64 (obj.png)
                                    if getattr(obj, "png", None):
                                        try:
                                            raw = base64.b64decode(obj.png)
                                            img = PILImage.open(BytesIO(raw))
                                            st.image(img)
                                            msg["vis_results"].append({"type":"image", "data": img})
                                        except Exception:
                                            st.write(obj)
                                            msg["vis_results"].append({"type":"text", "data": str(obj)})
                                    # ② matplotlib fig
                                    elif getattr(obj, "figure", None):
                                        st.pyplot(obj.figure)
                                        msg["vis_results"].append({"type":"matplotlib", "data": obj.figure})
                                    # ③ plotly
                                    elif getattr(obj, "show", None):
                                        st.plotly_chart(obj)
                                        msg["vis_results"].append({"type":"plotly", "data": obj})
                                    # ④ tables
                                    elif pd is not None and isinstance(obj, (pd.DataFrame, pd.Series)):
                                        st.dataframe(obj)
                                        msg["vis_results"].append({"type":"table", "data": obj})
                                    else:
                                        st.write(obj)
                                        msg["vis_results"].append({"type":"text", "data": str(obj)})
                                if res_count == 0:
                                    st.caption("No objects returned by sandbox.")
                        st.session_state.data_chat_history.append(msg)
                # Analyst branch
                else:
                    if not st.session_state.phi_openai_key:
                        st.error("Provide an OpenAI API key for φ in the sidebar.")
                    else:
                        if st.session_state.phi_agent is None:
                            st.session_state.phi_agent = build_phi_agent(st.session_state.datasets, st.session_state.phi_openai_key)
                        if st.session_state.phi_agent is None:
                            st.stop()
                        with st.spinner("🧠 φ-agent (DuckDB) is thinking…"):
                            run = st.session_state.phi_agent.run(prompt)
                        answer = run.content if hasattr(run, "content") else str(run)
                        with st.chat_message("assistant"):
                            st.markdown(answer)
                        st.session_state.data_chat_history.append({"role": "assistant", "content": answer})

            # Push last data answer to main chat context
            if st.session_state.data_chat_history:
                if st.button("📎 Send latest data-agent answer to Chat"):
                    for m in reversed(st.session_state.data_chat_history):
                        if m.get("role") == "assistant":
                            st.session_state.context_data_enabled = True
                            st.success("Latest data answer will be attached to the next chat turn.")
                            break

        # Helper info if libs are missing
        missing = []
        if pd is None: missing.append("pandas")
        if not _HAVE_PHI: missing.append("phi")
        if not _HAVE_TOGETHER: missing.append("together")
        if not _HAVE_E2B: missing.append("e2b-code-interpreter")
        if missing:
            st.info("Optional data stack components missing: " + ", ".join(missing))

# ─────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
