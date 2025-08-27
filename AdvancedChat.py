# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# """
# Advanced Unified Chat — Streamlit (E2B-compatible)

# One app, three powers:
# 1) 💬 LM Studio / OpenAI-compatible chat (streaming, multi-turn, system prompt, downloads)
# 2) 🖼️ Vision analysis (Gemini via agno) — general or medical presets, injectable into chat
# 3) 📊 Data agent (DuckDB/φ for SQL + Together+E2B for auto Python plots) — previews & chat, injectable into chat

# ───────────────────────────────────────────────────────────────────────────────
# Install (pick what you need):
#   pip install --upgrade streamlit openai tiktoken pillow pandas duckdb
#   pip install --upgrade agno google-generativeai             # (Vision)
#   pip install --upgrade phi together e2b-code-interpreter    # (Data visuals)
#   pip install --upgrade plotly pydicom                       # (optional: plots, DICOM preview)

# Run:
#   streamlit run AdvancedChat.py
# ───────────────────────────────────────────────────────────────────────────────
# """

# # ─────────────────────────────────────────────────────────────────────────────
# # Standard library
# # ─────────────────────────────────────────────────────────────────────────────
# import os
# import io
# import re
# import csv
# import json
# import base64
# import tempfile
# import contextlib
# import warnings
# from typing import List, Dict, Generator, Optional, Tuple, Any

# # ─────────────────────────────────────────────────────────────────────────────
# # Third-party
# # ─────────────────────────────────────────────────────────────────────────────
# import streamlit as st

# # OpenAI-compatible SDK (LM Studio adapter)
# try:
#     from openai import OpenAI
# except Exception as e:
#     raise RuntimeError("Missing dependency. Install with: pip install --upgrade openai streamlit") from e

# # Token counting (best-effort)
# try:
#     import tiktoken
#     _ENC = tiktoken.get_encoding("cl100k_base")
#     def count_tokens(text: str) -> int:
#         return len(_ENC.encode(text or ""))
# except Exception:
#     def count_tokens(text: str) -> int:
#         return max(1, len((text or "").split()))

# # Images
# try:
#     from PIL import Image as PILImage
# except Exception as e:
#     raise RuntimeError("Missing dependency. Install with: pip install --upgrade pillow streamlit") from e

# # Optional: DICOM support (preview)
# try:
#     import pydicom
#     _HAVE_PYDICOM = True
# except Exception:
#     _HAVE_PYDICOM = False

# # Optional: agno + Gemini (Vision)
# try:
#     from agno.agent import Agent as AgnoAgent
#     from agno.models.google import Gemini
#     from agno.media import Image as AgnoImage
#     from agno.tools.duckduckgo import DuckDuckGoTools
#     _HAVE_AGNO = True
# except Exception:
#     _HAVE_AGNO = False

# # Optional: Data agent stack
# try:
#     import pandas as pd
# except Exception:
#     pd = None

# try:
#     from e2b_code_interpreter import Sandbox
#     _HAVE_E2B = True
# except Exception:
#     _HAVE_E2B = False

# try:
#     from together import Together
#     _HAVE_TOGETHER = True
# except Exception:
#     _HAVE_TOGETHER = False

# try:
#     from phi.agent.duckdb import DuckDbAgent
#     from phi.model.openai import OpenAIChat as PhiOpenAIChat
#     from phi.tools.pandas import PandasTools
#     _HAVE_PHI = True
# except Exception:
#     _HAVE_PHI = False

# warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

# # ─────────────────────────────────────────────────────────────────────────────
# # Defaults & configuration
# # ─────────────────────────────────────────────────────────────────────────────
# APP_TITLE = "🧪📊 Unified Advanced Chat"

# DEFAULT_BASE_URL = (
#     os.getenv("LMSTUDIO_BASE_URL")
#     or os.getenv("OPENAI_BASE_URL")
#     or "http://localhost:1234/v1"
# )

# DEFAULT_API_KEY = (
#     os.getenv("LMSTUDIO_API_KEY")
#     or os.getenv("OPENAI_API_KEY")
#     or "lm-studio"
# )

# DEFAULT_MODEL          = os.getenv("LMSTUDIO_MODEL", "qwen2.5-7b-instruct")
# DEFAULT_TEMPERATURE    = float(os.getenv("LMSTUDIO_TEMPERATURE", "0.2"))
# DEFAULT_TOP_P          = float(os.getenv("LMSTUDIO_TOP_P", "0.95"))
# DEFAULT_MAX_TOKENS     = int(os.getenv("LMSTUDIO_MAX_TOKENS", "2048"))
# DEFAULT_SYSTEM_PROMPT  = os.getenv(
#     "LMSTUDIO_SYSTEM_PROMPT",
#     "You are a rigorous, practical metallurgy and welding assistant. "
#     "Explain clearly and concisely, show calculations step-by-step when relevant, "
#     "cite standards/specs by name and number if you mention them, and flag uncertainties. "
#     "When users give incomplete parameters, state reasonable assumptions explicitly."
# )

# # Vision / Data optional keys from env
# ENV_GOOGLE_API_KEY   = os.getenv("GOOGLE_API_KEY", "")
# ENV_TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "")
# ENV_E2B_API_KEY      = os.getenv("E2B_API_KEY", "")
# ENV_PHI_OPENAI_KEY   = os.getenv("OPENAI_API_KEY", "")  # for φ data agent

# # ─────────────────────────────────────────────────────────────────────────────
# # LLM Client wrapper (OpenAI-compatible: LM Studio, OpenAI, vLLM…)
# # ─────────────────────────────────────────────────────────────────────────────
# class LLMClient:
#     def __init__(self, base_url: str, api_key: str):
#         self._base_url = base_url.strip()
#         self._api_key  = api_key.strip()
#         self._client   = OpenAI(base_url=self._base_url, api_key=self._api_key)

#     @property
#     def base_url(self) -> str:
#         return self._base_url

#     def set_base_url(self, base_url: str):
#         self._base_url = base_url.strip()
#         self._client   = OpenAI(base_url=self._base_url, api_key=self._api_key)

#     def set_api_key(self, api_key: str):
#         self._api_key  = api_key.strip()
#         self._client   = OpenAI(base_url=self._base_url, api_key=self._api_key)

#     def list_models(self) -> List[str]:
#         try:
#             models = self._client.models.list()
#             return sorted([m.id for m in getattr(models, "data", [])])
#         except Exception:
#             return []

#     def stream_chat(
#         self,
#         model: str,
#         messages: List[Dict[str, str]],
#         temperature: float = DEFAULT_TEMPERATURE,
#         top_p: float = DEFAULT_TOP_P,
#         max_tokens: int = DEFAULT_MAX_TOKENS,
#     ) -> Generator[str, None, None]:
#         stream = self._client.chat.completions.create(
#             model=model,
#             messages=messages,
#             temperature=float(temperature),
#             top_p=float(top_p),
#             max_tokens=int(max_tokens),
#             stream=True,
#         )
#         for event in stream:
#             try:
#                 delta = event.choices[0].delta.content or ""
#             except Exception:
#                 delta = ""
#             if delta:
#                 yield delta

# # ─────────────────────────────────────────────────────────────────────────────
# # Session bootstrap
# # ─────────────────────────────────────────────────────────────────────────────
# def init_session_state():
#     defaults = {
#         # chat core
#         "chat_history": [],
#         "system_prompt": DEFAULT_SYSTEM_PROMPT,
#         "model": DEFAULT_MODEL,
#         "temperature": DEFAULT_TEMPERATURE,
#         "top_p": DEFAULT_TOP_P,
#         "max_tokens": DEFAULT_MAX_TOKENS,
#         "base_url": DEFAULT_BASE_URL,
#         "api_key": DEFAULT_API_KEY,
#         "client": None,

#         # tool contexts (to inject into chat)
#         "context_image_enabled": True,
#         "context_data_enabled": True,
#         "context_buffer": [],

#         # vision
#         "vision_api_key": ENV_GOOGLE_API_KEY,
#         "vision_preset": "General Vision",
#         "vision_last": None,
#         "vision_last_title": None,

#         # data agent
#         "tg_key": ENV_TOGETHER_API_KEY,
#         "e2b_key": ENV_E2B_API_KEY,
#         "phi_openai_key": ENV_PHI_OPENAI_KEY,
#         "tg_model_human": "Meta-Llama 3.1-405B",
#         "tg_model_id": "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",

#         "datasets": [],
#         "data_prepared": False,
#         "files_hash": None,
#         "phi_agent": None,
#         "data_chat_history": [],
#     }
#     for k, v in defaults.items():
#         if k not in st.session_state:
#             st.session_state[k] = v

#     if st.session_state.client is None:
#         st.session_state.client = LLMClient(st.session_state.base_url, st.session_state.api_key)

# # ─────────────────────────────────────────────────────────────────────────────
# # Sidebar controls
# # ─────────────────────────────────────────────────────────────────────────────
# def sidebar_controls():
#     with st.sidebar:
#         st.header("🔌 Chat Connection")
#         base_url = st.text_input(
#             "Base URL",
#             value=st.session_state.base_url,
#             help="LM Studio default: http://localhost:1234/v1. For hosted apps, use your public URL.",
#         )
#         api_key = st.text_input("API Key", value=st.session_state.api_key, type="password")

#         client: LLMClient = st.session_state.client
#         models_available = client.list_models()
#         model_help = "Pick a model (or type the model ID if not listed)."
#         if models_available:
#             model = st.selectbox(
#                 "Model",
#                 options=models_available,
#                 index=0 if st.session_state.model not in models_available
#                 else models_available.index(st.session_state.model),
#                 help=model_help,
#             )
#         else:
#             model = st.text_input("Model", value=st.session_state.model, help=model_help)

#         temperature = st.slider("Temperature", 0.0, 2.0, float(st.session_state.temperature), 0.05)
#         top_p       = st.slider("Top-p", 0.0, 1.0, float(st.session_state.top_p), 0.01)
#         max_tokens  = st.slider("Max tokens", 64, 8192, int(st.session_state.max_tokens), 32)

#         st.markdown("---")
#         st.subheader("🧭 System Prompt")
#         system_prompt = st.text_area("Instructions", value=st.session_state.system_prompt, height=140)

#         st.markdown("---")
#         st.subheader("🧩 Tool Context → Chat")
#         context_image_enabled = st.toggle("Auto-inject latest Vision analysis into chat", value=st.session_state.context_image_enabled)
#         context_data_enabled  = st.toggle("Auto-inject latest Data summary into chat", value=st.session_state.context_data_enabled)

#         st.markdown("---")
#         st.subheader("🔑 Optional API Keys")
#         vision_api_key = st.text_input("Google API Key (Gemini, agno - Vision)", type="password", value=st.session_state.vision_api_key)
#         tg_key         = st.text_input("Together API Key (Visual agent)", type="password", value=st.session_state.tg_key)
#         e2b_key        = st.text_input("E2B API Key (Sandbox)", type="password", value=st.session_state.e2b_key)
#         phi_openai_key = st.text_input("OpenAI API Key (φ / data analyst)", type="password", value=st.session_state.phi_openai_key)

#         st.caption("Env fallbacks: GOOGLE_API_KEY, TOGETHER_API_KEY, E2B_API_KEY, OPENAI_API_KEY")

#         st.markdown("### 🎛️ Together Model")
#         TG_MODELS = {
#             "Meta-Llama 3.1-405B": "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
#             "DeepSeek V3":         "deepseek-ai/DeepSeek-V3",
#             "Qwen 2.5-7B":         "Qwen/Qwen2.5-7B-Instruct-Turbo",
#             "Meta-Llama 3.3-70B":  "meta-llama/Llama-3.3-70B-Instruct-Turbo",
#         }
#         model_human = st.selectbox("Together model", list(TG_MODELS.keys()),
#                                    index=list(TG_MODELS.keys()).index(st.session_state.tg_model_human))
#         tg_model_id = TG_MODELS[model_human]

#         cols = st.columns(2)
#         if cols[0].button("Apply"):
#             dirty = (base_url.strip() != st.session_state.base_url.strip()) or \
#                     (api_key.strip()  != st.session_state.api_key.strip())
#             st.session_state.update({
#                 "base_url": base_url.strip(),
#                 "api_key": api_key.strip(),
#                 "model": model.strip(),
#                 "temperature": float(temperature),
#                 "top_p": float(top_p),
#                 "max_tokens": int(max_tokens),
#                 "system_prompt": system_prompt,
#                 "context_image_enabled": context_image_enabled,
#                 "context_data_enabled": context_data_enabled,
#                 "vision_api_key": vision_api_key,
#                 "tg_key": tg_key,
#                 "e2b_key": e2b_key,
#                 "phi_openai_key": phi_openai_key,
#                 "tg_model_human": model_human,
#                 "tg_model_id": tg_model_id,
#             })
#             if dirty:
#                 st.session_state.client = LLMClient(st.session_state.base_url, st.session_state.api_key)
#             st.success("Settings applied.", icon="✅")

#         if cols[1].button("Test connection"):
#             try:
#                 _ = st.session_state.client.list_models()
#                 st.success("Server reachable.", icon="🟢")
#             except Exception as e:
#                 st.error(f"Connection failed: {e}", icon="🔴")

# # ─────────────────────────────────────────────────────────────────────────────
# # Chat helpers
# # ─────────────────────────────────────────────────────────────────────────────
# def build_messages() -> List[Dict[str, str]]:
#     msgs = [{"role": "system", "content": st.session_state.system_prompt.strip()}]

#     # Inject tool context as additional system note (optional)
#     context_lines = []
#     if st.session_state.context_image_enabled and st.session_state.vision_last:
#         title = st.session_state.vision_last_title or "Vision analysis"
#         context_lines.append(f"## Attached vision context: {title}\n{st.session_state.vision_last}")

#     if st.session_state.context_data_enabled:
#         last_data = None
#         for m in reversed(st.session_state.data_chat_history):
#             if m.get("role") == "assistant":
#                 last_data = m.get("content")
#                 break
#         if last_data:
#             context_lines.append(f"## Attached data context (latest data-agent answer)\n{last_data}")

#     if context_lines:
#         msgs.append({
#             "role": "system",
#             "content": "Additional context from integrated tools:\n\n" + "\n\n---\n\n".join(context_lines)
#         })

#     for turn in st.session_state.chat_history:
#         role = turn.get("role", "user")
#         content = turn.get("content", "")
#         if content:
#             msgs.append({"role": role, "content": content})
#     return msgs

# def download_buttons():
#     hist = st.session_state.chat_history
#     json_bytes = json.dumps(hist, ensure_ascii=False, indent=2).encode("utf-8")

#     md_lines = [f"# {APP_TITLE}", ""]
#     md_lines += ["## System Prompt", "", "```\n" + st.session_state.system_prompt + "\n```", "", "## Conversation", ""]
#     for msg in hist:
#         if msg["role"] == "user":
#             md_lines.append("**User:**")
#         elif msg["role"] == "assistant":
#             md_lines.append("**Assistant:**")
#         else:
#             md_lines.append(f"**{msg['role'].title()}:**")
#         md_lines.append("")
#         md_lines.append(msg["content"])
#         md_lines.append("")

#     md_bytes = "\n".join(md_lines).encode("utf-8")

#     c1, c2 = st.columns(2)
#     c1.download_button("⬇️ Download JSON", json_bytes, file_name="chat_transcript.json", mime="application/json")
#     c2.download_button("⬇️ Download Markdown", md_bytes, file_name="chat_transcript.md", mime="text/markdown")

# # ─────────────────────────────────────────────────────────────────────────────
# # Vision (image analysis)
# # ─────────────────────────────────────────────────────────────────────────────
# VISION_GENERAL_PROMPT = """
# You are an expert visual analyst. Carefully examine the uploaded image and respond in markdown with:

# ### 1) Overview
# - What is shown? Scene type, objects, composition, quality.

# ### 2) Key Observations
# - Bullet points of the most salient, quantifiable details (colors, counts, text, geometry, defects, etc.).

# ### 3) Task-Relevant Insights
# - If the user mentions a goal (e.g., quality control, documentation, marketing), provide actionable observations tied to that goal.

# ### 4) Risks / Uncertainties
# - Ambiguities or limits of inference from a single image.

# ### 5) Compact Summary (≤ 5 lines)
# - A tight summary suitable for injecting into another LLM as context.
# """

# VISION_MEDICAL_PROMPT = """
# You are a highly skilled medical imaging expert. Analyze the image and respond with:

# ### 1) Image Type & Region
# - Modality (X-ray/MRI/CT/Ultrasound/Dermatology/Angiogram/etc.)
# - Anatomy & positioning
# - Image quality/adequacy

# ### 2) Key Findings
# - Primary observations; abnormalities with precise descriptions (location, size, shape, density/echo/signal).
# - Severity grading where applicable (Normal/Mild/Moderate/Severe).
# - Dermatology: count lesions + approximate size and location mapping.
# - Angiography: check for balloon catheter (wire + cylindrical low-attenuation balloon) and for focal/segmental vascular stenosis; avoid false positives (branching taper, overlap, motion, partial filling).

# ### 3) Diagnostic Assessment
# - Primary diagnosis + confidence, differentials with supporting evidence, critical findings.

# ### 4) Patient-Friendly Explanation
# - Clear lay summary, minimal jargon.

# ### 5) Research Context
# - Use web tools to cite 2–3 recent references or protocols if available; otherwise state none found.

# ### 6) Compact Summary (≤ 5 lines)
# - Tight summary suitable for injecting into another LLM as context.

# ⚠️ Educational use only; not medical advice.
# """

# def load_image_any(uploaded_file) -> Optional[PILImage.Image]:
#     """Load standard images with PIL, or DICOM via pydicom if available."""
#     name = uploaded_file.name.lower()
#     try:
#         if any(name.endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"]):
#             return PILImage.open(uploaded_file).convert("RGB")
#         if any(name.endswith(ext) for ext in [".dcm", ".dicom"]) and _HAVE_PYDICOM:
#             ds = pydicom.dcmread(io.BytesIO(uploaded_file.getvalue()))
#             arr = ds.pixel_array
#             # Normalize to 0-255 for preview
#             import numpy as np
#             arr = arr.astype("float32")
#             arr = (255 * (arr - arr.min()) / max(1e-6, (arr.max() - arr.min()))).clip(0, 255).astype("uint8")
#             return PILImage.fromarray(arr)
#     except Exception as e:
#         st.error(f"Image load error: {e}")
#     return None

# def run_vision_analysis(image: PILImage.Image, preset: str, title_hint: str) -> Optional[str]:
#     """Run analysis via agno+Gemini if available & key present, else return None."""
#     if not _HAVE_AGNO:
#         st.warning("Optional package 'agno' not installed. Install with `pip install agno google-generativeai`.")
#         return None
#     if not st.session_state.vision_api_key:
#         st.warning("Please provide a Google API Key in the sidebar for Gemini analysis.")
#         return None

#     prompt = VISION_GENERAL_PROMPT if preset == "General Vision" else VISION_MEDICAL_PROMPT

#     try:
#         width, height = image.size
#         max_w = 1200
#         if width > max_w:
#             new_h = int(height * (max_w / width))
#             image = image.resize((max_w, new_h))
#         tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
#         image.save(tmp.name)
#         agno_img = AgnoImage(filepath=tmp.name)
#     except Exception as e:
#         st.error(f"Temp image error: {e}")
#         return None

#     try:
#         agent = AgnoAgent(
#             model=Gemini(id="gemini-2.0-flash", api_key=st.session_state.vision_api_key),
#             tools=[DuckDuckGoTools()],
#             markdown=True,
#         )
#         with st.spinner("🔎 Gemini is analyzing the image…"):
#             response = agent.run(prompt, images=[agno_img])
#         text = getattr(response, "content", None) or str(response)
#         st.session_state.vision_last = text
#         st.session_state.vision_last_title = title_hint or "Image analysis"
#         return text
#     except Exception as e:
#         st.error(f"Vision analysis error: {e}")
#         return None

# # ─────────────────────────────────────────────────────────────────────────────
# # Data agent (φ / DuckDB + Together/E2B Visual) — helpers
# # ─────────────────────────────────────────────────────────────────────────────
# PYTHON_BLOCK = re.compile(r"```python\n(.*?)\n```", re.DOTALL)
# VISUAL_KEYWORDS = {
#     "plot","chart","graph","visual","visualise","visualize","scatter","bar",
#     "line","hist","histogram","heatmap","pie","boxplot","area","map",
# }
# def router_agent(user_query: str) -> str:
#     return "visual" if any(w in user_query.lower() for w in VISUAL_KEYWORDS) else "analyst"

# def safe_numeric_coercion(df: "pd.DataFrame") -> "pd.DataFrame":
#     """Future-proof replacement for errors='ignore' with a conservative policy."""
#     if pd is None:
#         return df
#     for col in df.columns:
#         if df[col].dtype == "object":
#             # Try numeric coercion; keep only if it meaningfully converts values.
#             try:
#                 tmp = pd.to_numeric(df[col], errors="coerce")
#                 # If we get at least ~80% non-null after coercion, adopt it.
#                 if tmp.notna().mean() >= 0.8:
#                     df[col] = tmp
#             except Exception:
#                 # Leave column as-is if conversion fails wholesale.
#                 pass
#     return df

# def preprocess_and_save(file) -> Tuple[Optional[str], Optional[List[str]], Optional["pd.DataFrame"]]:
#     """Sanitize uploaded CSV/XLSX → coerce types → persist to quoted temp CSV."""
#     if pd is None:
#         st.error("pandas is required. Install with: pip install pandas")
#         return None, None, None
#     try:
#         if file.name.lower().endswith(".csv"):
#             df = pd.read_csv(file, encoding="utf-8", na_values=["NA","N/A","missing"])
#         elif file.name.lower().endswith((".xls", ".xlsx")):
#             df = pd.read_excel(file, na_values=["NA","N/A","missing"])
#         else:
#             st.error("Unsupported file type. Upload CSV or Excel.")
#             return None, None, None

#         # Escape quotes for CSV safety
#         for col in df.select_dtypes(include="object"):
#             df[col] = df[col].astype(str).replace({r'"': '""'}, regex=True)

#         # Conservative type coercion (no errors='ignore')
#         df = safe_numeric_coercion(df)

#         # Persist to temp CSV for DuckDB
#         tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
#         df.to_csv(tmp.name, index=False, quoting=csv.QUOTE_ALL)
#         return tmp.name, df.columns.tolist(), df
#     except Exception as exc:
#         st.error(f"Pre-processing failed: {exc}")
#         return None, None, None

# def extract_python(llm_response: str) -> str:
#     m = PYTHON_BLOCK.search(llm_response)
#     return m.group(1) if m else ""

# def upload_to_sandbox(sb: "Sandbox", file) -> str:
#     sandbox_path = f"./{file.name}"
#     sb.files.write(sandbox_path, file.getvalue())
#     file.seek(0)
#     return sandbox_path

# def execute_in_sandbox(sb: "Sandbox", code: str) -> Tuple[Optional[List[Any]], Optional[str]]:
#     with st.spinner("🔧 Executing Python in sandbox…"):
#         run = sb.run_code(code)
#     return (None, run.error) if run.error else (run.results, None)

# # NEW: robust sandbox opener compatible with old/new E2B APIs
# from contextlib import contextmanager
# @contextmanager
# def open_sandbox(e2b_key: str):
#     """
#     Open an E2B sandbox that works across API versions.
#     - New API: Sandbox.create() and API key from env
#     - Old API: Sandbox(api_key=...) supported
#     """
#     if not _HAVE_E2B:
#         raise RuntimeError("E2B is not installed. Install with: pip install e2b-code-interpreter")

#     sb = None
#     try:
#         # Ensure env var for new API
#         if e2b_key:
#             os.environ["E2B_API_KEY"] = e2b_key

#         # Prefer new API if available
#         if hasattr(Sandbox, "create"):
#             sb = Sandbox.create()
#         else:
#             # Backward compatibility
#             try:
#                 sb = Sandbox(api_key=e2b_key) if e2b_key else Sandbox()
#             except TypeError:
#                 # Some builds removed api_key param entirely; fall back to env-only
#                 sb = Sandbox()

#         yield sb
#     finally:
#         try:
#             if sb is not None:
#                 if hasattr(sb, "close"):
#                     sb.close()
#         except Exception:
#             pass

# def visual_agent(query: str, dataset_infos: List[dict], tg_key: str, tg_model: str, e2b_key: str, max_retries: int = 2) -> Tuple[str, Optional[List[Any]]]:
#     if not (_HAVE_TOGETHER and _HAVE_E2B):
#         st.error("Together or E2B packages not installed. Install with: pip install together e2b-code-interpreter")
#         return "", None
#     if not (tg_key and e2b_key):
#         st.error("Provide Together and E2B keys in the sidebar.")
#         return "", None

#     def llm_call(msgs: List[dict]) -> str:
#         client = Together(api_key=tg_key)
#         resp = client.chat.completions.create(model=tg_model, messages=msgs)
#         return resp.choices[0].message.content

#     with open_sandbox(e2b_key) as sb:
#         # Upload datasets & mapping
#         for info in dataset_infos:
#             info["sb_path"] = upload_to_sandbox(sb, info["file"])
#         mapping_lines = [f"- **{d['name']}** ➜ `{d['sb_path']}`" for d in dataset_infos]

#         system_prompt = (
#             "You are a senior Python data-scientist and visualization expert.\n"
#             "Datasets available inside the sandbox:\n" + "\n".join(mapping_lines) +
#             "\n• Think step-by-step.\n"
#             "• Return exactly ONE ```python ...``` block that uses the paths above verbatim.\n"
#             "• After the code block, add a short explanation."
#         )
#         msgs = [{"role":"system","content":system_prompt}, {"role":"user","content":query}]
#         cumulative = ""

#         for attempt in range(max_retries + 1):
#             with st.spinner("🤖 LLM (Together) is writing Python…"):
#                 llm_text = llm_call(msgs)
#             cumulative += ("\n\n" + llm_text) if cumulative else llm_text

#             code = extract_python(llm_text)
#             if not code:
#                 st.warning("No ```python``` block found.")
#                 return cumulative, None

#             results, err = execute_in_sandbox(sb, code)
#             if err is None:
#                 return cumulative, results

#             st.warning(f"Sandbox error:\n\n```\n{err}\n```")
#             if attempt == max_retries:
#                 cumulative += "\n\n**Sandbox execution failed after retries. See error above.**"
#                 return cumulative, None

#             msgs.extend([
#                 {"role":"assistant","content":llm_text},
#                 {"role":"user","content":f"Your code errored:\n```\n{err}\n```\nFix and return ONLY a new ```python``` block."},
#             ])
#     return cumulative, None

# def build_phi_agent(csv_infos, openai_key: str) -> Optional["DuckDbAgent"]:
#     if not (_HAVE_PHI and pd is not None):
#         st.error("phi/pandas not available. Install with: pip install phi pandas")
#         return None
#     tables_meta = [
#         {"name": i["name"], "description": f"Dataset {os.path.basename(i['path'])}.", "path": i["path"]}
#         for i in csv_infos
#     ]
#     llm = PhiOpenAIChat(id="gpt-4o", api_key=openai_key)
#     return DuckDbAgent(
#         model=llm,
#         semantic_model=json.dumps({"tables": tables_meta}, indent=2),
#         tools=[PandasTools()],
#         markdown=True,
#         followups=False,
#         system_prompt=(
#             "You are an expert data analyst.\n"
#             "1) Write ONE SQL query in ```sql```\n"
#             "2) Execute it\n"
#             "3) Present the result plainly"
#         ),
#     )

# # ─────────────────────────────────────────────────────────────────────────────
# # Streamlit App
# # ─────────────────────────────────────────────────────────────────────────────
# def main():
#     st.set_page_config(page_title=APP_TITLE, page_icon="🧪", layout="wide")
#     init_session_state()

#     st.title(APP_TITLE)
#     st.caption(
#         "Chat with your LM Studio/OpenAI-compatible model. Add context from Vision and Data tools. "
#         "Everything in one place."
#     )

#     # Sidebar
#     sidebar_controls()

#     # Tabs
#     tab_chat, tab_vision, tab_data = st.tabs(["💬 Chat", "🖼️ Vision", "📊 Data"])

#     # ─────────────────────────────────────────────────────────────────────────
#     # CHAT TAB
#     # ─────────────────────────────────────────────────────────────────────────
#     with tab_chat:
#         with st.expander("📎 Context Drawer (latest tool outputs)", expanded=False):
#             colA, colB = st.columns(2)
#             with colA:
#                 st.subheader("Vision")
#                 if st.session_state.vision_last:
#                     st.markdown(f"**{st.session_state.vision_last_title or 'Image analysis'}**")
#                     st.markdown(st.session_state.vision_last)
#                     if st.button("➕ Inject this vision analysis into chat now"):
#                         st.session_state.context_image_enabled = True
#                         st.success("Vision analysis will be attached to the next chat turn.")
#                 else:
#                     st.caption("No vision analysis yet.")
#             with colB:
#                 st.subheader("Data")
#                 if st.session_state.data_chat_history:
#                     last_data = None
#                     for m in reversed(st.session_state.data_chat_history):
#                         if m.get("role") == "assistant":
#                             last_data = m["content"]; break
#                     if last_data:
#                         st.markdown(last_data)
#                         if st.button("➕ Inject this data answer into chat now"):
#                             st.session_state.context_data_enabled = True
#                             st.success("Data-summary will be attached to the next chat turn.")
#                     else:
#                         st.caption("No data-agent answer yet.")
#                 else:
#                     st.caption("No data-agent answer yet.")

#         # Render main chat history
#         for msg in st.session_state.chat_history:
#             with st.chat_message(msg["role"]):
#                 st.markdown(msg["content"])

#         # Chat input
#         user_input = st.chat_input("Ask about metallurgy/welding… or anything you like.")
#         if user_input:
#             st.session_state.chat_history.append({"role": "user", "content": user_input})
#             with st.chat_message("user"):
#                 st.markdown(user_input)

#             # Stream reply
#             with st.chat_message("assistant"):
#                 client: LLMClient = st.session_state.client
#                 messages = build_messages()

#                 def _generator():
#                     try:
#                         for chunk in client.stream_chat(
#                             model=st.session_state.model,
#                             messages=messages,
#                             temperature=st.session_state.temperature,
#                             top_p=st.session_state.top_p,
#                             max_tokens=st.session_state.max_tokens,
#                         ):
#                             yield chunk
#                     except Exception as e:
#                         yield f"\n\n**[Error]** {e}"

#                 full_text = st.write_stream(_generator())

#             st.session_state.chat_history.append({"role": "assistant", "content": full_text})

#             joined = [f"{m['role'].upper()}: {m['content']}" for m in st.session_state.chat_history]
#             approx_tokens = count_tokens("\n".join(joined))
#             st.caption(f"Approx. tokens in this session: **{approx_tokens}**")

#         st.divider()
#         c1, c2, c3 = st.columns([1, 1, 2])
#         if c1.button("🧹 Clear chat"):
#             st.session_state.chat_history = []
#             st.rerun()
#         download_buttons()
#         st.info(
#             f"Connected to **{st.session_state.base_url}** with model **{st.session_state.model}**.",
#             icon="🌐",
#         )

#     # ─────────────────────────────────────────────────────────────────────────
#     # VISION TAB
#     # ─────────────────────────────────────────────────────────────────────────
#     with tab_vision:
#         st.subheader("Upload an image and analyze it (General or Medical preset)")
#         col1, col2 = st.columns([1, 2])

#         with col1:
#             preset = st.radio("Analysis preset", ["General Vision", "Medical Imaging"], index=0)
#             st.session_state.vision_preset = preset

#             if not _HAVE_AGNO:
#                 st.warning("Vision agent optional: install `agno` and `google-generativeai`.", icon="ℹ️")

#             if not st.session_state.vision_api_key:
#                 st.info("Provide your Google API key in the sidebar to enable Gemini vision.", icon="🔑")

#         with col2:
#             uploaded = st.file_uploader(
#                 "Upload image (JPG/PNG/WebP/BMP/TIFF, or DICOM .dcm/.dicom)",
#                 type=["jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff", "dcm", "dicom"],
#                 accept_multiple_files=False,
#             )

#         image = None
#         if uploaded is not None:
#             image = load_image_any(uploaded)
#             if image is not None:
#                 st.image(image, caption=f"Preview — {uploaded.name}", use_container_width=True)
#                 analyze = st.button("🔍 Analyze Image", type="primary")
#                 if analyze:
#                     text = run_vision_analysis(image, st.session_state.vision_preset, title_hint=uploaded.name)
#                     if text:
#                         st.markdown("### 📋 Analysis Result")
#                         st.markdown("---")
#                         st.markdown(text)
#                         st.markdown("---")
#                         st.caption("Note: AI-generated; verify with appropriate domain experts when needed.")
#                         if st.button("📎 Send compact summary to Chat"):
#                             summary = text
#                             if "Compact Summary" in text:
#                                 summary = text.split("Compact Summary", 1)[-1].strip()
#                             st.session_state.vision_last = summary
#                             st.session_state.vision_last_title = f"Vision summary: {uploaded.name}"
#                             st.session_state.context_image_enabled = True
#                             st.success("Summary will be attached to the next chat turn.")
#             else:
#                 st.error("Could not load the uploaded file as an image.")

#         if uploaded is None:
#             st.info("👆 Upload an image to begin analysis.")

#     # ─────────────────────────────────────────────────────────────────────────
#     # DATA TAB
#     # ─────────────────────────────────────────────────────────────────────────
#     with tab_data:
#         st.subheader("Upload datasets and chat (SQL via φ / Visuals via Together+E2B)")

#         up_files = st.file_uploader(
#             "📁 Upload one or more CSV or Excel files",
#             type=["csv", "xls", "xlsx"], accept_multiple_files=True
#         )

#         # Handle uploads & preprocessing
#         if up_files and st.session_state.files_hash != hash(tuple(f.name for f in up_files)):
#             infos: List[dict] = []
#             for f in up_files:
#                 path, cols, df = preprocess_and_save(f)
#                 if path:
#                     infos.append({
#                         "file": f,
#                         "path": path,
#                         "name": re.sub(r"\W|^(?=\d)", "_", os.path.splitext(f.name)[0]),
#                         "columns": cols,
#                         "df": df,
#                     })
#             if infos:
#                 st.session_state.update({
#                     "datasets": infos,
#                     "data_prepared": True,
#                     "files_hash": hash(tuple(f.name for f in up_files)),
#                     "phi_agent": build_phi_agent(infos, st.session_state.phi_openai_key) if st.session_state.phi_openai_key else None,
#                 })

#         # Preview datasets
#         if st.session_state.data_prepared and st.session_state.datasets:
#             st.markdown("#### Dataset previews (first 8 rows)")
#             for d in st.session_state.datasets:
#                 with st.expander(f"📄 {d['file'].name}  (table `{d['name']}`)"):
#                     if pd is not None:
#                         st.dataframe(d["df"].head(8), use_container_width=True)
#                     st.caption(f"Columns: {', '.join(d['columns'])}")
#         else:
#             st.info("⬆️ Upload one or more datasets to start data chat.")

#         # Data chat area
#         if st.session_state.data_prepared:
#             # Render data-chat history
#             for m in st.session_state.data_chat_history:
#                 with st.chat_message(m["role"]):
#                     st.markdown(m["content"])
#                     for r in m.get("vis_results", []):
#                         typ = r.get("type")
#                         if typ == "image":
#                             st.image(r["data"])
#                         elif typ == "matplotlib":
#                             st.pyplot(r["data"])
#                         elif typ == "plotly":
#                             st.plotly_chart(r["data"])
#                         elif typ == "table":
#                             st.dataframe(r["data"])
#                         else:
#                             st.write(r["data"])

#             prompt = st.chat_input("Ask a question about your data (say 'plot'/'chart' to trigger visuals)…")
#             if prompt:
#                 st.session_state.data_chat_history.append({"role": "user", "content": prompt})
#                 with st.chat_message("user"):
#                     st.markdown(prompt)

#                 agent_type = router_agent(prompt)
#                 # Visual branch
#                 if agent_type == "visual":
#                     if not (st.session_state.tg_key and st.session_state.e2b_key):
#                         st.error("Provide Together and E2B keys in the sidebar.")
#                     else:
#                         text, results = visual_agent(
#                             query=prompt,
#                             dataset_infos=st.session_state.datasets,
#                             tg_key=st.session_state.tg_key,
#                             tg_model=st.session_state.tg_model_id,
#                             e2b_key=st.session_state.e2b_key,
#                         )
#                         msg = {"role": "assistant", "content": text or "(no text)", "vis_results": []}
#                         with st.chat_message("assistant"):
#                             st.markdown(text or "(no text)")
#                             if results:
#                                 from io import BytesIO
#                                 res_count = 0
#                                 for obj in results:
#                                     res_count += 1
#                                     if getattr(obj, "png", None):
#                                         try:
#                                             raw = base64.b64decode(obj.png)
#                                             img = PILImage.open(BytesIO(raw))
#                                             st.image(img)
#                                             msg["vis_results"].append({"type":"image", "data": img})
#                                         except Exception:
#                                             st.write(obj)
#                                             msg["vis_results"].append({"type":"text", "data": str(obj)})
#                                     elif getattr(obj, "figure", None):
#                                         st.pyplot(obj.figure)
#                                         msg["vis_results"].append({"type":"matplotlib", "data": obj.figure})
#                                     elif getattr(obj, "show", None):
#                                         st.plotly_chart(obj)
#                                         msg["vis_results"].append({"type":"plotly", "data": obj})
#                                     elif pd is not None and isinstance(obj, (pd.DataFrame, pd.Series)):
#                                         st.dataframe(obj)
#                                         msg["vis_results"].append({"type":"table", "data": obj})
#                                     else:
#                                         st.write(obj)
#                                         msg["vis_results"].append({"type":"text", "data": str(obj)})
#                                 if res_count == 0:
#                                     st.caption("No objects returned by sandbox.")
#                         st.session_state.data_chat_history.append(msg)
#                 # Analyst branch
#                 else:
#                     if not st.session_state.phi_openai_key:
#                         st.error("Provide an OpenAI API key for φ in the sidebar.")
#                     else:
#                         if st.session_state.phi_agent is None:
#                             st.session_state.phi_agent = build_phi_agent(st.session_state.datasets, st.session_state.phi_openai_key)
#                         if st.session_state.phi_agent is None:
#                             st.stop()
#                         with st.spinner("🧠 φ-agent (DuckDB) is thinking…"):
#                             run = st.session_state.phi_agent.run(prompt)
#                         answer = run.content if hasattr(run, "content") else str(run)
#                         with st.chat_message("assistant"):
#                             st.markdown(answer)
#                         st.session_state.data_chat_history.append({"role": "assistant", "content": answer})

#             if st.session_state.data_chat_history:
#                 if st.button("📎 Send latest data-agent answer to Chat"):
#                     for m in reversed(st.session_state.data_chat_history):
#                         if m.get("role") == "assistant":
#                             st.session_state.context_data_enabled = True
#                             st.success("Latest data answer will be attached to the next chat turn.")
#                             break

#         missing = []
#         if pd is None: missing.append("pandas")
#         if not _HAVE_PHI: missing.append("phi")
#         if not _HAVE_TOGETHER: missing.append("together")
#         if not _HAVE_E2B: missing.append("e2b-code-interpreter")
#         if missing:
#             st.info("Optional data stack components missing: " + ", ".join(missing))

# # ─────────────────────────────────────────────────────────────────────────────
# # Entrypoint
# # ─────────────────────────────────────────────────────────────────────────────
# if __name__ == "__main__":
#     main()








#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Advanced Unified Chat — Streamlit (E2B-compatible) + DeepResearch (Agno+Composio+Together)

One app, four powers:
1) 💬 LM Studio / OpenAI-compatible chat (streaming, multi-turn, system prompt, downloads)
2) 🖼️ Vision analysis (Gemini via agno) — general or medical presets, injectable into chat
3) 📊 Data agent (DuckDB/φ for SQL + Together+E2B for auto Python plots) — previews & chat, injectable into chat
4) 🔍 AI DeepResearch Agent (Agno + Composio + Together AI) — question generation, web-only research (Tavily+Perplexity),
   and Google Docs report compilation

───────────────────────────────────────────────────────────────────────────────
Install (pick what you need):
  pip install --upgrade streamlit openai tiktoken pillow pandas duckdb
  pip install --upgrade agno google-generativeai             # (Vision)
  pip install --upgrade phi together e2b-code-interpreter    # (Data visuals)
  pip install --upgrade plotly pydicom                       # (optional: plots, DICOM preview)
  # DeepResearch stack:
  pip install --upgrade python-dotenv agno composio-agno together

Run:
  streamlit run AdvancedChat_DeepResearch.py
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
from datetime import datetime, timezone

# Load .env early so both modules see env defaults
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

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

# Alias Together SDK (data visuals branch) to avoid name clash with Agno's Together model
try:
    from together import Together as TogetherSDK
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

# DeepResearch dependencies (Agno Together model + Composio tools)
try:
    # NOTE: We alias the Agno Together model to avoid confusion with TogetherSDK above
    from agno.models.together import Together as AgnoTogetherModel
    from agno.agent import Agent as AgnoAgentDR  # same class as AgnoAgent; aliased for clarity
    _HAVE_AGNO_DR = True
except Exception:
    _HAVE_AGNO_DR = False

try:
    from composio_agno import ComposioToolSet, Action
    _HAVE_COMPOSIO_AGNO = True
except Exception:
    _HAVE_COMPOSIO_AGNO = False

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

# ─────────────────────────────────────────────────────────────────────────────
# Defaults & configuration
# ─────────────────────────────────────────────────────────────────────────────
APP_TITLE = "🧪📊 Unified Advanced Chat + 🔍 DeepResearch"

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

# DeepResearch env defaults
ENV_DR_TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "")
ENV_DR_COMPOSIO_API_KEY = os.getenv("COMPOSIO_API_KEY", "")

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
        "context_buffer": [],

        # vision
        "vision_api_key": ENV_GOOGLE_API_KEY,
        "vision_preset": "General Vision",
        "vision_last": None,
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

        # DeepResearch (Agno+Composio) session keys — namespaced with 'dr_'
        "dr_together_api_key": ENV_DR_TOGETHER_API_KEY,
        "dr_composio_api_key": ENV_DR_COMPOSIO_API_KEY,
        "dr_model_id": "Qwen/Qwen3-235B-A22B-fp8-tput",
        "dr_temperature": 0.2,

        "dr_questions": [],
        "dr_question_answers": [],
        "dr_report_content": "",
        "dr_report_doc_url": "",
        "dr_research_complete": False,
        "dr_topic": "",
        "dr_domain": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    if st.session_state.client is None:
        st.session_state.client = LLMClient(st.session_state.base_url, st.session_state.api_key)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar controls (Chat/Data/Vision + DeepResearch config)
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

        st.markdown("### 🎛️ Together Model (Visual Agent)")
        TG_MODELS = {
            "Meta-Llama 3.1-405B": "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
            "DeepSeek V3":         "deepseek-ai/DeepSeek-V3",
            "Qwen 2.5-7B":         "Qwen/Qwen2.5-7B-Instruct-Turbo",
            "Meta-Llama 3.3-70B":  "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        }
        model_human = st.selectbox("Together model", list(TG_MODELS.keys()),
                                   index=list(TG_MODELS.keys()).index(st.session_state.tg_model_human))
        tg_model_id = TG_MODELS[model_human]

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

        # ─────────────────────────────────────────────────────────────────────
        # DeepResearch configuration (preserves original sidebar style)
        # ─────────────────────────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("🔍 DeepResearch — Configuration")

        dr_together_api_key = st.text_input(
            "Together AI API Key (DeepResearch)",
            value=st.session_state.dr_together_api_key, type="password",
            help="Create/manage key: https://together.ai"
        )
        dr_composio_api_key = st.text_input(
            "Composio API Key",
            value=st.session_state.dr_composio_api_key, type="password",
            help="Create/manage key: https://composio.ai"
        )

        st.markdown("##### Model")
        dr_model_id = st.text_input(
            "Together Model ID (Agno)",
            value=st.session_state.dr_model_id,
            help="Change if you prefer a different Together-hosted model for DeepResearch."
        )
        dr_temperature = st.slider(
            "Temperature (DeepResearch)",
            min_value=0.0, max_value=1.0, value=float(st.session_state.dr_temperature), step=0.05,
            help="Lower is more factual/consistent."
        )

        st.markdown("##### Tools Used (via Composio)")
        st.markdown("- 🔍 Tavily Search (COMPOSIO_SEARCH_TAVILY_SEARCH)")
        st.markdown("- 🧠 Perplexity AI (PERPLEXITYAI_PERPLEXITY_AI_SEARCH)")
        st.markdown("- 📄 Google Docs (GOOGLEDOCS_CREATE_DOCUMENT_MARKDOWN)")
        st.info(
            "DeepResearch uses web tools for every answer. "
            "All answers include direct source links and dates—no hidden reasoning."
        )

        if st.button("Apply DeepResearch Settings"):
            st.session_state.update({
                "dr_together_api_key": dr_together_api_key,
                "dr_composio_api_key": dr_composio_api_key,
                "dr_model_id": dr_model_id,
                "dr_temperature": float(dr_temperature),
            })
            st.success("DeepResearch settings applied.", icon="✅")

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
# Vision (image analysis)
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

Role & scope

Analyze static coronary angiography frames and determine if a stenosis exists, where it is, and its categorical severity.

Output findings + brief evidence. This is informational and not a medical diagnosis.

Key visual appearance of a true stenosis (what to look for)
A real, fixed coronary stenosis usually shows a combination of:

Luminal “waist”: a focal narrowest point (MLD) compared with adjacent reference lumen (RVD) with clear shoulders on each side.

Edge characteristics: irregular, eccentric or scalloped walls (atherosclerosis) vs smooth concentric narrowing (spasm/bridge).

Caliber mismatch: abrupt step-down not explained by normal taper or branch loss.

Hemodynamic markers (supportive, when visible in a frame): post-stenotic dilatation, contrast hold-up/hang-up just proximal to the lesion, delayed/weak distal opacification, or visible collateral filling.

Contextual clues: calcific flecks/rims overlying the lesion; stent edges near a target; bifurcation carina involvement for LM/LAD/LCx lesions.

Where it happens (localization rules)

Ostial: within ~3 mm of vessel origin (LM, LCx, RCA). Look for waist right at the mouth ± catheter damping.

Proximal / Mid / Distal: divide vessel length into thirds from origin to major distal bifurcation.

Bifurcation: narrowing that straddles parent and daughter vessels at the carina; record which arms are diseased (Medina concept).

Side-branch ostium (e.g., D1, OM): focal waist right after the take-off.

How NOT to get fooled (common mimics & how to tell them apart)

Side-branch take-off: a V-shaped “gap” at a branch that re-expands immediately; true stenosis persists distal to the branch and has shoulders.

Vessel overlap / crossing: stacked lumens or crossing lines create a dark line—edges don’t follow a single centerline; look for changing wall continuity.

Foreshortening: very short segment with exaggerated curvature; diameters look smaller everywhere—no discrete waist/shoulders.

Under-opacification / streaming: patchy or striped contrast early after injection; heterogeneous density without crisp walls; distal segment opacifies normally a moment later (in cine).

Catheter-induced spasm/ostial pseudostenosis: smooth concentric narrowing right at the ostium with the catheter sitting deep; often resolves or lessens after pullback or nitrates.

Diffuse tapering/negative remodeling: long, uniform small caliber without a focal waist → describe as diffuse atherosclerosis rather than a discrete stenosis.

Myocardial bridging: dynamic systolic squeeze with diastolic normalization (needs cine); a single still frame may not prove it—mark unsafe if suspected only by shape.

Valvular plane/epicardial fat lines or ribs: extravascular densities that do not trace the vessel across frames.

Stent struts: radiopaque mesh outlines; don’t call the struts a stenosis—assess in-stent lumen and edge segments for a true waist.

Calcification: bright nodular/linear radiopacities; calcification alone is not stenosis—confirm a luminal narrowing.

Quantifying severity (always map to categories below; avoid raw % unless asked)

Compute % diameter stenosis ≈ (1 − MLD/RVD) × 100 using the nearest disease-free reference (usually proximal; use distal if proximal is diseased).

If both sides diseased, use a visual best reference segment or expected normal for that vessel size.

Categorical grades (use verbatim):

No stenosis: <20%

Mild stenosis: 20–49%

Moderate stenosis: 50–69%

Severe stenosis: 70–99%

Total occlusion: 100% (blunt/tapered stump with no antegrade distal opacification; distal bed may fill retrograde via collaterals).

For bifurcation lesions, grade main vessel and side-branch ostium separately; note carina involvement.

If the frame is inadequate to size references (overlap/blur/poor fill), return “Unsafe estimate from this frame.”

Step-by-step search algorithm (what the model should do)

Verify modality & projection; note quality issues (blur, overlap, under-filling).

Trace centerlines of LM, LAD (± diagonals), LCx (± OM), RCA (± PDA/PL); mark ostia and divide each into proximal/mid/distal thirds.

Build a diameter profile along each vessel: detect local minima (candidate lesions) and adjacent reference maxima.

Reject mimics using the rules above (branch take-off, overlap, streaming, spasm).

For each accepted candidate, measure MLD vs RVD, check for shoulders, edge irregularity, post-stenotic features, and distal opacification.

Assign location & grade (ostial/proximal/mid/distal; main vs side branch).

Summarize evidence (“waist at mid-LAD with irregular edges; post-stenotic dilatation; distal runoff delayed”).

State limitations when any reference segment is questionable.

What “percentage” means here (practical notes)

Use diameter reduction (not area).

Eccentric lesions: choose MLD across the narrowest axis; if edges are fuzzy, bias toward higher uncertainty and keep to category labels.

Ostial LM/RCA/LCx: RVD may be partly within the aortic root; use the first normal segment just distal to the ostium as reference.

Stented segments: use expected stent inner diameter or adjacent normal as RVD; look for in-stent restenosis (focal or diffuse).

Long lesions: if the majority is ≥50% with some focal severe spots, label diffuse moderate with focal severe component.

Output you must produce (short & structured)

Summary: 1–2 sentences about presence/absence of significant stenosis.

Report by vessel: LM, LAD (± D1/D2), LCx (± OM), RCA (± PDA/PL); for each: location + grade.

Evidence: bullets linking each stenosis to where/how it is seen (waist, shoulders, calcification, post-stenotic changes, distal flow).

Limitations: quality issues; remind that this is a single-frame estimate.

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

def safe_numeric_coercion(df: "pd.DataFrame") -> "pd.DataFrame":
    """Future-proof replacement for errors='ignore' with a conservative policy."""
    if pd is None:
        return df
    for col in df.columns:
        if df[col].dtype == "object":
            # Try numeric coercion; keep only if it meaningfully converts values.
            try:
                tmp = pd.to_numeric(df[col], errors="coerce")
                # If we get at least ~80% non-null after coercion, adopt it.
                if tmp.notna().mean() >= 0.8:
                    df[col] = tmp
            except Exception:
                # Leave column as-is if conversion fails wholesale.
                pass
    return df

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

        # Conservative type coercion (no errors='ignore')
        df = safe_numeric_coercion(df)

        # Persist to temp CSV for DuckDB
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

# NEW: robust sandbox opener compatible with old/new E2B APIs
from contextlib import contextmanager
@contextmanager
def open_sandbox(e2b_key: str):
    """
    Open an E2B sandbox that works across API versions.
    - New API: Sandbox.create() and API key from env
    - Old API: Sandbox(api_key=...) supported
    """
    if not _HAVE_E2B:
        raise RuntimeError("E2B is not installed. Install with: pip install e2b-code-interpreter")

    sb = None
    try:
        # Ensure env var for new API
        if e2b_key:
            os.environ["E2B_API_KEY"] = e2b_key

        # Prefer new API if available
        if hasattr(Sandbox, "create"):
            sb = Sandbox.create()
        else:
            # Backward compatibility
            try:
                sb = Sandbox(api_key=e2b_key) if e2b_key else Sandbox()
            except TypeError:
                # Some builds removed api_key param entirely; fall back to env-only
                sb = Sandbox()

        yield sb
    finally:
        try:
            if sb is not None:
                if hasattr(sb, "close"):
                    sb.close()
        except Exception:
            pass

def visual_agent(query: str, dataset_infos: List[dict], tg_key: str, tg_model: str, e2b_key: str, max_retries: int = 2) -> Tuple[str, Optional[List[Any]]]:
    if not (_HAVE_TOGETHER and _HAVE_E2B):
        st.error("Together or E2B packages not installed. Install with: pip install together e2b-code-interpreter")
        return "", None
    if not (tg_key and e2b_key):
        st.error("Provide Together and E2B keys in the sidebar.")
        return "", None

    def llm_call(msgs: List[dict]) -> str:
        client = TogetherSDK(api_key=tg_key)
        resp = client.chat.completions.create(model=tg_model, messages=msgs)
        return resp.choices[0].message.content

    with open_sandbox(e2b_key) as sb:
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
# DeepResearch (Agno + Composio + Together AI) — helpers
# Preserves functionality from the provided `app.py` with careful namespacing.
# ─────────────────────────────────────────────────────────────────────────────
DR_UTC_NOW_ISO = datetime.now(timezone.utc).strftime("%Y-%m-%d")

def DR_strip_think_blocks(text: str) -> str:
    """
    Remove any <think>...</think> blocks to ensure hidden reasoning is never surfaced.
    Also strips HTML comments if leaked by a provider.
    """
    if not text:
        return text
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    return text.strip()

def DR_enforce_five_questions(raw_list):
    """
    Normalize to exactly five non-empty questions. If more, take first five; if fewer, pad with placeholders.
    """
    cleaned = [q.strip().lstrip("-").lstrip("*").strip() for q in raw_list if q and q.strip()]
    cleaned = [re.sub(r"^\s*\d+\s*[\.\)]\s*", "", q).strip() for q in cleaned]
    if len(cleaned) > 5:
        cleaned = cleaned[:5]
    while len(cleaned) < 5:
        cleaned.append(f"Placeholder question {len(cleaned)+1}?")
    return cleaned

def DR_initialize_agents(together_key: str, composio_key: str):
    llm = AgnoTogetherModel(
        id=st.session_state.dr_model_id,
        api_key=together_key,
        temperature=float(st.session_state.dr_temperature)
    )
    toolset = ComposioToolSet(api_key=composio_key)
    composio_tools = toolset.get_tools(actions=[
        Action.COMPOSIO_SEARCH_TAVILY_SEARCH,
        Action.PERPLEXITYAI_PERPLEXITY_AI_SEARCH,
        Action.GOOGLEDOCS_CREATE_DOCUMENT_MARKDOWN
    ])
    return llm, composio_tools

def DR_question_generator_agent(llm: "AgnoTogetherModel") -> "AgnoAgentDR":
    """
    Agent that ONLY generates 5 yes/no research questions as a numbered list.
    """
    return AgnoAgentDR(
        name="Question Generator",
        model=llm,
        instructions=(
            "You are a world-class research planner.\n"
            "TASK: Generate EXACTLY 5 specific, decision-oriented YES/NO research questions "
            "for the given topic and domain.\n"
            "CRITERIA:\n"
            " - Each question must be unambiguous and empirically answerable from web sources.\n"
            " - Focus on current, verifiable facts or policies—not opinions.\n"
            " - Keep each question to one sentence.\n"
            "OUTPUT:\n"
            " - Return ONLY the 5 questions as a numbered list (1. ... 2. ... 3. ... 4. ... 5.).\n"
            " - Do NOT include explanations, chain-of-thought, or any extra text."
        )
    )

def DR_research_agent(llm: "AgnoTogetherModel", composio_tools) -> "AgnoAgentDR":
    """
    Agent that answers a single question using ONLY web tools and returns a concise, sourced answer.
    """
    return AgnoAgentDR(
        name="DeepResearch",
        model=llm,
        tools=[composio_tools],
        instructions=(
            "ROLE: You are a meticulous research analyst.\n"
            "HARD RULES:\n"
            " - You MUST use web tools for evidence (Tavily search + Perplexity AI) — do not rely on prior knowledge.\n"
            " - Always cross-check at least 2 independent sources; prefer primary or authoritative references.\n"
            " - Include direct, clickable URLs and the source publication dates.\n"
            " - If evidence conflicts or is insufficient, say so explicitly and explain briefly.\n"
            " - NEVER include hidden reasoning or chain-of-thought. Do not use placeholders.\n"
            f" - Today is {DR_UTC_NOW_ISO} (UTC). Treat 'recent' as within the last 24 months unless the user specifies otherwise.\n"
            "TOOL USAGE:\n"
            " - First, call COMPOSIO_SEARCH_TAVILY_SEARCH to discover 5–10 relevant, recent sources.\n"
            " - Then, call PERPLEXITYAI_PERPLEXITY_AI_SEARCH to summarize/triangulate across those findings.\n"
            " - Cite at least 3 high-quality sources with URLs and dates.\n"
            "OUTPUT FORMAT (STRICT):\n"
            " - Start with **Answer: Yes/No/Unclear.**\n"
            " - Then provide a 2–5 sentence justification.\n"
            " - Add a 'Sources' section with a bulleted list of `[Publisher] Title — Date — URL` items.\n"
            " - No preambles, no chain-of-thought, no extra sections.\n"
        )
    )

def DR_report_compiler_agent(llm: "AgnoTogetherModel", composio_tools) -> "AgnoAgentDR":
    """
    Agent that compiles a professional report and creates a Google Doc via Composio.
    """
    return AgnoAgentDR(
        name="Report Compiler",
        model=llm,
        tools=[composio_tools],
        instructions=(
            "ROLE: You are a professional consultant compiling a McKinsey-style report.\n"
            "SOURCE MATERIAL: You will be given a topic, domain, and a set of question/answer sections "
            "that already include web citations.\n"
            "HARD RULES:\n"
            " - Use ONLY the provided Q/A content and perform minor smoothing/structuring; do not invent facts.\n"
            " - Preserve source links and include them inline as clickable URLs where relevant.\n"
            " - Do NOT include chain-of-thought or hidden reasoning.\n"
            "STRUCTURE (Markdown):\n"
            " 1. Executive Summary (5–8 sentences)\n"
            " 2. Research Analysis (one subsection per research question; analytical prose, not Q&A formatting)\n"
            " 3. Conclusion & Implications\n"
            "GOOGLE DOC:\n"
            " - You MUST call GOOGLEDOCS_CREATE_DOCUMENT_MARKDOWN to create a Google Doc.\n"
            " - Title: 'AI DeepResearch Report'\n"
            " - Content: the full Markdown report.\n"
            "OUTPUT:\n"
            " - After creating the doc, return ONLY a short confirmation with the doc title and a single line "
            "   'Google Doc: <URL>' where <URL> is the link returned by the tool.\n"
        )
    )

def DR_generate_questions(llm: "AgnoTogetherModel", topic: str, domain: str):
    agent = DR_question_generator_agent(llm)
    with st.spinner("🤖 Generating research questions..."):
        prompt = (
            f"Topic: {topic}\n"
            f"Domain: {domain}\n\n"
            "Generate exactly 5 yes/no research questions as instructed."
        )
        result = agent.run(prompt)
        text = DR_strip_think_blocks(getattr(result, "content", "") or str(result))
        raw_lines = [line.strip() for line in text.splitlines() if line.strip()]
        st.session_state.dr_questions = DR_enforce_five_questions(raw_lines)
    st.success("✅ Generated 5 research questions.")
    st.rerun()

def DR_answer_question(llm: "AgnoTogetherModel", composio_tools, topic: str, domain: str, question: str) -> str:
    agent = DR_research_agent(llm, composio_tools)
    query = (
        f"Research question (YES/NO): {question}\n"
        f"Topic: {topic}\n"
        f"Domain: {domain}\n"
        f"Today (UTC): {DR_UTC_NOW_ISO}\n\n"
        "Follow the HARD RULES and OUTPUT FORMAT exactly."
    )
    result = agent.run(query)
    return DR_strip_think_blocks(getattr(result, "content", "") or str(result))

def DR_compile_report(llm: "AgnoTogetherModel", composio_tools, topic: str, domain: str, qa_sections: list) -> tuple[str, str]:
    """
    Returns (report_confirmation_text, google_doc_url_if_any)
    """
    agent = DR_report_compiler_agent(llm, composio_tools)
    analysis_md = []
    for idx, qa in enumerate(qa_sections, start=1):
        analysis_md.append(f"### {idx}. {qa['question']}\n\n{qa['answer']}\n")
    analysis_blob = "\n".join(analysis_md)

    compile_prompt = (
        f"Topic: {topic}\nDomain: {domain}\n\n"
        f"Use the following question/answer sections to compile the report:\n\n{analysis_blob}\n\n"
        "Remember: You MUST create the Google Doc via GOOGLEDOCS_CREATE_DOCUMENT_MARKDOWN and then "
        "return ONLY the confirmation with the doc URL."
    )
    with st.spinner("📝 Compiling final report and creating Google Doc..."):
        result = agent.run(compile_prompt)
        content = DR_strip_think_blocks(getattr(result, "content", "") or str(result))
        url_match = re.search(r"https?://[^\s)>\]]+", content)
        doc_url = url_match.group(0) if url_match else ""
        return content, doc_url

# ─────────────────────────────────────────────────────────────────────────────
# Streamlit App
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="🧪", layout="wide")
    init_session_state()

    st.title(APP_TITLE)
    st.caption(
        "Chat with your LM Studio/OpenAI-compatible model. Add context from Vision and Data tools. "
        "Plus: Run a fully web-sourced DeepResearch workflow that compiles a Google Doc report."
    )

    # Sidebar
    sidebar_controls()

    # Tabs
    tab_chat, tab_vision, tab_data, tab_deep = st.tabs(["💬 Chat", "🖼️ Vision", "📊 Data", "🔍 DeepResearch"])

    # ─────────────────────────────────────────────────────────────────────────
    # CHAT TAB
    # ─────────────────────────────────────────────────────────────────────────
    with tab_chat:
        with st.expander("📎 Context Drawer (latest tool outputs)", expanded=False):
            colA, colB = st.columns(2)
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
                            summary = text
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
                                from io import BytesIO
                                res_count = 0
                                for obj in results:
                                    res_count += 1
                                    if getattr(obj, "png", None):
                                        try:
                                            raw = base64.b64decode(obj.png)
                                            img = PILImage.open(BytesIO(raw))
                                            st.image(img)
                                            msg["vis_results"].append({"type":"image", "data": img})
                                        except Exception:
                                            st.write(obj)
                                            msg["vis_results"].append({"type":"text", "data": str(obj)})
                                    elif getattr(obj, "figure", None):
                                        st.pyplot(obj.figure)
                                        msg["vis_results"].append({"type":"matplotlib", "data": obj.figure})
                                    elif getattr(obj, "show", None):
                                        st.plotly_chart(obj)
                                        msg["vis_results"].append({"type":"plotly", "data": obj})
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

            if st.session_state.data_chat_history:
                if st.button("📎 Send latest data-agent answer to Chat"):
                    for m in reversed(st.session_state.data_chat_history):
                        if m.get("role") == "assistant":
                            st.session_state.context_data_enabled = True
                            st.success("Latest data answer will be attached to the next chat turn.")
                            break

        missing = []
        if pd is None: missing.append("pandas")
        if not _HAVE_PHI: missing.append("phi")
        if not _HAVE_TOGETHER: missing.append("together")
        if not _HAVE_E2B: missing.append("e2b-code-interpreter")
        if missing:
            st.info("Optional data stack components missing: " + ", ".join(missing))

    # ─────────────────────────────────────────────────────────────────────────
    # DEEPRESEARCH TAB (Agno + Composio + Together AI)
    # ─────────────────────────────────────────────────────────────────────────
    with tab_deep:
        st.title("🔍 AI DeepResearch Agent (Agno + Composio)")
        st.header("Research Topic")

        col1, col2 = st.columns(2)
        with col1:
            topic = st.text_input("What topic would you like to research?", value=st.session_state.dr_topic, placeholder="American Tariffs")
        with col2:
            domain = st.text_input("What domain is this topic in?", value=st.session_state.dr_domain, placeholder="Politics, Economics, Technology, etc.")

        # Persist user edits
        st.session_state.dr_topic = topic
        st.session_state.dr_domain = domain

        # Initialize keys & tools when present (preserve original behavior)
        llm = None
        composio_tools = None
        keys_ready = bool(st.session_state.dr_together_api_key and st.session_state.dr_composio_api_key)

        if keys_ready and _HAVE_AGNO_DR and _HAVE_COMPOSIO_AGNO:
            try:
                llm, composio_tools = DR_initialize_agents(st.session_state.dr_together_api_key, st.session_state.dr_composio_api_key)
            except Exception as e:
                st.error(f"Failed to initialize model/tools: {e}")
        elif keys_ready and (not _HAVE_AGNO_DR or not _HAVE_COMPOSIO_AGNO):
            st.warning("DeepResearch dependencies missing. Install with: pip install agno composio-agno", icon="⚠️")

        # Buttons — enabled as soon as API keys are present (loosened conditions)
        colg1, colg2, colg3 = st.columns([1,1,1])
        with colg1:
            gen_btn = st.button("Generate Research Questions", type="primary", disabled=not (keys_ready and topic and domain))
        with colg2:
            start_btn = st.button("Start Research", disabled=not keys_ready)
        with colg3:
            compile_btn = st.button("Compile Final Report", disabled=not keys_ready)

        st.markdown("---")

        # Generate Questions
        if gen_btn:
            if not keys_ready:
                st.warning("⚠️ Enter your Together AI and Composio API keys first.")
            elif not (topic and domain):
                st.warning("⚠️ Please fill both Topic and Domain.")
            else:
                try:
                    DR_generate_questions(llm, topic, domain)
                except Exception as e:
                    st.error(f"Error generating questions: {e}")

        # Display Questions (if any)
        if st.session_state.dr_questions:
            st.subheader("Research Questions")
            for i, q in enumerate(st.session_state.dr_questions, start=1):
                st.markdown(f"**{i}. {q}**")

        # Start Research
        if start_btn:
            if not keys_ready:
                st.warning("⚠️ Enter your Together AI and Composio API keys first.")
            elif not (topic and domain):
                st.warning("⚠️ Please fill both Topic and Domain.")
            elif len(st.session_state.dr_questions) != 5:
                st.warning("⚠️ Please generate exactly 5 questions before starting research.")
            else:
                st.subheader("Research Results")
                st.session_state.dr_question_answers = []
                progress = st.progress(0)
                for i, q in enumerate(st.session_state.dr_questions, start=1):
                    progress.progress((i - 1) / 5.0)
                    with st.spinner(f"🔍 Researching question {i} of 5..."):
                        try:
                            ans = DR_answer_question(llm, composio_tools, topic, domain, q)
                        except Exception as e:
                            ans = (
                                "**Answer: Unclear.**\n\n"
                                f"An error occurred while researching: {e}\n\n"
                                "**Sources:**\n- None"
                            )
                        st.session_state.dr_question_answers.append({"question": q, "answer": ans})
                        st.markdown(f"### Question {i}\n**{q}**")
                        st.markdown(ans)
                    progress.progress(i / 5.0)
                st.success("✅ Research completed for all 5 questions.")
                st.rerun()

        # Compile Report
        if compile_btn:
            if not keys_ready:
                st.warning("⚠️ Enter your Together AI and Composio API keys first.")
            elif len(st.session_state.dr_question_answers) != 5:
                st.warning("⚠️ Please complete research on all 5 questions before compiling the report.")
            else:
                try:
                    confirmation, doc_url = DR_compile_report(
                        llm, composio_tools, topic, domain, st.session_state.dr_question_answers
                    )
                    st.session_state.dr_report_content = confirmation
                    st.session_state.dr_report_doc_url = doc_url
                    st.session_state.dr_research_complete = True
                    st.success("✅ Final report compiled and Google Doc created.")
                except Exception as e:
                    st.error(f"Error compiling report: {e}")

        # Display Final Report confirmation (with Google Doc link)
        if st.session_state.dr_research_complete and st.session_state.dr_report_content:
            st.header("Final Report")
            st.success("Your report has been compiled and a Google Doc has been created.")
            if st.session_state.dr_report_doc_url:
                st.markdown(f"📄 **Google Doc:** [{st.session_state.dr_report_doc_url}]({st.session_state.dr_report_doc_url})")
            with st.expander("View Report Creation Confirmation", expanded=True):
                st.markdown(st.session_state.dr_report_content)

        # Prior results (if user navigates without re-running)
        if len(st.session_state.dr_question_answers) > 0 and not st.session_state.dr_research_complete:
            st.subheader("Previous Research Results")
            for i, qa in enumerate(st.session_state.dr_question_answers, start=1):
                with st.expander(f"Question {i}: {qa['question']}"):
                    st.markdown(qa["answer"])

        # Helpful footer
        if not keys_ready:
            st.warning("⚠️ Please enter your Together AI and Composio API keys in the sidebar to get started.")

        st.markdown("---")
        with st.expander("How It Works"):
            st.markdown(
                "- **Define Topic/Domain** → Generate exactly 5 yes/no questions.\n"
                "- **Start Research** → For each question, the agent uses **Tavily** and **Perplexity** via Composio, "
                "returning a concise answer with **direct source links** and **publication dates**.\n"
                "- **Compile Final Report** → A polished report is created and saved to **Google Docs** with a shareable link."
            )

# ─────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()

























