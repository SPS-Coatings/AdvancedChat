#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Streamlit chat UI for LM Studio (OpenAI-compatible) — NO RAG, pure chat.

Now with **R1-style reasoning separation**:
- Detects `<think> ... </think>` blocks produced by chain-of-thought models (e.g., DeepSeek R1 variants).
- Hides the raw thinking text for safety, shows only the **final answer** stream in the chat bubble.
- Displays a compact **Reasoning (hidden)** panel with an optional one-click **High-level explanation** (summary) that does NOT reveal chain-of-thought.

Features:
- Multi-turn chat with history
- Live streaming of tokens (final answer only; thinking hidden)
- Sidebar controls: base URL, API key, model, temperature, top_p, max_tokens
- System prompt editor
- Download transcript (JSON/Markdown)
- Works with LM Studio (Enable "Local server (OpenAI-compatible)" in LM Studio)
- Also works with any OpenAI-compatible server (OpenAI, vLLM, etc.)

Environment variables (optional; also configurable in sidebar):
- LMSTUDIO_BASE_URL or OPENAI_BASE_URL  (e.g., "http://<public-ip-or-domain>:1234/v1")
- LMSTUDIO_API_KEY  or OPENAI_API_KEY   (LM Studio often ignores but requires a value; "lm-studio" is fine)
"""

import os
import json
import time
from typing import List, Dict, Generator, Optional, Tuple

import streamlit as st

# Use OpenAI SDK for maximum compatibility with LM Studio's OpenAI adapter
try:
    from openai import OpenAI
except Exception as e:
    raise RuntimeError("Missing dependency. Install with: pip install --upgrade openai streamlit") from e

# Optional token counting (best-effort)
try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")
    def count_tokens(text: str) -> int:
        return len(_ENC.encode(text or ""))
except Exception:
    def count_tokens(text: str) -> int:
        return max(1, len((text or "").split()))


# -----------------------------------------------------------------------------
# Defaults & configuration
# -----------------------------------------------------------------------------
def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default)

def normalize_base_url(url: str) -> str:
    """Ensure base URL ends with /v1 (no trailing slash after)."""
    if not url:
        return url
    u = url.strip()
    while u.endswith("/"):
        u = u[:-1]
    if not u.endswith("/v1"):
        u = u + "/v1"
    return u

DEFAULT_BASE_URL = normalize_base_url(
    _env("LMSTUDIO_BASE_URL") or _env("OPENAI_BASE_URL") or "http://localhost:1234/v1"
)

DEFAULT_API_KEY = _env("LMSTUDIO_API_KEY") or _env("OPENAI_API_KEY") or "lm-studio"

DEFAULT_MODEL          = _env("LMSTUDIO_MODEL", "qwen2.5-7b-instruct")
DEFAULT_TEMPERATURE    = float(_env("LMSTUDIO_TEMPERATURE", "0.2"))
DEFAULT_TOP_P          = float(_env("LMSTUDIO_TOP_P", "0.95"))
DEFAULT_MAX_TOKENS     = int(_env("LMSTUDIO_MAX_TOKENS", "2048"))
DEFAULT_SYSTEM_PROMPT  = _env(
    "LMSTUDIO_SYSTEM_PROMPT",
    "You are a rigorous, practical metallurgy and welding assistant. "
    "Explain clearly and concisely, show calculations step-by-step when relevant, "
    "cite standards/specs by name and number if you mention them, and flag uncertainties. "
    "When users give incomplete parameters, state reasonable assumptions explicitly."
)

APP_TITLE = "Metallurgy/Welding Chat — LM Studio (No RAG)"


# -----------------------------------------------------------------------------
# LLM Client wrapper
# -----------------------------------------------------------------------------
class LLMClient:
    def __init__(self, base_url: str, api_key: str):
        self._base_url = normalize_base_url(base_url.strip())
        self._api_key  = api_key.strip()
        self._client   = OpenAI(base_url=self._base_url, api_key=self._api_key)

    @property
    def base_url(self) -> str:
        return self._base_url

    def set_base_url(self, base_url: str):
        self._base_url = normalize_base_url(base_url.strip())
        self._client   = OpenAI(base_url=self._base_url, api_key=self._api_key)

    def set_api_key(self, api_key: str):
        self._api_key  = api_key.strip()
        self._client   = OpenAI(base_url=self._base_url, api_key=self._api_key)

    def list_models(self) -> List[str]:
        """
        Best-effort model listing. LM Studio typically supports GET /v1/models.
        """
        try:
            models = self._client.models.list()
            # OpenAI SDK returns a paged list of Model objects; we map to IDs.
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
        """
        Stream assistant tokens. Yields raw chunks (may include <think>...</think> for R1-like models).
        """
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

    def complete_once(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        top_p: float = 0.95,
        max_tokens: int = 512,
    ) -> str:
        """
        Non-streaming helper (used for optional high-level explanation).
        """
        resp = self._client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=float(temperature),
            top_p=float(top_p),
            max_tokens=int(max_tokens),
            stream=False,
        )
        return resp.choices[0].message.content or ""


# -----------------------------------------------------------------------------
# Streamlit helpers
# -----------------------------------------------------------------------------
def init_session_state():
    if "chat_history" not in st.session_state:
        # Store as list of {"role": "user"/"assistant", "content": "..."}
        st.session_state.chat_history = []
    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT
    if "model" not in st.session_state:
        st.session_state.model = DEFAULT_MODEL
    if "temperature" not in st.session_state:
        st.session_state.temperature = DEFAULT_TEMPERATURE
    if "top_p" not in st.session_state:
        st.session_state.top_p = DEFAULT_TOP_P
    if "max_tokens" not in st.session_state:
        st.session_state.max_tokens = DEFAULT_MAX_TOKENS
    if "base_url" not in st.session_state:
        st.session_state.base_url = DEFAULT_BASE_URL
    if "api_key" not in st.session_state:
        st.session_state.api_key = DEFAULT_API_KEY
    if "client" not in st.session_state:
        st.session_state.client = LLMClient(st.session_state.base_url, st.session_state.api_key)


def sidebar_controls():
    st.sidebar.header("Connection")
    base_url = st.sidebar.text_input(
        "Base URL",
        value=st.session_state.base_url,
        help=(
            "LM Studio default is http://localhost:1234/v1 (local). "
            "For a deployed app, use your publicly reachable LM Studio URL, e.g. "
            "http://YOUR_PUBLIC_IP:1234/v1 or https://your-domain/v1 via a reverse proxy."
        ),
    )
    api_key = st.sidebar.text_input(
        "API Key",
        value=st.session_state.api_key,
        type="password",
        help="LM Studio typically accepts any non-empty key (e.g., 'lm-studio').",
    )

    st.sidebar.header("Model & Generation")
    # Try to list available models from the server (best-effort)
    client: LLMClient = st.session_state.client
    models_available = client.list_models()
    model_help = "Type a model ID (or pick from the dropdown if listed)."

    if models_available:
        model = st.sidebar.selectbox(
            "Model",
            options=models_available,
            index=0 if st.session_state.model not in models_available
                    else models_available.index(st.session_state.model),
            help=model_help,
        )
    else:
        model = st.sidebar.text_input("Model", value=st.session_state.model, help=model_help)

    temperature = st.sidebar.slider("Temperature", 0.0, 2.0, float(st.session_state.temperature), 0.05)
    top_p       = st.sidebar.slider("Top-p", 0.0, 1.0, float(st.session_state.top_p), 0.01)
    max_tokens  = st.sidebar.slider("Max tokens", 64, 8192, int(st.session_state.max_tokens), 32)

    st.sidebar.header("System Prompt")
    system_prompt = st.sidebar.text_area("Instructions", value=st.session_state.system_prompt, height=150)

    col1, col2 = st.sidebar.columns(2)
    apply_btn  = col1.button("Apply")
    test_btn   = col2.button("Test connection")

    if apply_btn:
        # Apply changes and rebuild client if connection params changed
        dirty_connection = (normalize_base_url(base_url.strip()) != st.session_state.base_url.strip()) or \
                           (api_key.strip()  != st.session_state.api_key.strip())

        st.session_state.base_url = normalize_base_url(base_url.strip())
        st.session_state.api_key  = api_key.strip()
        st.session_state.model    = model.strip()
        st.session_state.temperature = float(temperature)
        st.session_state.top_p       = float(top_p)
        st.session_state.max_tokens  = int(max_tokens)
        st.session_state.system_prompt = system_prompt

        if dirty_connection:
            st.session_state.client = LLMClient(st.session_state.base_url, st.session_state.api_key)
        st.sidebar.success("Settings applied.", icon="✅")

    if test_btn:
        try:
            _ = st.session_state.client.list_models()
            st.sidebar.success("Server reachable.", icon="🟢")
        except Exception as e:
            st.sidebar.error(f"Connection failed: {e}", icon="🔴")


def build_messages() -> List[Dict[str, str]]:
    """
    Convert session history + system prompt to OpenAI messages.
    """
    msgs = [{"role": "system", "content": st.session_state.system_prompt.strip()}]
    for turn in st.session_state.chat_history:
        role = turn.get("role", "user")
        content = turn.get("content", "")
        if content:
            msgs.append({"role": role, "content": content})
    return msgs


def download_buttons():
    """
    Offer JSON and Markdown transcript downloads.
    """
    hist = st.session_state.chat_history
    json_bytes = json.dumps(hist, ensure_ascii=False, indent=2).encode("utf-8")

    # Build a simple markdown transcript
    md_lines = [f"# {APP_TITLE}", ""]
    md_lines.append("## System Prompt")
    md_lines.append("")
    md_lines.append("```\n" + st.session_state.system_prompt + "\n```")
    md_lines.append("")
    md_lines.append("## Conversation")
    md_lines.append("")
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


# -----------------------------------------------------------------------------
# R1-style stream parser (hide <think>…</think>, show only final answer)
# -----------------------------------------------------------------------------
def parse_r1_stream_and_render(
    chunks: Generator[str, None, None],
    final_box: "st.delta_generator.DeltaGenerator",
    think_expander_box: "st.delta_generator.DeltaGenerator",
) -> Tuple[str, str]:
    """
    Consumes the token stream, separates/hides <think>...</think>, and renders
    only the final answer live. Returns (final_text, think_text_hidden).

    We robustly handle tag boundaries across chunks by keeping a rolling buffer.
    """
    raw = ""               # full accumulated text
    pos = 0                # parser cursor into 'raw'
    in_think = False       # inside <think> ... </think>
    final_text = ""        # what we show to the user
    think_hidden = ""      # what we never show (for optional summary only)

    # show a placeholder in the reasoning expander
    think_expander_box.info("Reasoning content is hidden for safety. You can request a brief high-level explanation below.", icon="🧠")

    OPEN_TAGS = ("<think>", "<thinking>", "<Thoughts>", "<THINK>")  # generous variants
    CLOSE_TAGS = ("</think>", "</thinking>", "</Thoughts>", "</THINK>")

    def find_next_tag(s: str, start: int) -> Tuple[int, str, bool]:
        """Return (idx, tag, is_open) of the next tag occurrence or (-1, '', False)."""
        next_idx = len(s) + 1
        found_tag = ""
        is_open = False
        for t in OPEN_TAGS:
            i = s.find(t, start)
            if i != -1 and i < next_idx:
                next_idx = i
                found_tag = t
                is_open = True
        for t in CLOSE_TAGS:
            i = s.find(t, start)
            if i != -1 and i < next_idx:
                next_idx = i
                found_tag = t
                is_open = False
        if next_idx == len(s) + 1:
            return -1, "", False
        return next_idx, found_tag, is_open

    # stream loop
    for chunk in chunks:
        raw += chunk
        # parse newly appended region
        while True:
            idx, tag, is_open = find_next_tag(raw, pos)
            if idx == -1:
                # no more tags visible yet; output whatever is outside think
                if not in_think and pos < len(raw):
                    new_visible = raw[pos:]
                    final_text += new_visible
                    pos = len(raw)
                    final_box.markdown(final_text)
                break

            if not in_think:
                # output visible text up to the tag
                if idx > pos:
                    new_visible = raw[pos:idx]
                    final_text += new_visible
                    final_box.markdown(final_text)
                    pos = idx
                # process tag
                if is_open:
                    # enter <think>
                    pos = idx + len(tag)
                    in_think = True
                else:
                    # a stray close tag while not in think; skip it
                    pos = idx + len(tag)
            else:
                # currently masking think text; collect it until we find a close tag
                if is_open:
                    # nested open: treat as plain think text (rare), include tag text
                    think_hidden += raw[pos: idx + len(tag)]
                    pos = idx + len(tag)
                else:
                    # close tag: capture think segment up to tag, then exit think
                    if idx > pos:
                        think_hidden += raw[pos:idx]
                    pos = idx + len(tag)
                    in_think = False

    # If stream ended while still inside think, we just keep it hidden
    return final_text, think_hidden


# -----------------------------------------------------------------------------
# Main UI
# -----------------------------------------------------------------------------
def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="🧪", layout="centered")
    init_session_state()

    st.title(APP_TITLE)
    st.caption(
        "Connects to an OpenAI-compatible endpoint (LM Studio or similar). "
        "If your model emits `<think>…</think>` reasoning (e.g., R1), this app hides it and shows a clean final answer."
    )

    sidebar_controls()

    # Render history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input box
    user_input = st.chat_input("Ask about metallurgy/welding…")
    if user_input:
        # Append user message to history and render immediately
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Prepare assistant container
        with st.chat_message("assistant"):
            client: LLMClient = st.session_state.client
            messages = build_messages()

            # UI sub-containers
            think_expander = st.expander("🧠 Reasoning (hidden)")
            think_box = think_expander.empty()
            final_box = st.empty()

            # Stream model output and split out <think> ... </think>
            try:
                final_text, think_hidden = parse_r1_stream_and_render(
                    client.stream_chat(
                        model=st.session_state.model,
                        messages=messages,
                        temperature=st.session_state.temperature,
                        top_p=st.session_state.top_p,
                        max_tokens=st.session_state.max_tokens,
                    ),
                    final_box=final_box,
                    think_expander_box=think_box,
                )
            except Exception as e:
                final_text = f"**[Error]** {e}"
                think_hidden = ""
                final_box.markdown(final_text)

            # Optional: Request a high-level explanation (summary), not chain-of-thought
            with think_expander:
                st.caption(
                    "To keep answers safe and clear, hidden reasoning is not shown. "
                    "You can request a brief high-level explanation (no step-by-step)."
                )
                exp_key = f"explain_{len(st.session_state.chat_history)}_{int(time.time())}"
                if st.button("📝 High-level explanation", key=exp_key):
                    try:
                        explain_msgs = [
                            {"role": "system", "content":
                             "You are an assistant that provides concise, high-level justifications WITHOUT revealing chain-of-thought. "
                             "Do not enumerate steps. Summarize the key factors and assumptions behind the final answer in 2–4 bullets."
                             " Avoid LaTeX or equation formatting."},
                            {"role": "user", "content":
                             f"Question: {user_input}\n\nFinal answer:\n{final_text}\n\n"
                             "Provide a short justification as bullet points (no steps, no internal chain-of-thought)."}
                        ]
                        summary = client.complete_once(
                            model=st.session_state.model,
                            messages=explain_msgs,
                            temperature=0.2,
                            top_p=0.95,
                            max_tokens=200,
                        )
                        st.markdown(summary)
                    except Exception as e:
                        st.error(f"Could not generate explanation: {e}")

        # Save assistant reply (final answer only) into history
        st.session_state.chat_history.append({"role": "assistant", "content": final_text})

        # Token stats (best-effort)
        joined = []
        for m in st.session_state.chat_history:
            joined.append(f"{m['role'].upper()}: {m['content']}")
        approx_tokens = count_tokens("\n".join(joined))
        st.caption(f"Approx. tokens in this session: **{approx_tokens}**")

    st.divider()
    colA, colB, colC = st.columns([1,1,2])
    if colA.button("🧹 Clear chat"):
        st.session_state.chat_history = []
        st.rerun()
    download_buttons()

    st.info(
        f"Connected to **{st.session_state.base_url}** with model **{st.session_state.model}**. "
        "If this is a public deployment, ensure your LM Studio endpoint is reachable from the internet.",
        icon="🌐",
    )


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
