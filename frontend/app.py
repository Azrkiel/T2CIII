"""
Text-to-CAD Streamlit Frontend.

Form-based UI with prompt input, CadQuery code viewer, and 3D model viewer.
"""

import base64
import json
import os

import requests
import streamlit as st
import streamlit.components.v1 as components
from streamlit_ace import st_ace

st.set_page_config(page_title="Text-to-CAD", page_icon="\U0001f529", layout="wide")

# Reduce top padding for a tighter layout
st.markdown(
    "<style>.block-container{padding-top:1.5rem;}</style>",
    unsafe_allow_html=True,
)

BACKEND_URL = os.environ.get("BACKEND_URL", "http://backend:8000")

STEP_LABELS = {
    "planner": "Planning blueprint",
    "assembler": "Compiling assembly",
    "export": "Exporting .glb",
    "script": "CadQuery code captured",
    "complete": "Model ready",
}

# ── Session State ───────────────────────────────────────────
for key, default in [("script_code", ""), ("glb_bytes", None), ("status", "Ready"), ("script_version", 0)]:
    if key not in st.session_state:
        st.session_state[key] = default


# ── Helpers ─────────────────────────────────────────────────

def _step_label(step: str) -> str:
    if step in STEP_LABELS:
        return STEP_LABELS[step]
    if step.startswith("machinist:"):
        return f"Building '{step.split(':', 1)[1]}'"
    return step


def _icon(status: str) -> str:
    return {"running": "\u23f3", "done": "\u2705", "error": "\u274c"}.get(status, "\u2b1c")


def render_glb_viewer(glb_bytes: bytes, height: int = 500) -> None:
    assert isinstance(height, int) and 100 <= height <= 2000
    b64 = base64.b64encode(glb_bytes).decode("ascii")
    data_uri = f"data:model/gltf-binary;base64,{b64}"
    html = f"""
    <script type="module"
        src="https://ajax.googleapis.com/ajax/libs/model-viewer/3.1.1/model-viewer.min.js">
    </script>
    <model-viewer
        src="{data_uri}"
        auto-rotate camera-controls shadow-intensity="1"
        style="width:100%;height:{height}px;background:#1a1a2e;">
    </model-viewer>
    """
    components.html(html, height=height + 20)


def _progress_md(steps: list[dict]) -> str:
    lines = []
    for s in steps:
        if s["step"] == "complete":
            continue
        icon = _icon(s["status"])
        label = _step_label(s["step"])
        detail = s.get("detail", "")
        line = f"{icon} **{label}**"
        if detail and s["status"] != "done":
            line += f"  \n&nbsp;&nbsp;&nbsp;&nbsp;_{detail}_"
        lines.append(line)
    return "\n\n".join(lines)


def stream_pipeline(prompt: str, progress_ph) -> None:
    """Stream SSE events from /generate, updating session state."""
    steps, idx = [], {}
    script_code, glb_bytes = "", None
    error_message = ""

    try:
        with requests.post(
            f"{BACKEND_URL}/generate",
            json={"prompt": prompt},
            stream=True,
            timeout=300,
        ) as resp:
            if resp.status_code != 200:
                try:
                    detail = resp.json().get("detail", resp.text)
                except Exception:
                    detail = resp.text
                st.session_state.status = f"Error (HTTP {resp.status_code})"
                st.error(f"Generation failed:\n\n```\n{detail}\n```")
                return

            st.session_state.status = "Generating..."

            for raw in resp.iter_lines(decode_unicode=True):
                if not raw or not raw.startswith("data: "):
                    continue
                ev = json.loads(raw[6:])
                step, status = ev["step"], ev["status"]

                if step == "complete":
                    if status == "done":
                        glb_bytes = base64.b64decode(ev.get("glb", ""))
                    elif status == "error":
                        error_message = ev.get("message", "")
                    script_code = ev.get("script", script_code)
                    continue
                if step == "script" and status == "done":
                    script_code = ev.get("detail", "")

                if step in idx:
                    steps[idx[step]] = ev
                else:
                    idx[step] = len(steps)
                    steps.append(ev)
                progress_ph.markdown(_progress_md(steps))

    except requests.ConnectionError:
        st.session_state.status = "Connection error"
        st.error("Could not connect to backend. Is it running?")
        return
    except requests.Timeout:
        st.session_state.status = "Timeout"
        st.error("Request timed out.")
        return

    st.session_state.script_code = script_code
    st.session_state.glb_bytes = glb_bytes
    if script_code:
        st.session_state.script_version += 1

    has_error = any(s["status"] == "error" for s in steps)
    if has_error:
        err = next(s for s in steps if s["status"] == "error")
        st.session_state.status = f"Error: {_step_label(err['step'])}"
        if error_message:
            st.error(
                f"CadQuery compilation failed:\n\n```\n{error_message}\n```"
            )
        progress_ph.markdown(_progress_md(steps))
    else:
        st.session_state.status = "Ready"
        progress_ph.empty()


def run_script(script: str) -> None:
    """Execute CadQuery code via the /run endpoint."""
    st.session_state.status = "Running..."
    try:
        data = requests.post(
            f"{BACKEND_URL}/run", json={"script": script}, timeout=120
        ).json()
        if data.get("status") == "error":
            st.session_state.status = "Run failed"
            st.error(f"Script execution failed:\n\n```\n{data.get('detail', 'Unknown error')}\n```")
            return
        st.session_state.glb_bytes = base64.b64decode(data["glb"])
        st.session_state.status = "Ready"
    except requests.ConnectionError:
        st.session_state.status = "Connection error"
        st.error("Could not connect to backend.")
    except requests.Timeout:
        st.session_state.status = "Timeout"
        st.error("Script execution timed out.")


# ── Page Layout ─────────────────────────────────────────────

# 1) Prompt
st.markdown("**Describe your 3D model**")
prompt = st.text_area(
    "prompt",
    placeholder="A smooth, ergonomic shell for a computer mouse, roughly 120mm long and 60mm wide.",
    label_visibility="collapsed",
    height=80,
)

# 2) Generate button
generate = st.button(
    "Generate (Ctrl+Enter)", type="primary", use_container_width=True
)

# 3) Progress area (populated during streaming)
progress_ph = st.empty()

if generate and prompt.strip():
    stream_pipeline(prompt.strip(), progress_ph)

# 4) 3D viewer (when model exists)
if st.session_state.glb_bytes:
    render_glb_viewer(st.session_state.glb_bytes)
    st.download_button(
        "Download .glb",
        st.session_state.glb_bytes,
        "assembly.glb",
        "model/gltf-binary",
    )

# 5) CadQuery Code header row
col_label, col_run = st.columns([5, 1])
with col_label:
    st.markdown("**CadQuery Code**")
with col_run:
    run_clicked = st.button("Run (Ctrl+Enter)")

# 6) Editable code editor (always visible)
edited_code = st_ace(
    value=st.session_state.script_code or "",
    language="python",
    theme="monokai",
    height=400,
    key=f"ace_editor_{st.session_state.script_version}",
    auto_update=True,
)
st.session_state.script_code = edited_code

if run_clicked and st.session_state.script_code:
    run_script(st.session_state.script_code)
    st.rerun()

# 7) Status bar
st.caption(st.session_state.status)
