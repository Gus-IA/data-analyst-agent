"""Streamlit UI for the data analyst agent."""

import os
import sys
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from agent import ask, build_agent

UPLOAD_DIR = Path(__file__).resolve().parent.parent / "workspace" / "uploads"
MAX_UPLOAD_MB = 20


def extract_steps(messages):
    """List each tool call with its output, matched by tool_call_id."""
    steps = {}
    for m in messages:
        for call in getattr(m, "tool_calls", None) or []:
            steps[call["id"]] = {
                "tool": call["name"],
                "args": call["args"],
                "result": "",
            }
        if m.type == "tool" and m.tool_call_id in steps:
            steps[m.tool_call_id]["result"] = m.content
    return list(steps.values())


def render_turn(answer, steps, images):
    st.markdown(answer)
    for img in images:
        st.image(img)
    if steps:
        with st.expander(f"Steps ({len(steps)} tool calls)"):
            for s in steps:
                st.markdown(f"**{s['tool']}**")
                code = s["args"].get("code")
                if code:
                    st.code(code, language="python")
                st.text(str(s["result"])[:1500])


st.set_page_config(page_title="Data Analyst Agent", page_icon="📊", layout="wide")
st.title("📊 Data Analyst Agent")
st.caption(
    "Upload a CSV and ask questions. The agent writes Python and runs it in an isolated sandbox."
)

with st.sidebar:
    st.header("Dataset")
    uploaded = st.file_uploader("CSV file", type="csv")
    max_steps = st.slider("Max agent steps", 4, 20, 12)

if uploaded is None:
    st.info("Upload a CSV file to start.")
    st.stop()

if uploaded.size > MAX_UPLOAD_MB * 1024 * 1024:
    st.error(f"The file is larger than {MAX_UPLOAD_MB} MB.")
    st.stop()

# Rebuild the agent only when a new file is uploaded
if st.session_state.get("file_key") != (uploaded.name, uploaded.size):
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    path = UPLOAD_DIR / f"{uuid.uuid4().hex[:8]}.csv"
    path.write_bytes(uploaded.getvalue())
    agent, runs = build_agent(str(path))
    st.session_state.update(
        file_key=(uploaded.name, uploaded.size),
        path=path,
        agent=agent,
        runs=runs,
        history=[],
    )

try:
    with st.expander("Dataset preview"):
        st.dataframe(pd.read_csv(st.session_state.path).head(10))
except Exception as exc:
    st.error(f"Could not read the CSV: {exc}")
    st.stop()

for item in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(item["question"])
    with st.chat_message("assistant"):
        render_turn(item["answer"], item["steps"], item["images"])

if question := st.chat_input("Ask about your data..."):
    with st.chat_message("user"):
        st.markdown(question)

    runs = st.session_state.runs
    before = len(runs)  # sandbox runs created by this question start after this index
    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            try:
                answer, messages = ask(
                    st.session_state.agent, question, max_steps=max_steps
                )
            except Exception as exc:
                answer, messages = f"Something went wrong: {exc}", []
        steps = extract_steps(messages)
        images = [
            str(Path(r["out_dir"]) / f)
            for r in runs[before:]
            for f in r["files"]
            if f.lower().endswith(".png")
        ]
        render_turn(answer, steps, images)

    st.session_state.history.append(
        {"question": question, "answer": answer, "steps": steps, "images": images}
    )
