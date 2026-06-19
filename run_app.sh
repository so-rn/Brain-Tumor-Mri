#!/usr/bin/env bash
# Launcher for the Brain Tumor MRI Classifier Streamlit app.
# Sets the protobuf env var BEFORE python starts (setting it inside app.py is
# too late — streamlit imports protobuf first).
set -e
cd "$(dirname "$0")"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export TF_CPP_MIN_LOG_LEVEL=2
PORT="${1:-8501}"
exec streamlit run app.py \
  --server.port "$PORT" \
  --browser.gatherUsageStats false
