#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
streamlit run app.py --server.fileWatcherType none
