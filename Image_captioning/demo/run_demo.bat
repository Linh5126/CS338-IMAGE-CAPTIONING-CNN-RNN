@echo off
cd /d %~dp0
streamlit run app.py --server.fileWatcherType none
