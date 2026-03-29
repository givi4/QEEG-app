@echo off
cd /d "%~dp0"
set MPLBACKEND=Agg
set PYTHONPATH=%~dp0..
streamlit run eeg_cleaner.py
