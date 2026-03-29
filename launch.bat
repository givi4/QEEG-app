@echo off
cd /d "%~dp0"
set MPLBACKEND=Agg
streamlit run qeeg_launcher.py
