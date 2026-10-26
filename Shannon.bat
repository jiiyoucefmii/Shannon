@echo off
rem Starts Shannon (the NLP tutor) on this PC and opens it in your browser. Close this window to stop it.
cd /d "%~dp0"
title Shannon - NLP tutor (close this window to stop)
".venv\Scripts\python.exe" scripts\shannon_server.py --open
pause
