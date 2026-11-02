@echo off
rem Share ISL Bridge with testers: a public https link served from this laptop.
rem Keep this window open (and the laptop awake + online) while people test. Close it to stop sharing.
cd /d "%~dp0.."
title ISL Bridge - sharing with testers
".venv-ml\Scripts\python.exe" -u deploy\share.py %*
pause
