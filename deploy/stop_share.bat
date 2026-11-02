@echo off
rem Stop sharing ISL Bridge: the public link goes offline. Start again with share.bat (you get a new link).
cd /d "%~dp0.."
".venv-ml\Scripts\python.exe" deploy\share.py --stop
"%SystemRoot%\System32\timeout.exe" /t 5
