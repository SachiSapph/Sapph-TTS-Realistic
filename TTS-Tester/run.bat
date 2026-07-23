@echo off
setlocal
cd /d "%~dp0"

set "FFMPEG_DIR=%LOCALAPPDATA%\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.2-full_build\bin"
if exist "%FFMPEG_DIR%" set "PATH=%PATH%;%FFMPEG_DIR%"
set PYTHONUTF8=1

echo Starting Sapph-TTS Chat Demo...
start "Sapph-TTS Server" "..\.venv\Scripts\python.exe" -m uvicorn chat_demo:app --host 127.0.0.1 --port 3001

echo Waiting for the server to finish loading models (can take up to a minute)...
:waitloop
curl -s -o nul http://127.0.0.1:3001/emotions
if errorlevel 1 (
    timeout /t 2 /nobreak >nul
    goto waitloop
)

start "" http://127.0.0.1:3001
echo Ready — opened http://127.0.0.1:3001 in your browser.
echo (The server keeps running in the other window titled "Sapph-TTS Server" — close it to stop.)
