@echo off
setlocal
cd /d "%~dp0"

set "FFMPEG_DIR=%LOCALAPPDATA%\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.2-full_build\bin"
if exist "%FFMPEG_DIR%" set "PATH=%PATH%;%FFMPEG_DIR%"
set PYTHONUTF8=1

REM Clear out any server still holding port 3001 first. Otherwise the wait
REM loop below can see THAT leftover process answering and open the browser
REM immediately, even if the server we're about to start fails outright
REM (e.g. it can't bind the port at all and exits right away).
for /f "tokens=5" %%p in ('netstat -ano ^| findstr "3001" ^| findstr "LISTENING"') do (
    echo Stopping a leftover server on port 3001 ^(PID %%p^)...
    taskkill /F /PID %%p >nul 2>&1
)

echo Starting Sapph-TTS Chat Demo...
start "Sapph-TTS Server" "..\.venv\Scripts\python.exe" -m uvicorn chat_demo:app --host 127.0.0.1 --port 3001

echo Waiting for the server to finish loading models (can take up to a minute)...
set _WAIT_TICKS=0
:waitloop
curl -s -o nul http://127.0.0.1:3001/emotions
if not errorlevel 1 goto ready
set /a _WAIT_TICKS+=1
if %_WAIT_TICKS% geq 60 (
    echo.
    echo [ERROR] The server still isn't responding after 2 minutes.
    echo Check the "Sapph-TTS Server" window for the real error.
    pause
    exit /b 1
)
timeout /t 2 /nobreak >nul
goto waitloop

:ready
start "" http://127.0.0.1:3001
echo Ready, opened http://127.0.0.1:3001 in your browser.
echo (The server keeps running in the other window titled "Sapph-TTS Server", close it to stop.)
