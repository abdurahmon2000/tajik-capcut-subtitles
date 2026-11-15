@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================================
echo   Установка генератора субтитров — hafizoff dev
echo ============================================================
echo.

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [1/4] Python не найден. Запускаю автоматическую установку Python 3.11...
    winget install -e --id Python.Python.3.11 --silent --accept-package-agreements --accept-source-agreements >nul 2>&1
    
    if %errorlevel% neq 0 (
        echo Скачивание установщика Python напрямую с python.org...
        powershell -Command "Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe' -OutFile 'python_installer.exe'"
        echo Установка Python (пожалуйста, подождите 1-2 минуты)...
        start /wait python_installer.exe /quiet InstallAllUsers=0 PrependPath=1 Include_test=0
        del python_installer.exe >nul 2>&1
    )
    
    :: Обновляем PATH в текущем окне, чтобы сразу увидеть установленный Python
    set "PATH=%LOCALAPPDATA%\Programs\Python\Python311;%LOCALAPPDATA%\Programs\Python\Python311\Scripts;%PATH%"
)

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ОШИБКА] Не удалось автоматически установить Python.
    echo Установите Python 3.11+ вручную с сайта https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [2/4] Создание виртуального окружения (venv)...
if not exist venv (
    python -m venv venv
)

echo [3/4] Обновление pip и установка библиотек (PyTorch, Transformers, FFmpeg)...
venv\Scripts\python.exe -m pip install --upgrade pip
venv\Scripts\pip.exe install -r requirements.txt

echo.
echo [4/4] Установка успешно завершена!
echo Теперь вы можете запускать программу двойным кликом по файлу run.bat
echo ============================================================
pause