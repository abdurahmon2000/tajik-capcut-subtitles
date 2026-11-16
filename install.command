#!/bin/bash
cd "$(dirname "$0")"
echo "============================================================"
echo "  Установка генератора субтитров — hafizoff dev (macOS)"
echo "============================================================"

PY_BIN=""
for p in /opt/homebrew/bin/python3.11 /usr/local/bin/python3.11 /Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11 $(command -v python3.11); do
    if [ -x "$p" ]; then
        PY_BIN="$p"
        break
    fi
done

if [ -z "$PY_BIN" ]; then
    echo "[1/3] Поиск/установка совместимого Python 3.11..."
    if command -v brew &> /dev/null; then
        brew install python@3.11 python-tk@3.11
        PY_BIN="$(brew --prefix python@3.11)/bin/python3.11"
    elif command -v python3 &> /dev/null; then
        PY_BIN="$(command -v python3)"
    else
        echo "ОШИБКА: Python 3 не найден! Скачайте Python 3.11 с https://www.python.org/downloads/macos/"
        read -p "Нажмите Enter для выхода..."
        exit 1
    fi
fi

echo "Используется Python: $PY_BIN"
rm -rf venv
echo "[2/3] Создание виртуального окружения (venv)..."
"$PY_BIN" -m venv venv

echo "[3/3] Установка библиотек (PyTorch, Transformers, FFmpeg)..."
./venv/bin/python -m pip install --upgrade pip
./venv/bin/python -m pip install -r requirements.txt

if ! ./venv/bin/python -c "import torch, transformers, soundfile, imageio_ffmpeg" &> /dev/null; then
    echo ""
    echo "ОШИБКА: Не удалось установить PyTorch (возможно, установлена неподходящая версия Python)."
    echo "Установите Python 3.11: https://www.python.org/downloads/macos/ и запустите install.command снова."
    read -p "Нажмите Enter для выхода..."
    exit 1
fi

echo ""
echo "Установка успешно завершена! Запускайте программу файлом run.command"
read -p "Нажмите Enter для закрытия окна..."
