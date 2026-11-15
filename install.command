#!/bin/bash
cd "$(dirname "$0")"
echo "============================================================"
echo "  Установка генератора субтитров — hafizoff dev (macOS)"
echo "============================================================"

if ! command -v python3 &> /dev/null; then
    echo "[1/3] Python 3 не найден. Попытка установки через Homebrew..."
    if command -v brew &> /dev/null; then
        brew install python@3.11 python-tk@3.11
    else
        echo "Установите Python 3.11 с официального сайта https://www.python.org/downloads/macos/ и запустите install.command снова."
        read -p "Нажмите Enter для выхода..."
        exit 1
    fi
fi

echo "[2/3] Создание виртуального окружения (venv)..."
python3 -m venv venv

echo "[3/3] Установка зависимостей (PyTorch, Transformers, FFmpeg)..."
./venv/bin/python -m pip install --upgrade pip
./venv/bin/pip install -r requirements.txt

echo ""
echo "Установка завершена! Запускайте программу файлом run.command"
read -p "Нажмите Enter для закрытия окна..."
