#!/bin/bash
cd "$(dirname "$0")"
echo "============================================================"
echo "  Установка генератора субтитров — hafizoff dev (macOS)"
echo "============================================================"

unset CONDA_PREFIX CONDA_DEFAULT_ENV
ARCH=$(uname -m)
echo "Архитектура Mac: $ARCH"

PY_BIN=""
for p in /opt/homebrew/bin/python3.11 /usr/local/bin/python3.11 /Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11 $(command -v python3.11) /opt/homebrew/bin/python3 /usr/local/bin/python3 $(command -v python3); do
    if [ -x "$p" ] && [[ "$p" != *"anaconda"* ]] && [[ "$p" != *"miniconda"* ]]; then
        PY_BIN="$p"
        break
    fi
done

if [ -z "$PY_BIN" ]; then
    if command -v brew &> /dev/null; then
        brew install python@3.11 python-tk@3.11
        PY_BIN="$(brew --prefix python@3.11)/bin/python3.11"
    else
        echo "ОШИБКА: Python 3 не найден! Скачайте Python 3.11 с https://www.python.org/downloads/macos/"
        read -p "Нажмите Enter для выхода..."
        exit 1
    fi
fi

echo "Используется Python: $PY_BIN"
rm -rf venv
"$PY_BIN" -m venv venv
./venv/bin/python -m pip install --upgrade pip

if [ "$ARCH" = "x86_64" ]; then
    echo "Обнаружен Intel Mac: установка совместимого PyTorch 2.2.2..."
    ./venv/bin/python -m pip install "numpy<2" "torch==2.2.2" "torchaudio==2.2.2" transformers soundfile imageio-ffmpeg
else
    echo "Обнаружен Apple Silicon ($ARCH): установка PyTorch с поддержкой MPS..."
    ./venv/bin/python -m pip install -r requirements.txt
fi

if ! ./venv/bin/python -c "import torch, transformers, soundfile, imageio_ffmpeg" &> /dev/null; then
    echo "ОШИБКА: Не удалось установить PyTorch! Проверьте интернет или установите Python 3.11."
    read -p "Нажмите Enter для выхода..."
    exit 1
fi

echo ""
echo "Установка успешно завершена! Запускайте run.command"
read -p "Нажмите Enter для выхода..."