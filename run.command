#!/bin/bash
cd "$(dirname "$0")"
unset CONDA_PREFIX CONDA_DEFAULT_ENV
if [ ! -f "venv/bin/python" ]; then
    echo "Окружение не найдено. Сначала запустите install.command"
    read -p "Нажмите Enter для выхода..."
    exit 1
fi
./venv/bin/python capcut_inject.py