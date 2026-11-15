#!/bin/bash
cd "$(dirname "$0")"
if [ ! -d "venv" ]; then
    echo "Сначала запустите install.command для установки библиотек!"
    read -p "Нажмите Enter..."
    exit 1
fi
./venv/bin/python capcut_inject.py
