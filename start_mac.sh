#!/bin/bash

echo "Starting Medical Entity Classification Service..."
echo

cd "$(dirname "$0")" || exit 1

uvicorn main:app --reload --host 0.0.0.0 --port 8000

read -p "Press Enter to close..."
