#!/usr/bin/env bash
# One-step start for the Nasal0 studio on macOS and Linux: ./start-studio.sh
# First run: makes a Python environment in ~/.venvs/ct-scan-nasal and installs the studio.
# Every run: prints the setup check, then opens the studio in the browser.
# Optional: NASALSEG_ROOT=/path/to/nasalseg ./start-studio.sh   lists its patients (images/ and labels/ inside).
#           ./start-studio.sh --update                          reinstalls after pulling new code.
#           ./start-studio.sh --no-browser                      starts the studio without opening a browser.
set -euo pipefail
cd "$(dirname "$0")"
VENV="${HOME}/.venvs/ct-scan-nasal"
PY="${VENV}/bin/python"
update=0
open_flag=--open
for arg in "$@"; do
    case "$arg" in
        --update) update=1 ;;
        --no-browser) open_flag= ;;
    esac
done

if [ ! -x "$PY" ]; then
    echo "Creating the Python environment in ${VENV} ..."
    PYTHON="$(command -v python3.12 || command -v python3 || true)"
    if [ -z "$PYTHON" ]; then
        echo "Python 3.12 is needed: https://www.python.org/downloads/"
        exit 1
    fi
    "$PYTHON" -m venv "$VENV"
    update=1
fi
"$PY" -c "import nasal, gradio, pydicom, nibabel" 2>/dev/null || update=1
if [ "$update" = 1 ]; then
    echo "Installing the studio (a few minutes the first time) ..."
    "$PY" -m pip install --upgrade pip -q
    "$PY" -m pip install -q -e ".[studio,segment,agents,patients]"
fi

"$PY" -m nasal.setup_check

args=()
if [ -n "${NASALSEG_ROOT:-}" ]; then
    args=(--images "${NASALSEG_ROOT}/images" --labels "${NASALSEG_ROOT}/labels")
fi
echo "Opening the studio at http://127.0.0.1:7861 (Ctrl+C to stop) ..."
exec "$PY" -m nasal.studio ${open_flag} ${args[@]+"${args[@]}"}
