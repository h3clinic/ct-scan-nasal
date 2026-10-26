"""Run a command while keeping Windows from idle-sleeping, for long training or OpenFOAM runs:

    python scripts/keep_awake.py -- python scripts/distill_train.py --data data/distill/student.jsonl

Asks Windows (SetThreadExecutionState) to stay awake until the command exits, then lets go. Power settings are not
changed; closing the lid or choosing Sleep still sleeps. The display may still turn off. On other systems the
command just runs.
"""

import ctypes
import subprocess
import sys

ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001


def main() -> int:
    cmd = sys.argv[1:]
    if cmd[:1] == ["--"]:
        cmd = cmd[1:]
    if not cmd:
        print(__doc__)
        return 2
    windows = sys.platform == "win32"
    if windows:
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    try:
        return subprocess.call(cmd)
    finally:
        if windows:
            ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)


if __name__ == "__main__":
    sys.exit(main())
