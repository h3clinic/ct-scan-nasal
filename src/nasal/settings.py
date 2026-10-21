"""Studio settings that touch the outside world: the OpenAI API key and the Docker check.

The key is kept in the app's memory for the browser session. "Remember on this computer" stores it in the
operating system's credential store through `keyring` (Windows Credential Manager, macOS Keychain), never in
a plain file or in the repository. The key is never logged, shown in full or sent anywhere but OpenAI.
"""

import shutil
import subprocess

KEYRING_SERVICE = "nasal0"
KEYRING_USER = "openai_api_key"


def looks_like_key(key: str) -> bool:
    """A pasted OpenAI key: starts with sk-, no spaces, long enough to be real."""
    key = (key or "").strip()
    return key.startswith("sk-") and len(key) >= 20 and not any(c.isspace() for c in key)


def mask(key: str | None) -> str:
    """'sk-…abcd' for display; never the full key."""
    return f"sk-…{key[-4:]}" if key else "not set"


def _keyring():
    try:
        import keyring

        return keyring
    except ImportError:
        return None


def can_remember() -> bool:
    return _keyring() is not None


def load_saved_key() -> str | None:
    kr = _keyring()
    try:
        return kr.get_password(KEYRING_SERVICE, KEYRING_USER) if kr else None
    except Exception:  # no usable credential store: behave as if nothing is saved
        return None


def save_key(key: str) -> bool:
    kr = _keyring()
    if kr is None:
        return False
    try:
        kr.set_password(KEYRING_SERVICE, KEYRING_USER, key)
        return True
    except Exception:
        return False


def forget_saved_key() -> None:
    kr = _keyring()
    if kr is None:
        return
    try:
        kr.delete_password(KEYRING_SERVICE, KEYRING_USER)
    except Exception:  # nothing saved
        pass


def test_key(key: str) -> tuple[bool, str]:
    """Ask OpenAI to list models with this key; (works, message). Sends nothing but the key."""
    from openai import OpenAI

    try:
        OpenAI(api_key=key, base_url="https://api.openai.com/v1", timeout=15).models.list()
        return True, "Key works."
    except Exception as err:  # authentication, network or quota: show the kind, never the key
        return False, f"Key test failed: {type(err).__name__}."


def docker_status(image: str) -> str:
    """One line: whether Docker is running and has the OpenFOAM image."""
    docker = shutil.which("docker")
    if not docker:
        return "Docker is not installed."
    try:
        info = subprocess.run([docker, "info", "--format", "{{.ServerVersion}}"], capture_output=True, text=True, timeout=20)
        if info.returncode != 0:
            return "Docker is installed but not running: start Docker Desktop."
        found = subprocess.run([docker, "image", "inspect", image], capture_output=True, text=True, timeout=20)
    except subprocess.TimeoutExpired:
        return "Docker did not answer in time."
    have = "image ready" if found.returncode == 0 else f"image {image} will be downloaded on the first run"
    return f"Docker {info.stdout.strip()} running, {have}."
