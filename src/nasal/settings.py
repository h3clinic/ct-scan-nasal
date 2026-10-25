"""Studio settings that touch the outside world: the language-model API key and the Docker check.

The key, its provider (OpenAI, Anthropic, Google Gemini, OpenRouter or a custom OpenAI-compatible address, see
`nasal.providers`) and the address are kept in the app's memory for the browser session. "Remember on this
computer" stores them in the operating system's credential store through `keyring` (Windows Credential Manager,
macOS Keychain), never in a plain file or in the repository. The key is never logged, shown in full or sent
anywhere but that provider.
"""

import json

from .providers import PROVIDERS, looks_like_key  # noqa: F401  (looks_like_key is part of this module's API)

KEYRING_SERVICE = "nasal0"
KEYRING_USER = "api_key"
LEGACY_USER = "openai_api_key"  # earlier versions stored a bare OpenAI key here


def mask(key: str | None) -> str:
    """'sk-a…abcd' for display: the provider prefix and the last four characters, never the full key."""
    return f"{key[:4]}…{key[-4:]}" if key else "not set"


def _keyring():
    try:
        import keyring

        return keyring
    except ImportError:
        return None


def can_remember() -> bool:
    return _keyring() is not None


def load_saved() -> dict | None:
    """{"provider", "key", "base_url"} saved on this computer, or None."""
    kr = _keyring()
    if kr is None:
        return None
    try:
        raw = kr.get_password(KEYRING_SERVICE, KEYRING_USER)
        if raw:
            saved = json.loads(raw)
            return saved if saved.get("key") else None
        legacy = kr.get_password(KEYRING_SERVICE, LEGACY_USER)
        return {"provider": "OpenAI", "key": legacy, "base_url": None} if legacy else None
    except Exception:  # no usable credential store, or an unreadable entry: behave as if nothing is saved
        return None


def save(key: str, provider: str, base_url: str | None = None) -> bool:
    kr = _keyring()
    if kr is None:
        return False
    try:
        kr.set_password(KEYRING_SERVICE, KEYRING_USER, json.dumps({"provider": provider, "key": key, "base_url": base_url}))
        return True
    except Exception:
        return False


def forget_saved() -> None:
    kr = _keyring()
    if kr is None:
        return
    for user in (KEYRING_USER, LEGACY_USER):
        try:
            kr.delete_password(KEYRING_SERVICE, user)
        except Exception:  # nothing saved under that name
            pass


def test_key(key: str, provider: str = "OpenAI", base_url: str | None = None) -> tuple[bool, str]:
    """Ask the provider to list its models with this key; (works, message). Sends nothing but the key."""
    from openai import OpenAI

    url = base_url or PROVIDERS.get(provider, PROVIDERS["OpenAI"]).base_url
    if not url:
        return False, "Key test failed: a custom provider needs its API address."
    try:
        OpenAI(api_key=key, base_url=url, timeout=15).models.list()
        return True, "Key works."
    except Exception as err:  # authentication, network or quota: show the kind, never the key
        return False, f"Key test failed: {type(err).__name__}."
