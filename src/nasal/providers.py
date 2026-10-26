"""Language-model providers the agents can use: any service with an OpenAI-compatible chat API and tool calling.

The user pastes a key; its prefix picks the provider (they can override it), and the default model and address
come from here. Only OpenAI accepts `store=False`, so it is sent only there. What each provider keeps of a
request is set by that provider's own data policy.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Provider:
    name: str
    base_url: str | None  # None: the user gives the address (custom endpoint)
    default_model: str
    key_prefix: str | None  # how its keys start, for detection
    supports_store: bool = False


PROVIDERS = {
    p.name: p for p in (
        Provider("OpenAI", "https://api.openai.com/v1", "gpt-4.1-mini", "sk-", supports_store=True),
        Provider("Anthropic (Claude)", "https://api.anthropic.com/v1/", "claude-sonnet-5-5", "sk-ant-"),
        Provider("Google Gemini", "https://generativelanguage.googleapis.com/v1beta/openai/", "gemini-2.5-flash", "AIza"),
        Provider("OpenRouter", "https://openrouter.ai/api/v1", "openai/gpt-4.1-mini", "sk-or-"),
        Provider("Custom (OpenAI-compatible)", None, "", None),
    )
}
DEFAULT = "OpenAI"


def detect(key: str) -> str:
    """Provider name for a pasted key; the longest matching prefix wins (sk-ant- before sk-)."""
    key = (key or "").strip()
    matches = [p for p in PROVIDERS.values() if p.key_prefix and key.startswith(p.key_prefix)]
    return max(matches, key=lambda p: len(p.key_prefix)).name if matches else "Custom (OpenAI-compatible)"


def looks_like_key(key: str) -> bool:
    """Any plausible API key: long enough, no spaces."""
    key = (key or "").strip()
    return len(key) >= 20 and not any(c.isspace() for c in key)


def request_options(provider: str) -> dict:
    """Extra chat-completion arguments for this provider."""
    return {"store": False} if PROVIDERS.get(provider, PROVIDERS[DEFAULT]).supports_store else {}
