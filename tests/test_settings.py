"""Studio settings tests: key handling without a real credential store, a provider or Docker."""
import pytest

from nasal import agents, settings


class FakeKeyring:
    def __init__(self):
        self.store = {}

    def get_password(self, service, user):
        return self.store.get((service, user))

    def set_password(self, service, user, value):
        self.store[(service, user)] = value

    def delete_password(self, service, user):
        del self.store[(service, user)]


def test_key_format_and_mask_never_show_the_key():
    key = "sk-proj-" + "a" * 40 + "WXYZ"
    assert settings.looks_like_key(key) and settings.looks_like_key(f"  {key}\n")
    assert settings.looks_like_key("AIza" + "b" * 35)  # a Google key is fine too
    assert not settings.looks_like_key("hello") and not settings.looks_like_key("sk-short")
    assert not settings.looks_like_key("sk-" + "a" * 10 + " " + "b" * 10)
    assert settings.mask(key) == "sk-p…WXYZ" and "a" * 8 not in settings.mask(key)
    assert settings.mask(None) == "not set"


def test_provider_is_detected_from_the_key():
    from nasal import providers

    assert providers.detect("sk-ant-api03-" + "x" * 30) == "Anthropic (Claude)"
    assert providers.detect("sk-or-v1-" + "x" * 30) == "OpenRouter"
    assert providers.detect("sk-proj-" + "x" * 30) == "OpenAI"
    assert providers.detect("AIza" + "x" * 35) == "Google Gemini"
    assert providers.detect("gsk_" + "x" * 30) == "Custom (OpenAI-compatible)"
    assert providers.request_options("OpenAI") == {"store": False} and providers.request_options("OpenRouter") == {}


def test_remembered_key_round_trips_through_the_credential_store(monkeypatch):
    fake = FakeKeyring()
    monkeypatch.setattr(settings, "_keyring", lambda: fake)
    assert settings.load_saved() is None
    assert settings.save("sk-ant-" + "k" * 30, "Anthropic (Claude)")
    assert settings.load_saved() == {"provider": "Anthropic (Claude)", "key": "sk-ant-" + "k" * 30, "base_url": None}
    settings.forget_saved()
    settings.forget_saved()  # forgetting twice is fine
    assert settings.load_saved() is None
    fake.store[(settings.KEYRING_SERVICE, settings.LEGACY_USER)] = "sk-" + "o" * 30  # saved by an earlier version
    assert settings.load_saved() == {"provider": "OpenAI", "key": "sk-" + "o" * 30, "base_url": None}


def test_without_a_credential_store_nothing_is_saved(monkeypatch):
    monkeypatch.setattr(settings, "_keyring", lambda: None)
    assert not settings.can_remember() and not settings.save("sk-" + "k" * 30, "OpenAI")
    assert settings.load_saved() is None


def test_agents_need_an_openai_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="Settings"):
        agents.run_pipeline("narrow valve")


def test_openai_errors_are_explained_in_plain_words():
    quota = "RateLimitError: Error code: 429 - {'error': {'type': 'insufficient_quota', 'code': 'credit_balance_exhausted'}}"
    assert "no credits left" in agents.explain_error(quota) and agents.explain_error(quota).endswith("(RateLimitError)")
    assert "did not accept the API key" in agents.explain_error("AuthenticationError: Error code: 401 invalid_api_key")
    assert "RuntimeError: boom" in agents.explain_error("RuntimeError: boom")
