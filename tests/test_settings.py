"""Studio settings tests: key handling without a real credential store, OpenAI or Docker."""
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
    assert not settings.looks_like_key("hello") and not settings.looks_like_key("sk-short")
    assert not settings.looks_like_key("sk-" + "a" * 10 + " " + "b" * 10)
    assert settings.mask(key) == "sk-…WXYZ" and "a" * 8 not in settings.mask(key)
    assert settings.mask(None) == "not set"


def test_remembered_key_round_trips_through_the_credential_store(monkeypatch):
    fake = FakeKeyring()
    monkeypatch.setattr(settings, "_keyring", lambda: fake)
    assert settings.load_saved_key() is None
    assert settings.save_key("sk-" + "k" * 30)
    assert settings.load_saved_key() == "sk-" + "k" * 30
    assert fake.store == {(settings.KEYRING_SERVICE, settings.KEYRING_USER): "sk-" + "k" * 30}
    settings.forget_saved_key()
    settings.forget_saved_key()  # forgetting twice is fine
    assert settings.load_saved_key() is None


def test_without_a_credential_store_nothing_is_saved(monkeypatch):
    monkeypatch.setattr(settings, "_keyring", lambda: None)
    assert not settings.can_remember() and not settings.save_key("sk-" + "k" * 30)
    assert settings.load_saved_key() is None


def test_docker_status_without_docker(monkeypatch):
    monkeypatch.setattr(settings.shutil, "which", lambda name: None)
    assert settings.docker_status("img") == "Docker is not installed."


def test_openai_provider_needs_a_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="Settings"):
        agents.run_pipeline("narrow valve", provider="openai")
