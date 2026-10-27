from __future__ import annotations

import os

import pytest

from yblocalizer.translation_setup import key_page_for, list_translation_presets, resolve_translation_choice
from yblocalizer.workbench_api import WorkbenchHandler
from yblocalizer.workbench_config import InputValidationError, default_options


def test_catalog_includes_cloud_vendors_and_local_servers() -> None:
    ids = {item["id"] for item in list_translation_presets()}
    assert {"deepseek", "openai", "moonshot", "zhipu", "qwen", "siliconflow", "openrouter", "groq", "ollama", "local", "custom"} <= ids
    for item in list_translation_presets():
        assert item["requirements"]
        assert item["engine"] in {"deepseek", "openai", "ollama"}


def test_moonshot_uses_its_own_endpoint_and_model() -> None:
    updates, summary = resolve_translation_choice("moonshot", "sk-moonshot", "https://evil.example/v1", "moonshot-v1-32k")
    assert updates["OPENAI_BASE_URL"] == "https://api.moonshot.cn/v1"
    assert updates["OPENAI_TRANSLATE_MODEL"] == "moonshot-v1-32k"
    assert updates["YBLOCALIZER_TRANSLATOR"] == "openai"
    assert summary["translation_base_url"] == "https://api.moonshot.cn/v1"
    assert "sk-moonshot" not in summary.values()


def test_cloud_preset_requires_a_key() -> None:
    with pytest.raises(InputValidationError, match="API Key"):
        resolve_translation_choice("groq", "", "", "")


def test_ollama_accepts_one_existing_model_and_ignores_keys() -> None:
    updates, summary = resolve_translation_choice("ollama", "sk-ignored", "", "qwen3.5:9b")
    assert "OPENAI_API_KEY" not in updates
    assert "DEEPSEEK_API_KEY" not in updates
    assert summary["translator"] == "ollama"
    assert summary["translate_model"] == "qwen3.5:9b"
    custom, _summary = resolve_translation_choice("ollama", "", "", "llama3.1:8b")
    assert custom["YBLOCALIZER_TRANSLATE_MODEL"] == "llama3.1:8b"
    with pytest.raises(InputValidationError, match="模型名"):
        resolve_translation_choice("ollama", "", "", "bad model")


def test_local_server_accepts_loopback_and_a_blank_key() -> None:
    updates, summary = resolve_translation_choice("local", "", "http://127.0.0.1:8000/v1", "qwen2.5-7b-instruct")
    assert updates["OPENAI_API_KEY"] == "local"
    assert updates["OPENAI_BASE_URL"] == "http://127.0.0.1:8000/v1"
    assert summary["translate_model"] == "qwen2.5-7b-instruct"


@pytest.mark.parametrize("base_url", ["file:///C:/model", "http://user:pass@127.0.0.1:1234/v1", "javascript:alert(1)", ""])
def test_custom_endpoint_rejects_unsafe_urls(base_url: str) -> None:
    with pytest.raises(InputValidationError):
        resolve_translation_choice("custom", "sk-custom", base_url, "my-model")


def test_key_page_stays_on_the_vendor_catalog() -> None:
    assert key_page_for("moonshot") == "https://platform.moonshot.cn/console/api-keys"
    with pytest.raises(InputValidationError):
        key_page_for("ollama")
    with pytest.raises(InputValidationError):
        key_page_for("https://evil.example")


def test_default_options_follow_the_saved_service(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("YBLOCALIZER_TRANSLATOR", "openai")
    monkeypatch.setenv("YBLOCALIZER_TRANSLATE_MODEL", "moonshot-v1-8k")
    options = default_options()
    assert options["translator"] == "openai"
    assert options["translate_model"] == "moonshot-v1-8k"


def test_save_settings_persists_a_compatible_vendor_without_returning_the_key() -> None:
    names = [
        "YBLOCALIZER_TRANSLATION_PRESET", "YBLOCALIZER_TRANSLATOR", "YBLOCALIZER_TRANSLATE_MODEL",
        "OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_TRANSLATE_MODEL",
    ]
    before = {name: os.environ.get(name) for name in names}
    handler = object.__new__(WorkbenchHandler)
    try:
        saved = handler._save_settings({
            "translator": "openai",
            "translation_preset": "siliconflow",
            "api_key": "sk-silicon",
            "base_url": "https://evil.example/v1",
            "translate_model": "",
            "cookies_from_browser": "",
            "cookies_file": "",
        })
        assert saved["translation_base_url"] == "https://api.siliconflow.cn/v1"
        assert saved["translate_model"] == "Qwen/Qwen2.5-7B-Instruct"
        assert os.environ["OPENAI_API_KEY"] == "sk-silicon"
        assert "sk-silicon" not in saved.values()
    finally:
        for name, value in before.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def test_save_settings_can_turn_translation_off_without_erasing_the_key() -> None:
    names = ["YBLOCALIZER_TRANSLATOR", "DEEPSEEK_API_KEY", "YBLOCALIZER_COOKIES_FROM_BROWSER"]
    before = {name: os.environ.get(name) for name in names}
    os.environ["DEEPSEEK_API_KEY"] = "sk-keep"
    handler = object.__new__(WorkbenchHandler)
    try:
        saved = handler._save_settings({
            "translator": "none",
            "translation_preset": "deepseek",
            "api_key": "sk-should-not-write",
            "cookies_from_browser": "edge",
            "cookies_file": "",
        })
        assert saved["translator"] == "none"
        assert saved["translation_preset"] == ""
        assert os.environ["YBLOCALIZER_TRANSLATOR"] == "none"
        assert os.environ["DEEPSEEK_API_KEY"] == "sk-keep"
        assert os.environ["YBLOCALIZER_COOKIES_FROM_BROWSER"] == "edge"
        assert "sk-should-not-write" not in os.environ.values()
    finally:
        for name, value in before.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
