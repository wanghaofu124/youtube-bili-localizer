"""Translation vendor presets the workbench can actually call.

Cloud vendors other than DeepSeek go through the OpenAI-compatible client.
Ollama stays on its own local client. A recommended tag can be downloaded alone;
any other well-formed tag is used only when that model is already installed.
"""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import urlparse

from .dependencies import normalize_ollama_model_tag
from .workbench_config import InputValidationError, require_text

def _preset(
    preset_id: str,
    label: str,
    group: str,
    engine: str,
    *,
    base_url: str,
    model: str,
    key_url: str = "",
    key_placeholder: str = "sk-…",
    needs_key: bool = True,
    key_optional: bool = False,
    base_editable: bool = False,
    model_editable: bool = True,
    model_choices: tuple[tuple[str, str], ...] = (),
    requirements: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "id": preset_id,
        "label": label,
        "group": group,
        "engine": engine,
        "needs_key": needs_key,
        "key_optional": key_optional,
        "base_url": base_url,
        "base_editable": base_editable,
        "model": model,
        "model_editable": model_editable,
        "model_choices": [{"id": item_id, "label": item_label} for item_id, item_label in model_choices],
        "key_url": key_url,
        "key_placeholder": key_placeholder,
        "requirements": list(requirements),
    }


_PRESET_LIST: tuple[dict[str, Any], ...] = (
    _preset(
        "deepseek", "DeepSeek", "online", "deepseek",
        base_url="https://api.deepseek.com",
        model="deepseek-v4-flash",
        key_url="https://platform.deepseek.com/api_keys",
        requirements=(
            "需要能访问 api.deepseek.com。按 token 计费，密钥只写入本机，界面不会再显示。",
            "在 platform.deepseek.com 注册后，打开 API keys 创建密钥，一般以 sk- 开头。",
            "接口地址固定为 https://api.deepseek.com，不要自行加上 /v1。",
            "默认模型 deepseek-v4-flash。翻译设置里的思考模式默认关闭，这是最省的一档。",
            "模型名可以改成账号里已开通的其他 DeepSeek 模型。",
        ),
    ),
    _preset(
        "openai", "OpenAI", "online", "openai",
        base_url="https://api.openai.com/v1",
        model="gpt-4.1-mini",
        key_url="https://platform.openai.com/api-keys",
        requirements=(
            "需要能访问 api.openai.com。费用记在 OpenAI 账单上。",
            "在 platform.openai.com 创建 API Key，然后粘贴到下面。",
            "接口地址固定为 https://api.openai.com/v1。",
            "默认模型 gpt-4.1-mini。账号没有开通这个模型时，改成已开通的模型 ID。",
            "密钥写入本机 OPENAI_API_KEY。换到其他兼容厂商时，会覆盖这一项。",
        ),
    ),
    _preset(
        "moonshot", "Moonshot（Kimi）", "online", "openai",
        base_url="https://api.moonshot.cn/v1",
        model="moonshot-v1-8k",
        key_url="https://platform.moonshot.cn/console/api-keys",
        requirements=(
            "需要能访问 api.moonshot.cn。按 token 计费。",
            "在 Moonshot 开放平台创建 API Key。",
            "接口必须是 https://api.moonshot.cn/v1。程序按 OpenAI Chat Completions 调用。",
            "默认模型 moonshot-v1-8k，上下文较短。更长字幕可改成账号已开通的模型，例如 moonshot-v1-32k。",
            "密钥写入 OPENAI_API_KEY，会覆盖以前保存的 OpenAI 兼容密钥。",
        ),
    ),
    _preset(
        "zhipu", "智谱 GLM", "online", "openai",
        base_url="https://open.bigmodel.cn/api/paas/v4",
        model="glm-4-flash",
        key_url="https://open.bigmodel.cn/usercenter/apikeys",
        requirements=(
            "需要能访问 open.bigmodel.cn。按智谱的计费规则扣费。",
            "在智谱开放平台的 API Key 页面创建密钥。",
            "接口必须是 https://open.bigmodel.cn/api/paas/v4，不要改成带 /v1 的其他路径。",
            "默认模型 glm-4-flash。模型名必须是账号已开通、且支持对话接口的 ID。",
            "密钥写入 OPENAI_API_KEY，会覆盖以前保存的 OpenAI 兼容密钥。",
        ),
    ),
    _preset(
        "qwen", "通义千问（百炼兼容模式）", "online", "openai",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="qwen-plus",
        key_url="https://bailian.console.aliyun.com/",
        requirements=(
            "需要阿里云百炼的 API Key，并使用它的 OpenAI 兼容模式。",
            "接口必须是 https://dashscope.aliyuncs.com/compatible-mode/v1。不要填 DashScope 原生地址。",
            "默认模型 qwen-plus。请改成控制台里已开通的模型 ID。",
            "密钥在百炼控制台创建，写入 OPENAI_API_KEY，会覆盖以前的兼容密钥。",
        ),
    ),
    _preset(
        "siliconflow", "硅基流动", "online", "openai",
        base_url="https://api.siliconflow.cn/v1",
        model="Qwen/Qwen2.5-7B-Instruct",
        key_url="https://cloud.siliconflow.cn/account/ak",
        requirements=(
            "需要硅基流动账号和 API Key。按该平台的模型价格计费。",
            "接口固定为 https://api.siliconflow.cn/v1。",
            "默认模型 Qwen/Qwen2.5-7B-Instruct。模型名必须带厂商前缀，和模型广场里的 ID 一致。",
            "密钥写入 OPENAI_API_KEY，会覆盖以前保存的 OpenAI 兼容密钥。",
        ),
    ),
    _preset(
        "openrouter", "OpenRouter", "online", "openai",
        base_url="https://openrouter.ai/api/v1",
        model="openai/gpt-4.1-mini",
        key_url="https://openrouter.ai/keys",
        requirements=(
            "需要 OpenRouter 的 API Key。费用从 OpenRouter 账户扣除，不从本程序扣。",
            "接口固定为 https://openrouter.ai/api/v1。",
            "默认模型 openai/gpt-4.1-mini。模型名必须是 OpenRouter 的完整 ID，例如 deepseek/deepseek-chat。",
            "密钥写入 OPENAI_API_KEY，会覆盖以前保存的 OpenAI 兼容密钥。",
        ),
    ),
    _preset(
        "groq", "Groq", "online", "openai",
        base_url="https://api.groq.com/openai/v1",
        model="llama-3.3-70b-versatile",
        key_url="https://console.groq.com/keys",
        requirements=(
            "需要 Groq 控制台的 API Key。按 Groq 的额度或价格使用。",
            "接口固定为 https://api.groq.com/openai/v1。",
            "默认模型 llama-3.3-70b-versatile。如果接口提示模型不存在，改成控制台里当前可用的模型 ID。",
            "密钥写入 OPENAI_API_KEY，会覆盖以前保存的 OpenAI 兼容密钥。",
        ),
    ),
    _preset(
        "ollama", "Ollama（本机）", "local", "ollama",
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3.5:4b",
        needs_key=False,
        model_editable=True,
        model_choices=(
            ("qwen3.5:4b", "qwen3.5:4b · 约 3.4 GB，选这一个即可"),
            ("qwen3.5:9b", "qwen3.5:9b · 约 6.6 GB，选这一个即可"),
            ("translategemma:12b", "translategemma:12b · 约 8.1 GB，选这一个即可"),
        ),
        requirements=(
            "不需要密钥。字幕留在本机，不发到云端。",
            "先安装 Ollama，并保持它在运行。默认地址是 http://127.0.0.1:11434。",
            "下面三个只是可选项，选一个就行，不会把三个模型一起下载。",
            "Ollama 里已经有别的模型时，选「本机已经有的其他模型」，填写和 ollama list 一致的名字，不必再下载。",
            "只有你点「下载所选模型」时，才会下载当前这一个。已经有的模型不会重复下载。",
        ),
    ),
    _preset(
        "local", "本机兼容接口", "local", "openai",
        base_url="http://127.0.0.1:1234/v1",
        model="",
        key_placeholder="多数服务可留空，或填写 local",
        key_optional=True,
        base_editable=True,
        requirements=(
            "用于 LM Studio、vLLM、llama.cpp server 等已经打开 OpenAI 兼容接口的程序。",
            "服务必须提供 /v1/chat/completions。先在那个程序里加载模型并启动服务。",
            "LM Studio 默认 http://127.0.0.1:1234/v1。vLLM 常见 http://127.0.0.1:8000/v1。llama.cpp server 常见 http://127.0.0.1:8080/v1。",
            "模型名必须和该服务里已加载的那一个名称完全一致，例如 qwen2.5-7b-instruct。程序不会替你下载模型。",
            "密钥可留空，程序会写入 local。如果那个服务要求特定密钥，按它的说明填写。",
            "地址只接受 http 或 https，可以是本机或局域网，不能带账号密码。",
        ),
    ),
    _preset(
        "custom", "其他兼容接口", "online", "openai",
        base_url="",
        model="",
        key_placeholder="按该厂商要求填写",
        base_editable=True,
        requirements=(
            "用于上面没有列出、但提供 OpenAI Chat Completions 的服务。",
            "接口地址以 http 或 https 开头，通常以 /v1 结尾，不能带账号密码。",
            "模型名按该厂商文档填写，必须是它实际接受的 ID。",
            "密钥写入 OPENAI_API_KEY，会覆盖以前保存的 OpenAI 兼容密钥。",
            "只提供自家私有接口、没有 Chat Completions 的服务不能用这一项。官方 DeepSeek 请选第一项。",
        ),
    ),
)
_PRESETS = {item["id"]: item for item in _PRESET_LIST}


def list_translation_presets() -> list[dict[str, Any]]:
    return [dict(item) for item in _PRESET_LIST]


def translation_preset(preset_id: str) -> dict[str, Any] | None:
    preset = _PRESETS.get(str(preset_id or "").strip())
    return dict(preset) if preset else None


def saved_translation() -> dict[str, str]:
    """Public, secret-free view of the persisted vendor choice."""
    preset_id = os.getenv("YBLOCALIZER_TRANSLATION_PRESET", "").strip()
    translator = os.getenv("YBLOCALIZER_TRANSLATOR", "").strip().lower()
    preset = _PRESETS.get(preset_id)
    if preset is None and translator in _PRESETS:
        preset = _PRESETS[translator]
    if preset is None or translator == "none":
        return {"translation_preset": "", "translation_base_url": "", "translate_model": os.getenv("YBLOCALIZER_TRANSLATE_MODEL", "")}
    if preset["engine"] == "deepseek":
        base = os.getenv("DEEPSEEK_BASE_URL", preset["base_url"])
    elif preset["engine"] == "openai":
        base = os.getenv("OPENAI_BASE_URL", preset["base_url"])
    else:
        base = preset["base_url"]
    return {
        "translation_preset": preset["id"],
        "translation_base_url": base,
        "translate_model": os.getenv("YBLOCALIZER_TRANSLATE_MODEL", "") or preset["model"],
    }


def key_page_for(preset_id: str) -> str:
    preset = _PRESETS.get(str(preset_id or "deepseek").strip())
    if preset is None:
        raise InputValidationError("没有这个翻译服务。", code="invalid_choice", field="preset")
    if not preset["key_url"]:
        raise InputValidationError("这个服务没有可打开的申请页面。", code="invalid_choice", field="preset")
    return str(preset["key_url"])


def _service_base_url(value: str) -> str:
    text = require_text(value, "base_url", 2048, allow_empty=False).strip()
    if any(character.isspace() for character in text):
        raise InputValidationError("接口地址不能包含空格。", field="base_url")
    parsed = urlparse(text)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise InputValidationError("接口地址必须是不带账号密码的 http 或 https 地址。", field="base_url")
    return text.rstrip("/")


def resolve_translation_choice(
    preset_id: str,
    api_key: str,
    base_url: str,
    model: str,
    existing_key: str = "",
) -> tuple[dict[str, str], dict[str, str]]:
    """Validate one preset and return env updates plus a secret-free summary."""
    preset = _PRESETS.get(str(preset_id or "").strip())
    if preset is None:
        raise InputValidationError("没有这个翻译服务。", code="invalid_choice", field="translation_preset")
    chosen_base = preset["base_url"]
    if preset["base_editable"]:
        chosen_base = _service_base_url(base_url)
    elif preset["base_url"]:
        chosen_base = _service_base_url(preset["base_url"])
    chosen_model = require_text(model, "translate_model", 128).strip() or str(preset["model"])
    if preset["engine"] == "ollama":
        try:
            chosen_model = normalize_ollama_model_tag(chosen_model)
        except ValueError as exc:
            raise InputValidationError(str(exc), code="invalid_choice", field="translate_model") from exc
    elif not chosen_model:
        raise InputValidationError("请填写模型名。", code="required", field="translate_model")
    key = require_text(api_key, "api_key", 1024).strip()
    kept = require_text(existing_key, "api_key", 1024).strip()
    if preset["engine"] == "ollama":
        key = ""
    elif not key and kept:
        key = kept
    elif not key and preset["key_optional"]:
        key = "local"
    elif not key:
        raise InputValidationError("请填写 API Key。", code="required", field="api_key")
    updates = {
        "YBLOCALIZER_TRANSLATION_PRESET": preset["id"],
        "YBLOCALIZER_TRANSLATOR": preset["engine"],
        "YBLOCALIZER_TRANSLATE_MODEL": chosen_model,
    }
    if preset["engine"] == "deepseek":
        updates["DEEPSEEK_API_KEY"] = key
        updates["DEEPSEEK_BASE_URL"] = chosen_base
        updates["DEEPSEEK_TRANSLATE_MODEL"] = chosen_model
    elif preset["engine"] == "openai":
        updates["OPENAI_API_KEY"] = key
        updates["OPENAI_BASE_URL"] = chosen_base
        updates["OPENAI_TRANSLATE_MODEL"] = chosen_model
    summary = {
        "translator": preset["engine"],
        "translate_model": chosen_model,
        "translation_preset": preset["id"],
        "translation_base_url": chosen_base,
    }
    return updates, summary
