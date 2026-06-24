from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any

import requests

from rag_eval.config import ModelConfig


@dataclass(slots=True)
class ChatMessage:
    role: str
    content: str


class OpenAICompatibleClient:
    def __init__(self, config: ModelConfig):
        self.config = config

    def chat(self, messages: list[ChatMessage | dict[str, Any]]) -> str:
        headers = self._headers()

        payload = {
            "model": self.config.model,
            "messages": [_message_to_dict(message) for message in messages],
            "temperature": self.config.temperature,
        }
        response = requests.post(
            self.config.base_url,
            headers=headers,
            json=payload,
            timeout=self.config.timeout_seconds,
            verify=self.config.verify_ssl,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.config.auth_type == "bearer_env" and self.config.api_key_env:
            api_key = os.getenv(self.config.api_key_env, "")
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
        return headers


class GigaChatLangChainClient:
    def __init__(self, config: ModelConfig):
        self.config = config
        self._client = None

    def chat(self, messages: list[ChatMessage | dict[str, Any]]) -> str:
        response = self._model().invoke([_to_langchain_message(message) for message in messages])
        return str(response.content)

    def _model(self):
        if self._client is None:
            try:
                from langchain_gigachat.chat_models import GigaChat
            except ImportError:
                from langchain_community.chat_models.gigachat import GigaChat

            self._client = GigaChat(
                base_url=self.config.base_url,
                access_token=self.config.access_token,
                model=self.config.model,
                temperature=self.config.temperature,
                rate_limiter=_rate_limiter(self.config),
            )
        return self._client


class TransformersLangChainClient:
    def __init__(self, config: ModelConfig):
        self.config = config
        self._client = None

    def chat(self, messages: list[ChatMessage | dict[str, Any]]) -> str:
        return str(self._model().invoke(_messages_to_prompt(messages))).strip()

    def _model(self):
        if self._client is None:
            from langchain_community.llms import HuggingFacePipeline

            model_kwargs: dict[str, Any] = {}
            if self.config.device_map:
                model_kwargs["device_map"] = self.config.device_map
            if self.config.torch_dtype:
                model_kwargs["torch_dtype"] = self.config.torch_dtype
            if self.config.local_files_only:
                model_kwargs["local_files_only"] = True

            pipeline_kwargs = {
                "max_new_tokens": self.config.max_new_tokens,
                "do_sample": self.config.do_sample,
                "temperature": self.config.temperature,
                "return_full_text": self.config.return_full_text,
            }

            kwargs: dict[str, Any] = {
                "model_id": self.config.model,
                "task": self.config.task,
                "model_kwargs": model_kwargs,
                "pipeline_kwargs": pipeline_kwargs,
                "batch_size": 1,
            }
            if self.config.device is not None:
                kwargs["device"] = self.config.device
            self._client = HuggingFacePipeline.from_model_id(**kwargs)
        return self._client


def make_model_client(provider: str, configs: dict[str, ModelConfig]) -> Any:
    if provider not in configs:
        raise ValueError(f"Model provider '{provider}' is not configured.")
    config = configs[provider]
    if config.provider == "qwen_transformers":
        return TransformersLangChainClient(config)
    if config.provider == "gigachat_api" or provider == "gigachat":
        return GigaChatLangChainClient(config)
    return OpenAICompatibleClient(config)


def _message_to_dict(message: ChatMessage | dict[str, Any]) -> dict[str, Any]:
    if isinstance(message, ChatMessage):
        return asdict(message)
    return message


def _to_langchain_message(message: ChatMessage | dict[str, Any]):
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    data = _message_to_dict(message)
    role = data.get("role", "user")
    content = str(data.get("content", ""))
    if role == "system":
        return SystemMessage(content=content)
    if role == "assistant":
        return AIMessage(content=content)
    return HumanMessage(content=content)


def _messages_to_prompt(messages: list[ChatMessage | dict[str, Any]]) -> str:
    parts: list[str] = []
    for message in messages:
        data = _message_to_dict(message)
        role = str(data.get("role", "user")).strip()
        content = str(data.get("content", "")).strip()
        if content:
            parts.append(f"{role}: {content}")
    parts.append("assistant:")
    return "\n\n".join(parts)


def _rate_limiter(config: ModelConfig):
    if config.min_seconds_between_requests <= 0:
        return None
    from langchain_core.rate_limiters import InMemoryRateLimiter

    return InMemoryRateLimiter(
        requests_per_second=1 / config.min_seconds_between_requests,
        check_every_n_seconds=0.1,
        max_bucket_size=1,
    )
