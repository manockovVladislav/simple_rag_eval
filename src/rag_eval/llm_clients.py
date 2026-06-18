from __future__ import annotations

import os
import uuid
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


class GigaChatApiClient(OpenAICompatibleClient):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._access_token: str | None = None

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        token = self.get_bearer_token()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    def get_bearer_token(self) -> str:
        if self.config.auth_type == "bearer_env":
            if not self.config.api_key_env:
                raise ValueError("GigaChat bearer auth requires api_key_env in config.")
            token = os.getenv(self.config.api_key_env, "")
            if not token:
                raise ValueError(f"Environment variable {self.config.api_key_env} is empty.")
            return token

        if self.config.auth_type == "gigachat_oauth":
            if self._access_token:
                return self._access_token
            if not self.config.token_url:
                raise ValueError("GigaChat OAuth auth requires token_url in config.")
            if not self.config.credentials_env:
                raise ValueError("GigaChat OAuth auth requires credentials_env in config.")
            credentials = os.getenv(self.config.credentials_env, "")
            if not credentials:
                raise ValueError(f"Environment variable {self.config.credentials_env} is empty.")
            response = requests.post(
                self.config.token_url,
                headers={
                    "Authorization": f"Basic {credentials}",
                    "RqUID": str(uuid.uuid4()),
                    "Content-Type": "application/x-www-form-urlencoded",
                },
                data={"scope": self.config.scope},
                timeout=self.config.timeout_seconds,
                verify=self.config.verify_ssl,
            )
            response.raise_for_status()
            data = response.json()
            token = data.get("access_token")
            if not token:
                raise ValueError("GigaChat token response does not contain access_token.")
            self._access_token = str(token)
            return self._access_token

        raise ValueError("GigaChat API config must use auth_type='bearer_env' or 'gigachat_oauth'.")


def make_model_client(provider: str, configs: dict[str, ModelConfig]) -> OpenAICompatibleClient:
    if provider not in configs:
        raise ValueError(f"Model provider '{provider}' is not configured.")
    config = configs[provider]
    if config.provider == "qwen_transformers":
        raise ValueError("Qwen is configured through transformers, not HTTP API. Use RagasEvaluator or your pipeline adapter.")
    if config.provider == "gigachat_api" or provider == "gigachat":
        return GigaChatApiClient(config)
    return OpenAICompatibleClient(config)


def _message_to_dict(message: ChatMessage | dict[str, Any]) -> dict[str, Any]:
    if isinstance(message, ChatMessage):
        return asdict(message)
    return message
