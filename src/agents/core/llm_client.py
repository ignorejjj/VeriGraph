from __future__ import annotations

import asyncio
import copy
import inspect
from typing import Any, Dict, List, Optional


DEFAULT_LLM_CONFIG: Dict[str, Any] = {
    "temperature": 0.6,
    "top_p": 0.95,
    "max_tokens": 8192,
    "timeout": 600,
    "extra_body": {
        "top_k": 20,
        "include_stop_str_in_output": True,
        "repetition_penalty": 1.1,
    },
}


class LLMClient:
    def __init__(self, llm_config: Dict[str, Any]):
        config = copy.deepcopy(DEFAULT_LLM_CONFIG)
        config.update({key: value for key, value in (llm_config or {}).items() if key != "extra_body"})
        config["extra_body"].update((llm_config or {}).get("extra_body", {}))
        self.model_name = config.pop("model_name")
        self.api_key = config.pop("api_key")
        self.api_url = config.pop("api_url")
        self.config = config
        self._sync_client = None
        self._async_client = None

    def _get_sync_client(self):
        if self._sync_client is None:
            from openai import OpenAI

            self._sync_client = OpenAI(api_key=self.api_key, base_url=self.api_url)
        return self._sync_client

    def _get_async_client(self):
        if self._async_client is None:
            from openai import AsyncOpenAI

            self._async_client = AsyncOpenAI(api_key=self.api_key, base_url=self.api_url)
        return self._async_client

    def _build_request_kwargs(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        stop: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        if (prompt is None) == (messages is None):
            raise ValueError("Exactly one of `prompt` or `messages` must be provided.")

        request_kwargs = copy.deepcopy(self.config)
        request_kwargs["model"] = self.model_name
        request_kwargs["stop"] = stop or []
        request_kwargs.setdefault("extra_body", {})
        if prompt is not None:
            request_kwargs["prompt"] = prompt
        else:
            request_kwargs["messages"] = messages
        return request_kwargs

    def _coerce_response_text(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        if isinstance(value, list):
            parts: List[str] = []
            for item in value:
                text = self._coerce_response_text(item)
                if text:
                    parts.append(text)
            return "".join(parts)
        if isinstance(value, dict):
            for key in ("text", "content", "output_text"):
                if key in value:
                    return self._coerce_response_text(value.get(key))
            return str(value)

        text = getattr(value, "text", None)
        if text is not None:
            return self._coerce_response_text(text)
        content = getattr(value, "content", None)
        if content is not None:
            return self._coerce_response_text(content)
        return str(value)

    def _extract_chat_text(self, response: Any) -> str:
        message = getattr(response.choices[0], "message", None)
        if message is None:
            return ""
        return self._coerce_response_text(getattr(message, "content", None))

    def generate(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        stop: Optional[List[str]] = None,
    ) -> str:
        request_kwargs = self._build_request_kwargs(prompt=prompt, messages=messages, stop=stop)
        if messages is not None:
            response = self._get_sync_client().chat.completions.create(**request_kwargs)
            return self._extract_chat_text(response)
        response = self._get_sync_client().completions.create(**request_kwargs)
        return str(response.choices[0].text or "")

    async def agenerate(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        stop: Optional[List[str]] = None,
    ) -> str:
        request_kwargs = self._build_request_kwargs(prompt=prompt, messages=messages, stop=stop)
        if messages is not None:
            response = await self._get_async_client().chat.completions.create(**request_kwargs)
            return self._extract_chat_text(response)
        response = await self._get_async_client().completions.create(**request_kwargs)
        return str(response.choices[0].text or "")

    def _close_sync_client(self) -> None:
        client = self._sync_client
        self._sync_client = None
        if client is None:
            return

        close = getattr(client, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass

    async def _close_async_client(self) -> None:
        client = self._async_client
        self._async_client = None
        if client is None:
            return

        aclose = getattr(client, "aclose", None)
        close = getattr(client, "close", None)
        try:
            if callable(aclose):
                result = aclose()
                if inspect.isawaitable(result):
                    await result
                return
            if callable(close):
                result = close()
                if inspect.isawaitable(result):
                    await result
        except Exception:
            pass

    def close(self) -> None:
        self._close_sync_client()

        client = self._async_client
        self._async_client = None
        if client is None:
            return

        aclose = getattr(client, "aclose", None)
        close = getattr(client, "close", None)
        try:
            if callable(aclose):
                try:
                    asyncio.get_running_loop()
                except RuntimeError:
                    asyncio.run(self._close_async_client_with_callable(aclose))
                return
            if callable(close):
                result = close()
                if inspect.isawaitable(result):
                    try:
                        asyncio.get_running_loop()
                    except RuntimeError:
                        asyncio.run(self._await_close_result(result))
        except Exception:
            pass

    async def aclose(self) -> None:
        self._close_sync_client()
        await self._close_async_client()

    async def _close_async_client_with_callable(self, aclose_callable) -> None:
        result = aclose_callable()
        if inspect.isawaitable(result):
            await result

    async def _await_close_result(self, result) -> None:
        if inspect.isawaitable(result):
            await result
