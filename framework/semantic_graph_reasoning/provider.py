"""Model communication and structured decisions for Coordinator and Inspector."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

from semantic_graph_reasoning.models import ExecutionAction, ExecutionLimits

T = TypeVar("T", bound=BaseModel)
ENDPOINT = "https://api.deepseek.com/beta"
OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = "google/gemini-3.8-flash"
OPENROUTER_PROVIDER = "google-ai-studio/flex"
DEEPSEEK_MODELS = ("deepseek-v4-pro",)
SUPPORTED_MODELS = ("gpt-5.6-luna", *DEEPSEEK_MODELS, OPENROUTER_MODEL)


def safe_message(value: str) -> str:
    return re.sub(r"(?:sk|rk|key)-[A-Za-z0-9_-]{8,}", "[REDACTED]", value)


TOOL_NAME = "submit_result"


def strict_schema(output_type: type[BaseModel]) -> dict[str, Any]:
    """Keep the contract intact in DeepSeek's documented reference dialect.

    Beta rejects an anyOf branch consisting only of a $ref (missing type).
    Inline that branch, while retaining recursive references elsewhere.
    """
    original = output_type.model_json_schema()
    definitions = original.get("$defs", {})

    def visit(value: Any) -> Any:
        if isinstance(value, list):
            return [visit(item) for item in value]
        if not isinstance(value, dict):
            return value
        result = {}
        for key, item in value.items():
            if key == "$ref":
                if not item.startswith("#/$defs/"):
                    raise ValueError("strict contract requires local definition references")
                result[key] = item.replace("#/$defs/", "#/$def/", 1)
            elif key == "anyOf":
                result[key] = [visit(definitions[branch["$ref"].removeprefix("#/$defs/")])
                               if set(branch) == {"$ref"} else visit(branch) for branch in item]
            else:
                result["$def" if key == "$defs" else key] = visit(item)
        return result

    result = dict(visit(original))
    rendered = result.pop("$def", {})

    def references(value: Any) -> set[str]:
        if isinstance(value, list):
            return set().union(*(references(item) for item in value))
        if not isinstance(value, dict):
            return set()
        found = {value["$ref"].removeprefix("#/$def/")} if "$ref" in value else set()
        return found.union(*(references(item) for item in value.values()))

    # Inlining anyOf branches makes some definitions unreachable (notably Plan).
    # Keep the transitive closure, including recursive Formula references.
    needed, pending = set(), references(result)
    while pending:
        name = pending.pop()
        if name not in needed:
            needed.add(name)
            pending.update(references(rendered[name]) - needed)
    if needed:
        result["$def"] = {name: definition for name, definition in rendered.items() if name in needed}
    return result


def tool_message(response: dict[str, Any]) -> dict[str, Any]:
    choices = response.get("choices") or []
    if len(choices) != 1 or choices[0].get("finish_reason") != "tool_calls":
        raise ValueError("strict output requires one completed tool-call choice")
    message = choices[0]["message"]
    calls = message.get("tool_calls") or []
    if (len(calls) != 1 or calls[0].get("type") != "function" or not calls[0].get("id")
            or calls[0].get("function", {}).get("name") != TOOL_NAME):
        raise ValueError("strict output requires exactly one submit_result function call")
    return dict(message)


def parse_tool_output(response: dict[str, Any], output_type: type[T]) -> T:
    message = tool_message(response)
    return output_type.model_validate_json(message["tool_calls"][0]["function"]["arguments"])


def strict_request(model: str, effort: str, inputs: list[dict[str, Any]],
                   output_type: type[BaseModel], max_output_tokens: int) -> dict[str, Any]:
    """Each decision uses a fresh request with the declared output schema."""
    if any(item["role"] not in {"system", "user"} for item in inputs):
        raise RuntimeError("stateless requests cannot include assistant/tool history")
    result = {"model": model, "messages": [{"role": item["role"], "content": item["content"]} for item in inputs],
              "tools": [{"type": "function", "function": {
                  "name": TOOL_NAME, "description": "Submit the single structured result for this agent turn.",
                  # Server strict mode requires every declared property, including
                  # unused action slots. Local typed validation still applies.
                  "strict": not issubclass(output_type, ExecutionAction), "parameters": strict_schema(output_type)}}],
              "tool_choice": {"type": "function", "function": {"name": TOOL_NAME}},
              "max_tokens": max_output_tokens, "temperature": 0, "stream": False, "store": False,
              "thinking": {"type": "disabled" if effort == "none" else "enabled"}}
    if effort != "none":
        result["reasoning_effort"] = effort
    return result

def routing() -> dict[str, Any]:
    return {"order": [OPENROUTER_PROVIDER], "only": [OPENROUTER_PROVIDER], "allow_fallbacks": False,
            "require_parameters": True}


def structured_request(model: str, effort: str, inputs: list[dict[str, Any]],
                       output_type: type[BaseModel], max_output_tokens: int) -> dict[str, Any]:
    from openai.lib._parsing._responses import type_to_text_format_param

    if model != OPENROUTER_MODEL or effort != "low":
        raise ValueError("OpenRouter request requires Gemini 3.8 Flash / low")
    if any(item["role"] not in {"system", "user"} for item in inputs):
        raise RuntimeError("stateless requests cannot include assistant/tool history")
    schema: dict[str, Any] = dict(type_to_text_format_param(output_type))
    if issubclass(output_type, ExecutionAction):
        schema.update(strict=False, schema=output_type.model_json_schema())
    schema.pop("type")
    # This route clips recursive structured-output arrays to zero elements.
    # Serialize only Formula as JSON text: no tree-depth limit or lost constraints.
    # The original recursive schema is supplied verbatim in the field description
    # and the decoded object must still pass the original Pydantic/runtime checks.
    definitions = schema["schema"].get("$defs", {})
    if "Formula" in definitions:
        formula_schema = {"$defs": {"Formula": definitions["Formula"]}, "$ref": "#/$defs/Formula"}
        definitions["Formula"] = {"type": "string", "description": (
            "JSON text encoding the full recursive Formula object, not a condition name. "
            "Decode this string to obtain the formula described in the agent instructions. "
            "All original fields are required, including children=[] on leaves; no depth truncation. "
            "Decoded JSON must satisfy this schema: " + json.dumps(formula_schema, ensure_ascii=False))}
    return {"model": model, "messages": inputs, "provider": routing(),
            "service_tier": "flex", "reasoning": {"effort": effort},
            "response_format": {"type": "json_schema", "json_schema": schema},
            "max_tokens": max_output_tokens, "temperature": 0, "stream": False, "store": False}


def validate_response(response: dict[str, Any]) -> None:
    """Check returned identity/tier; Chat does not promise an effort echo."""
    if response.get("model") != OPENROUTER_MODEL:
        raise ValueError("OpenRouter response model differs from the pinned model")
    if response.get("provider") != "Google AI Studio" or response.get("service_tier") != "flex":
        raise ValueError("OpenRouter response did not confirm Google AI Studio / flex")
    if response.get("reasoning") and response["reasoning"].get("effort") not in (None, "low"):
        raise ValueError("OpenRouter response reasoning effort contradicts low")


def parse_structured_output(response: dict[str, Any], output_type: type[T]) -> T:
    choices = response.get("choices") or []
    if len(choices) != 1 or choices[0].get("finish_reason") != "stop":
        raise ValueError("structured output requires one completed choice")
    message = choices[0].get("message") or {}
    if message.get("tool_calls") or message.get("refusal") or not isinstance(message.get("content"), str):
        raise ValueError("structured output requires JSON content without tool calls or refusal")
    decoded = json.loads(message["content"])
    if isinstance(decoded, dict) and isinstance(decoded.get("plan"), dict):
        formula = decoded["plan"].get("formula")
        if not isinstance(formula, str):
            raise ValueError("OpenRouter Plan formula must be serialized JSON text")
        decoded["plan"]["formula"] = json.loads(formula)
    return output_type.model_validate(decoded)

class ModelOutputError(RuntimeError):
    """Unusable model content fails this query without accepting a fabricated answer."""


class QueryBudgetError(RuntimeError):
    """A query exhausted its provider budget."""


class ProviderConfigurationError(RuntimeError):
    """The returned model or thinking configuration violates the request."""


def validate_model_effort(model: str, effort: str) -> None:
    allowed = {"low"} if model == OPENROUTER_MODEL else {"high", "none"}
    if model not in SUPPORTED_MODELS or effort not in allowed:
        raise ValueError("unsupported native model or reasoning effort")


def parse_output(response: dict[str, Any], output_type: type[T]) -> T:
    """Parse terminal messages only; commentary cannot issue runtime actions."""
    texts = [item["text"] for message in response["output"] if message["type"] == "message"
             and message.get("phase") in (None, "final_answer")
             for item in message["content"] if item["type"] == "output_text"]
    if not texts:
        raise ValueError("provider returned no structured output")
    outputs = [output_type.model_validate_json(text) for text in texts]
    if any(value != outputs[0] for value in outputs[1:]):
        raise ValueError("provider returned conflicting structured decisions")
    return outputs[0]


class Provider(Protocol):
    calls: int
    tokens: int

    async def request(self, role: str, instructions: str, messages: list[dict[str, Any]],
                      output_type: type[T]) -> T: ...


def request_input(instructions: str, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Canonical agent input; output contracts belong to the provider schema."""
    if any(item["role"] != "user" for item in messages):
        raise RuntimeError("stateless agents require freshly assembled user context")
    return [{"role": "system", "content": instructions}, *messages]

class NativeProvider:
    def __init__(self, model: str, limits: ExecutionLimits, reasoning_effort: str = "none") -> None:
        from dotenv import load_dotenv
        from openai import AsyncOpenAI

        validate_model_effort(model, reasoning_effort)
        load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)
        variable, endpoint = (("OpenRouter", OPENROUTER_ENDPOINT) if model == OPENROUTER_MODEL else
                              ("DEEPSEEK_API_KEY1", ENDPOINT) if model in DEEPSEEK_MODELS else
                              ("OPENAI_API_KEY", "https://api.openai.com/v1"))
        credential = os.environ.get(variable, "").strip()
        if not credential:
            raise ValueError(f"{variable} is required for {model}")
        self.client: Any = AsyncOpenAI(api_key=credential, base_url=endpoint,
                                      max_retries=0, timeout=limits.provider_seconds)
        self.model, self.limits = model, limits
        self.reasoning_effort = reasoning_effort
        self.calls = 0
        self.tokens = 0

    async def close(self) -> None:
        await self.client.close()

    async def request(self, role: str, instructions: str, messages: list[dict[str, Any]], output_type: type[T]) -> T:
        from openai.lib._parsing._responses import type_to_text_format_param
        if self.limits.max_calls is not None and self.calls >= self.limits.max_calls or (self.limits.max_tokens is not None and self.tokens >= self.limits.max_tokens):
            raise QueryBudgetError('query provider budget exhausted')
        self.calls += 1
        payload = request_input(instructions, messages)
        deepseek = self.model in DEEPSEEK_MODELS
        openrouter = self.model == OPENROUTER_MODEL
        chat = deepseek or openrouter
        wire = strict_request(self.model, self.reasoning_effort, payload, output_type, self.limits.max_output_tokens) if deepseek else None
        if openrouter:
            wire = structured_request(self.model, self.reasoning_effort, payload, output_type, self.limits.max_output_tokens)
        response_format = type_to_text_format_param(output_type)
        sparse_action = issubclass(output_type, ExecutionAction)
        if sparse_action:
            response_format = {'type': 'json_schema', 'name': output_type.__name__, 'strict': False, 'schema': output_type.model_json_schema()}
        if openrouter and wire is not None:
            extra = {k: wire[k] for k in ('provider', 'reasoning')}
            response = await self.client.chat.completions.create(**{k: v for k, v in wire.items() if k not in extra}, extra_body=extra)
        elif wire is not None:
            response = await self.client.chat.completions.create(**{k: v for k, v in wire.items() if k != 'thinking'}, extra_body={'thinking': wire['thinking']})
        else:
            response = await self.client.responses.create(model=self.model, input=payload, reasoning={'effort': self.reasoning_effort}, text={'format': response_format}, max_output_tokens=self.limits.max_output_tokens, temperature=0, store=False)
        raw = response.model_dump(mode='json', warnings=False)
        usage = response.usage
        if usage is not None:
            self.tokens += int(usage.total_tokens)
        if raw.get('model') != self.model:
            raise ProviderConfigurationError('provider response model differs from the requested native model')
        if openrouter:
            try:
                validate_response(raw)
            except ValueError as error:
                raise ProviderConfigurationError(str(error)) from error
        if not chat and (raw.get('reasoning') or {}).get('effort') != self.reasoning_effort:
            raise ProviderConfigurationError('provider response reasoning effort differs from the requested setting')
        if self.reasoning_effort == 'none':
            details = 'completion_tokens_details' if deepseek else 'output_tokens_details'
            reasoning_tokens = ((raw.get('usage') or {}).get(details) or {}).get('reasoning_tokens')
            reasoning_output = any(c.get('message', {}).get('reasoning_content') for c in raw.get('choices', [])) if deepseek else any(item.get('type') == 'reasoning' for item in raw.get('output', []))
            valid_counter = reasoning_tokens in (None, 0) if deepseek else reasoning_tokens == 0
            if not valid_counter or reasoning_output:
                raise ProviderConfigurationError('non-thinking response contains reasoning or incompatible usage')
        truncated = any(c.get('finish_reason') == 'length' for c in raw.get('choices', [])) if chat else raw.get('status') == 'incomplete' and (raw.get('incomplete_details') or {}).get('reason') == 'max_output_tokens'
        if truncated:
            raise ModelOutputError('model output exhausted max_output_tokens before completion')
        if not chat and response.status != 'completed':
            raise RuntimeError(f'provider did not complete the required output: {response.status}')
        try:
            if openrouter:
                parsed = parse_structured_output(raw, output_type)
            else:
                parsed = parse_tool_output(raw, output_type) if deepseek else parse_output(raw, output_type)
        except ValueError as error:
            raise ModelOutputError(str(error)) from error
        return parsed
