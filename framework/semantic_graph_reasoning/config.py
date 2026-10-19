"""Credential-safe configuration for native OpenAI and compatible providers.

Native OpenAI credentials and OpenAI-compatible credentials are deliberately
kept separate. This prevents a native OpenAI key from being accidentally sent
to a compatible-provider base URL left over in a developer environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from semantic_graph_reasoning.exceptions import ConfigurationError

ProviderName = Literal["fake", "openai", "replay"]
TransportKind = Literal["responses", "chat_completions"]


@dataclass(frozen=True, slots=True)
class OpenAIConnection:
    """Resolved non-persistent connection details for one LLM invocation."""

    api_key: str = dataclass_field(repr=False)
    model: str
    base_url: str | None
    transport: TransportKind


class Settings(BaseSettings):
    """Environment-backed settings used by LLM and graph-execution integrations."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=False
    )

    openai_api_key: SecretStr | None = Field(
        default=None, repr=False, validation_alias="OPENAI_API_KEY"
    )
    openai_model: str | None = Field(default=None, validation_alias="OPENAI_MODEL")
    openai_base_url: str | None = Field(default=None, validation_alias="OPENAI_BASE_URL")

    deepseek_api_key: SecretStr | None = Field(
        default=None, repr=False, validation_alias="DEEPSEEK_API_KEY"
    )
    deepseek_model: str | None = Field(default=None, validation_alias="DEEPSEEK_MODEL")
    deepseek_base_url: str | None = Field(default=None, validation_alias="DEEPSEEK_BASE_URL")

    # Optional overrides match only roles that exist in the canonical framework.
    openai_planner_model: str | None = None
    openai_code_generator_model: str | None = None
    openai_local_inspector_model: str | None = None

    # GPT-5 snapshots can reject non-default sampling values. None preserves
    # the provider default unless a caller explicitly opts in.
    llm_temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    llm_timeout_seconds: int = Field(default=120, ge=1, le=3600)
    llm_max_retries: int = Field(default=3, ge=0, le=10)
    data_root: Path = Path("data")
    enable_tracing: bool = False

    @field_validator(
        "openai_model",
        "openai_base_url",
        "deepseek_model",
        "deepseek_base_url",
        "openai_planner_model",
        "openai_code_generator_model",
        "openai_local_inspector_model",
        mode="before",
    )
    @classmethod
    def blank_strings_to_none(cls, value: object) -> object:
        """Normalize blank environment values before deterministic fallback."""

        return None if isinstance(value, str) and not value.strip() else value

    def _role_override(self, role: str) -> str | None:
        return {
            "planner": self.openai_planner_model,
            "code_generator": self.openai_code_generator_model,
            "local_inspector": self.openai_local_inspector_model,
        }.get(role)

    def model_for_role(self, role: str) -> str | None:
        """Return a current-role override or the selected provider's shared model."""

        override = self._role_override(role)
        if override:
            return override
        if self.openai_api_key is not None:
            return self.openai_model
        if self.deepseek_api_key is not None:
            return self.deepseek_model
        return self.openai_model or self.deepseek_model

    @property
    def api_key_is_set(self) -> bool:
        """Report credential presence without revealing either credential."""

        return any(
            secret is not None and bool(secret.get_secret_value().strip())
            for secret in (self.openai_api_key, self.deepseek_api_key)
        )

    @property
    def active_base_url(self) -> str | None:
        """Return the active compatible base URL without exposing a key."""

        return self.deepseek_base_url if self.deepseek_api_key is not None else None

    def with_openai_model(self, model: str) -> Settings:
        """Return a copy pinned to one native OpenAI model identifier."""

        normalized = model.strip()
        if not normalized:
            raise ConfigurationError("OpenAI model override must not be blank.")
        return self.model_copy(update={"openai_model": normalized})

    def require_openai(self, role: str = "planner") -> OpenAIConnection:
        """Resolve one safe native or compatible provider connection."""

        native_key = _secret_value(self.openai_api_key)
        compatible_key = _secret_value(self.deepseek_api_key)
        if native_key and compatible_key:
            raise ConfigurationError(
                "Configure either native OPENAI_* credentials or DEEPSEEK_* compatible "
                "credentials for one run, not both."
            )
        if native_key:
            if self.openai_base_url:
                raise ConfigurationError(
                    "Native OPENAI_API_KEY must not be combined with OPENAI_BASE_URL. "
                    "Use explicit DEEPSEEK_* compatible-provider variables instead."
                )
            model = self._role_override(role) or self.openai_model
            if not model:
                raise ConfigurationError(
                    "OPENAI_MODEL (or a role-specific override) is required for native OpenAI."
                )
            return OpenAIConnection(
                api_key=native_key, model=model, base_url=None, transport="responses"
            )
        if compatible_key:
            if self.openai_base_url:
                raise ConfigurationError(
                    "DEEPSEEK_API_KEY must use DEEPSEEK_BASE_URL, not OPENAI_BASE_URL."
                )
            if not self.deepseek_base_url:
                raise ConfigurationError(
                    "DEEPSEEK_BASE_URL is required when DEEPSEEK_API_KEY is configured."
                )
            model = self._role_override(role) or self.deepseek_model
            if not model:
                raise ConfigurationError(
                    "DEEPSEEK_MODEL (or a role-specific override) is required for compatible "
                    "OpenAI transport."
                )
            return OpenAIConnection(
                api_key=compatible_key,
                model=model,
                base_url=self.deepseek_base_url,
                transport="chat_completions",
            )
        raise ConfigurationError(
            "An API key is required for provider=openai. Set OPENAI_API_KEY for native OpenAI "
            "or DEEPSEEK_API_KEY with DEEPSEEK_BASE_URL for a compatible provider."
        )

    def validate_provider(self, provider: ProviderName, role: str = "planner") -> None:
        """Validate a provider only when it actually needs a remote credential."""

        if provider == "openai":
            self.require_openai(role)


def _secret_value(value: SecretStr | None) -> str | None:
    """Extract a non-blank secret only at the final in-memory transport boundary."""

    if value is None:
        return None
    candidate = value.get_secret_value().strip()
    return candidate or None
