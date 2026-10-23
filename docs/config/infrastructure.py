from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import Field, SecretStr, field_validator, model_validator

from config.models import FrozenConfig


class MySqlCharset(StrEnum):
    UTF8MB4 = "utf8mb4"
    UTF8 = "utf8"
    LATIN1 = "latin1"
    CP1251 = "cp1251"


class MySqlConnectionConfig(FrozenConfig):
    host: str = "localhost"
    port: int = Field(default=3306, ge=1, le=65535)
    unix_socket: Path | None = None

    username: str
    password: SecretStr
    database: str

    charset: MySqlCharset = MySqlCharset.UTF8MB4
    connect_timeout: int = Field(default=5, ge=1)
    read_timeout: int | None = Field(default=None, ge=1)

    min_pool_size: int = Field(default=1, ge=0)
    max_pool_size: int = Field(default=10, ge=1)
    pool_recycle: int = -1
    echo: bool = False

    autocommit: bool = True
    read_only: bool = True
    init_command: str | None = None
    sql_mode: str | None = None
    local_infile: bool = False
    program_name: str | None = None
    server_public_key: str | None = None
    client_flag: int | None = None

    ssl_enabled: bool = False
    ssl_ca_file: Path | None = None
    ssl_cert_file: Path | None = None
    ssl_key_file: Path | None = None
    ssl_key_password: SecretStr | None = None
    ssl_ciphers: str | None = None
    ssl_check_hostname: bool = True
    ssl_verify_cert: bool = True

    extra_options: dict[str, Any] = Field(default_factory=dict)

    @field_validator("host", "username", "database")
    @classmethod
    def validate_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("MySQL host, username, and database must not be empty")
        return value

    @model_validator(mode="after")
    def validate_pool_size(self) -> MySqlConnectionConfig:
        if self.min_pool_size > self.max_pool_size:
            raise ValueError("MySQL min_pool_size cannot be greater than max_pool_size")
        if self.ssl_key_file is not None and self.ssl_cert_file is None:
            raise ValueError("MySQL ssl_key_file requires ssl_cert_file")
        if self.ssl_key_password is not None and self.ssl_cert_file is None:
            raise ValueError("MySQL ssl_key_password requires ssl_cert_file")
        return self


class ClickHouseProtocol(StrEnum):
    HTTP = "http"
    HTTPS = "https"


class ClickHouseCompression(StrEnum):
    NONE = "none"
    LZ4 = "lz4"
    ZSTD = "zstd"
    GZIP = "gzip"
    BROTLI = "br"


class ClickHouseConnectionConfig(FrozenConfig):
    dsn: SecretStr | None = None

    host: str = "localhost"
    port: int | None = Field(default=None, ge=1, le=65535)

    username: str = "default"
    password: SecretStr = SecretStr("")
    database: str | None = None

    protocol: ClickHouseProtocol = ClickHouseProtocol.HTTP
    secure: bool = False
    verify: bool = True
    ca_cert: Path | None = None
    client_cert: Path | None = None
    client_cert_key: Path | None = None
    server_host_name: str | None = None

    connect_timeout: int = Field(default=10, ge=1)
    send_receive_timeout: int = Field(default=300, ge=1)

    compression: ClickHouseCompression = ClickHouseCompression.LZ4
    query_limit: int = Field(default=0, ge=0)
    query_retries: int = Field(default=2, ge=0)

    autogenerate_session_id: bool = False
    session_id: str | None = None
    app_name: str | None = "relatives-api"

    http_proxy: str | None = None
    https_proxy: str | None = None

    read_only: bool = True
    settings: dict[str, Any] = Field(default_factory=dict)
    extra_options: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_security(self) -> ClickHouseConnectionConfig:
        if self.secure and self.protocol is ClickHouseProtocol.HTTP:
            raise ValueError("ClickHouse secure=True requires protocol=https")

        return self


class S3AddressingStyle(StrEnum):
    AUTO = "auto"
    PATH = "path"
    VIRTUAL = "virtual"


class S3SignatureVersion(StrEnum):
    S3V4 = "s3v4"


class S3RetryMode(StrEnum):
    LEGACY = "legacy"
    STANDARD = "standard"
    ADAPTIVE = "adaptive"


class S3ConnectionConfig(FrozenConfig):
    bucket: str

    endpoint_url: str | None = None
    region_name: str | None = None

    access_key_id: str
    secret_access_key: SecretStr

    secure: bool = True
    verify: bool | str = True

    addressing_style: S3AddressingStyle = S3AddressingStyle.AUTO
    signature_version: S3SignatureVersion = S3SignatureVersion.S3V4

    connect_timeout: int = Field(default=5, ge=1)
    read_timeout: int = Field(default=60, ge=1)
    max_pool_connections: int = Field(default=10, ge=1)
    retries_max_attempts: int | None = Field(default=None, ge=0)
    retries_mode: S3RetryMode | None = None
    user_agent_extra: str | None = None

    create_bucket_if_missing: bool = False

    extra_config: dict[str, Any] = Field(default_factory=dict)

    @field_validator("bucket")
    @classmethod
    def validate_bucket(cls, value: str) -> str:
        if not value:
            raise ValueError("S3 bucket must not be empty")

        return value


class MetrikaConnectionConfig(FrozenConfig):
    oauth_token: SecretStr
    counter_id: int = Field(ge=1)

    base_url: str = "https://api-metrika.yandex.net"
    timezone: str = "+03:00"
    accuracy: str = "full"

    connect_timeout: float = Field(default=10.0, ge=0.1)
    timeout: float = Field(default=60.0, ge=0.1)
    verify: bool = True

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: str) -> str:
        value = value.rstrip("/")

        if not value:
            raise ValueError("Metrika base_url must not be empty")

        return value

    @field_validator("timezone")
    @classmethod
    def validate_timezone(cls, value: str) -> str:
        if not value:
            raise ValueError("Metrika timezone must not be empty")

        return value


class GrowthBookConnectionConfig(FrozenConfig):
    api_key: SecretStr
    base_url: str

    connect_timeout: float = Field(default=10.0, ge=0.1)
    timeout: float = Field(default=20.0, ge=0.1)
    verify: bool = True

    max_retries: int = Field(default=4, ge=1)
    retry_base_seconds: float = Field(default=2.0, ge=0)

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: str) -> str:
        value = value.rstrip("/")

        if not value:
            raise ValueError("GrowthBook base_url must not be empty")

        return value


class BitrixConnectionConfig(FrozenConfig):
    webhook_url: SecretStr
    bot_id: int = Field(ge=1)
    client_id: str
    dialog_id: str

    connect_timeout: float = Field(default=10.0, ge=0.1)
    timeout: float = Field(default=30.0, ge=0.1)
    verify: bool = True

    @field_validator("webhook_url")
    @classmethod
    def validate_webhook_url(cls, value: SecretStr) -> SecretStr:
        raw = value.get_secret_value().rstrip("/")

        if not raw:
            raise ValueError("Bitrix webhook_url must not be empty")

        return SecretStr(raw)

    @field_validator("client_id", "dialog_id")
    @classmethod
    def validate_non_empty(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("Bitrix client_id and dialog_id must not be empty")

        return value


class LLMDeployment(StrEnum):
    GPT_5dot5 = "gpt-5.5"


class LLMConnectionConfig(FrozenConfig):
    api_key: SecretStr
    endpoint: str
    api_version: str = "2024-12-01-preview"
    deployment: LLMDeployment = LLMDeployment.GPT_5dot5

    timeout: float = Field(default=60.0, ge=0.1)
    max_completion_tokens: int = Field(default=4000, ge=1)
    max_tool_rounds: int = Field(default=20, ge=1)

    @field_validator("endpoint")
    @classmethod
    def validate_endpoint(cls, value: str) -> str:
        value = value.rstrip("/")

        if not value:
            raise ValueError("LLM endpoint must not be empty")

        return value

    @field_validator("api_version")
    @classmethod
    def validate_api_version(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("LLM api_version must not be empty")

        return value


class LangfuseConnectionConfig(FrozenConfig):
    public_key: str
    secret_key: SecretStr
    base_url: str

    timeout: int = Field(default=10, ge=1)
    prompt_label: str = "production"
    cache_ttl_seconds: int = Field(default=60, ge=0)

    @field_validator("public_key", "prompt_label")
    @classmethod
    def validate_non_empty(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("Langfuse public_key and prompt_label must not be empty")

        return value

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: str) -> str:
        value = value.rstrip("/")

        if not value:
            raise ValueError("Langfuse base_url must not be empty")

        return value


class YandexWikiAuthScheme(StrEnum):
    OAUTH = "OAuth"
    IAM = "Bearer"


class YandexWikiConnectionConfig(FrozenConfig):
    token: SecretStr
    auth_scheme: YandexWikiAuthScheme = YandexWikiAuthScheme.OAUTH
    org_id: str | None = None
    cloud_org_id: str | None = None

    base_url: str = "https://api.wiki.yandex.net/v1"
    # An existing page readable by this user; used only for the startup probe.
    ping_slug: str
    connect_timeout: float = Field(default=10.0, ge=0.1)
    timeout: float = Field(default=30.0, ge=0.1)
    verify: bool = True

    @field_validator("base_url", "ping_slug", "org_id", "cloud_org_id")
    @classmethod
    def validate_non_empty(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("Wiki URL, page slug and organization IDs must not be empty")
        return value

    @field_validator("base_url")
    @classmethod
    def normalize_base_url(cls, value: str) -> str:
        return value.rstrip("/")

    @model_validator(mode="after")
    def validate_auth(self) -> YandexWikiConnectionConfig:
        if not self.token.get_secret_value().strip():
            raise ValueError("Wiki token must not be empty")
        if (self.org_id is None) == (self.cloud_org_id is None):
            raise ValueError("Set exactly one of Wiki org_id or cloud_org_id")
        if self.auth_scheme is YandexWikiAuthScheme.IAM and self.cloud_org_id is None:
            raise ValueError("Wiki IAM authentication requires cloud_org_id")
        return self


class RedisConnectionConfig(FrozenConfig):
    # redis://, rediss:// (TLS), or unix://; credentials stay masked in repr.
    url: SecretStr = SecretStr("redis://localhost:6379/0")
    decode_responses: bool = False
    encoding: str = "utf-8"
    connect_timeout: float = Field(default=5.0, ge=0.1)
    timeout: float = Field(default=10.0, ge=0.1)
    max_connections: int = Field(default=10, ge=1)
    health_check_interval: int = Field(default=30, ge=0)

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().startswith(("redis://", "rediss://", "unix://")):
            raise ValueError("Redis URL must use redis://, rediss:// or unix://")
        return value


class InfrastructureConfig(FrozenConfig):
    mysql: MySqlConnectionConfig
    clickhouse: ClickHouseConnectionConfig
    s3: S3ConnectionConfig
    metrika: MetrikaConnectionConfig
    growthbook: GrowthBookConnectionConfig
    bitrix: BitrixConnectionConfig
    llm: LLMConnectionConfig
    langfuse: LangfuseConnectionConfig
    yandex_wiki: YandexWikiConnectionConfig | None = None
    redis: RedisConnectionConfig | None = None
