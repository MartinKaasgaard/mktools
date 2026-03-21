from __future__ import annotations

import logging
import os
import socket
import sys
import warnings
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Optional


class ExecutionEnvironment(str, Enum):
    COLAB = "colab"
    KAGGLE = "kaggle"
    SPARK_DGX = "spark_dgx"
    WINDOWS_LOCAL = "windows_local"
    LINUX_LOCAL = "linux_local"
    UNKNOWN = "unknown"


class RuntimeMode(IntEnum):
    PRODUCTION = 0
    DEVELOPMENT = 1
    VERBOSE = 9

    @classmethod
    def coerce(cls, value: int | str | "RuntimeMode" | None) -> "RuntimeMode":
        if value is None:
            return cls.DEVELOPMENT
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            alias_map = {
                "0": cls.PRODUCTION,
                "1": cls.DEVELOPMENT,
                "9": cls.VERBOSE,
                "prod": cls.PRODUCTION,
                "production": cls.PRODUCTION,
                "dev": cls.DEVELOPMENT,
                "development": cls.DEVELOPMENT,
                "debug": cls.VERBOSE,
                "verbose": cls.VERBOSE,
                "verbose_debugging": cls.VERBOSE,
            }
            if normalized in alias_map:
                return alias_map[normalized]
            try:
                return cls(int(normalized))
            except Exception as exc:
                raise ValueError(f"Unsupported runtime mode: {value!r}") from exc
        try:
            return cls(int(value))
        except Exception as exc:
            raise ValueError(f"Unsupported runtime mode: {value!r}") from exc


class AnalysisDepth(str, Enum):
    LIGHT = "light"
    STANDARD = "standard"
    DEEP = "deep"

    @classmethod
    def coerce(cls, value: str | "AnalysisDepth" | None) -> "AnalysisDepth":
        if value is None:
            return cls.STANDARD
        if isinstance(value, cls):
            return value
        normalized = str(value).strip().lower()
        alias_map = {
            "simple": cls.LIGHT,
            "light": cls.LIGHT,
            "lite": cls.LIGHT,
            "normal": cls.STANDARD,
            "standard": cls.STANDARD,
            "default": cls.STANDARD,
            "extensive": cls.DEEP,
            "deep": cls.DEEP,
            "full": cls.DEEP,
            "comprehensive": cls.DEEP,
        }
        if normalized not in alias_map:
            raise ValueError(
                f"Unsupported analysis depth {value!r}. "
                f"Supported values: {sorted(alias_map)}"
            )
        return alias_map[normalized]


def detect_exec_env() -> ExecutionEnvironment:
    override = os.getenv("EXEC_ENV")
    if override:
        try:
            return ExecutionEnvironment(override.strip().lower())
        except ValueError:
            pass

    try:
        import google.colab  # type: ignore  # noqa: F401
        return ExecutionEnvironment.COLAB
    except Exception:
        pass

    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ:
        return ExecutionEnvironment.KAGGLE

    host = ""
    try:
        host = socket.gethostname().lower()
    except Exception:
        host = ""

    if "spark" in host or "dgx" in host:
        return ExecutionEnvironment.SPARK_DGX
    if os.name == "nt":
        return ExecutionEnvironment.WINDOWS_LOCAL
    if os.name == "posix":
        return ExecutionEnvironment.LINUX_LOCAL
    return ExecutionEnvironment.UNKNOWN


def _build_logger(name: str, level: int) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s %(levelname)s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


@dataclass
class RuntimeConfig:
    mode: RuntimeMode = RuntimeMode.DEVELOPMENT
    exec_env: ExecutionEnvironment = field(default_factory=detect_exec_env)
    logger_name: str = "mktools.kstat"
    logger: Optional[logging.Logger] = None

    def __post_init__(self) -> None:
        self.mode = RuntimeMode.coerce(self.mode)
        self.exec_env = (
            self.exec_env
            if isinstance(self.exec_env, ExecutionEnvironment)
            else detect_exec_env()
        )
        self._apply_warning_policy()
        self.logger = self.logger or _build_logger(self.logger_name, self.log_level)

    @property
    def debug(self) -> bool:
        return self.mode >= RuntimeMode.DEVELOPMENT

    @property
    def verbose(self) -> bool:
        return self.mode >= RuntimeMode.VERBOSE

    @property
    def log_level(self) -> int:
        return {
            RuntimeMode.PRODUCTION: logging.INFO,
            RuntimeMode.DEVELOPMENT: logging.DEBUG,
            RuntimeMode.VERBOSE: logging.DEBUG,
        }[self.mode]

    @classmethod
    def from_env(
        cls,
        env_var: str = "RUNLEVEL",
        default: RuntimeMode = RuntimeMode.DEVELOPMENT,
        logger_name: str = "mktools.kstat",
    ) -> "RuntimeConfig":
        raw = os.getenv(env_var, str(int(default)))
        return cls(mode=RuntimeMode.coerce(raw), logger_name=logger_name)

    def _apply_warning_policy(self) -> None:
        warnings.resetwarnings()
        if self.mode == RuntimeMode.PRODUCTION:
            warnings.filterwarnings("ignore")
        elif self.mode == RuntimeMode.DEVELOPMENT:
            warnings.filterwarnings("default")
        else:
            warnings.filterwarnings("always")

    def log(self, message: str, *, level: int = logging.INFO) -> None:
        if self.logger is not None:
            self.logger.log(level, message)
