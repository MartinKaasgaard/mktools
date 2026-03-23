
from __future__ import annotations

import importlib
import json
import logging
import os
import random
import shutil
import socket
import subprocess
import sys
import tempfile
import uuid
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .runtime import AnalysisDepth, ExecutionEnvironment, RuntimeConfig, RuntimeMode, detect_exec_env


class FrameworkMode(str):
    NONE = "none"
    TENSORFLOW = "tensorflow"
    PYTORCH = "pytorch"
    BOTH = "both"
    AUTO = "auto"

    @classmethod
    def coerce(cls, value: str | "FrameworkMode" | None) -> str:
        if value is None:
            return cls.NONE
        normalized = str(value).strip().lower()
        allowed = {cls.NONE, cls.TENSORFLOW, cls.PYTORCH, cls.BOTH, cls.AUTO}
        if normalized not in allowed:
            raise ValueError(f"Unsupported framework mode: {value!r}")
        return normalized



class FilePolicy(str):
    DISABLED = "disabled"
    OPTIONAL = "optional"
    REQUIRED = "required"

    @classmethod
    def coerce(cls, value: str | "FilePolicy" | None) -> str:
        if value is None:
            return cls.OPTIONAL
        normalized = str(value).strip().lower()
        allowed = {cls.DISABLED, cls.OPTIONAL, cls.REQUIRED}
        if normalized not in allowed:
            raise ValueError(f"Unsupported file policy: {value!r}")
        return normalized


class SecretPolicy(str):
    DISABLED = "disabled"
    OPTIONAL = "optional"
    REQUIRED = "required"

    @classmethod
    def coerce(cls, value: str | "SecretPolicy" | None) -> str:
        if value is None:
            return cls.OPTIONAL
        normalized = str(value).strip().lower()
        allowed = {cls.DISABLED, cls.OPTIONAL, cls.REQUIRED}
        if normalized not in allowed:
            raise ValueError(f"Unsupported secret policy: {value!r}")
        return normalized


@dataclass(slots=True)
class BootstrapSettings:
    """
    High-level bootstrap settings for notebooks and scripts.

    Public-safe defaults:
    - no secrets are required by default
    - no personal paths are embedded
    - Colab persistent storage defaults to MyDrive/Projects
    - non-Colab persistent storage defaults to ~/projects
    """
    project_name: str
    runlevel: RuntimeMode | int | str = RuntimeMode.DEVELOPMENT
    analysis_depth: AnalysisDepth | str = AnalysisDepth.STANDARD
    framework_mode: str = FrameworkMode.NONE
    random_seed: int = 42

    # Environment / platform behavior
    force_exec_env: ExecutionEnvironment | str | None = None
    mount_drive: bool = True
    configure_inline_matplotlib: bool = True
    configure_seaborn_theme: bool = True

    # Paths
    persistent_root: str | Path | None = None
    workspace_root: str | Path | None = None
    common_code_root: str | Path | None = None
    kaggle_home: str | Path | None = None
    overlay_json_path: str | Path | None = None
    overlay_policy: str = FilePolicy.OPTIONAL

    # Optional package installation
    install_requirements: Sequence[str] = field(default_factory=tuple)
    pip_quiet: bool = True

    # Secrets / integrations
    configure_kaggle: bool = False
    kaggle_policy: str = SecretPolicy.OPTIONAL

    # Project structure
    create_project_dirs: bool = True
    strict_readonly_paths: bool = False
    extra_dirs: Mapping[str, str | Path] = field(default_factory=dict)

    # Environment variables written by bootstrap
    env_prefix: str = "MKTOOLS"

    def __post_init__(self) -> None:
        if not str(self.project_name).strip():
            raise ValueError("project_name must be a non-empty string")
        self.runlevel = RuntimeMode.coerce(self.runlevel)
        self.analysis_depth = AnalysisDepth.coerce(self.analysis_depth)
        self.framework_mode = FrameworkMode.coerce(self.framework_mode)
        self.kaggle_policy = SecretPolicy.coerce(self.kaggle_policy)
        self.overlay_policy = FilePolicy.coerce(self.overlay_policy)
        if not isinstance(self.random_seed, int):
            raise TypeError("random_seed must be an int")
        if self.random_seed < 0:
            raise ValueError("random_seed must be >= 0")


@dataclass(slots=True, frozen=True)
class CapabilityReport:
    in_ipython: bool
    in_notebook: bool
    colab_module_available: bool
    colab_userdata_available: bool
    drive_mount_available: bool
    kaggle_credentials_present: bool
    nvidia_smi_available: bool
    gpu_visible: bool
    tensorflow_importable: bool
    torch_importable: bool
    seaborn_importable: bool
    git_available: bool


@dataclass(slots=True, frozen=True)
class ProjectPaths:
    persistent_root: Path
    workspace_root: Path
    project_persistent: Path
    project_workspace: Path
    notebooks_dir: Path
    src_dir: Path
    data_raw: Path
    data_external: Path
    data_interim: Path
    data_processed: Path
    models_dir: Path
    models_ckpt: Path
    models_export: Path
    logs_dir: Path
    reports_dir: Path
    reports_figures: Path
    reports_tables: Path
    configs_dir: Path
    deps_dir: Path
    env_dir: Path
    cache_dir: Path
    tmp_dir: Path
    common_code_dir: Path | None
    kaggle_home: Path | None

    def as_dict(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for key, value in self.__dict__.items():
            out[key.upper()] = str(value) if value is not None else ""
        return out


@dataclass(slots=True)
class BootstrapResult:
    settings: BootstrapSettings
    runtime: RuntimeConfig
    exec_env: ExecutionEnvironment
    run_id: str
    started_at_utc: str
    capabilities: CapabilityReport
    paths: ProjectPaths
    env_vars_written: dict[str, str]
    packages_installed: list[str]
    package_failures: list[str]
    framework_notes: list[str]
    warnings_raised: list[str]

    @property
    def logger(self) -> logging.Logger:
        assert self.runtime.logger is not None
        return self.runtime.logger


def _utc_now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def _coerce_exec_env(value: ExecutionEnvironment | str | None) -> ExecutionEnvironment:
    if value is None:
        return detect_exec_env()
    if isinstance(value, ExecutionEnvironment):
        return value
    try:
        return ExecutionEnvironment(str(value).strip().lower())
    except Exception as exc:
        raise ValueError(f"Unsupported execution environment: {value!r}") from exc


def _in_ipython() -> bool:
    try:
        get_ipython  # type: ignore[name-defined]
        return True
    except Exception:
        return False


def _in_notebook() -> bool:
    if not _in_ipython():
        return False
    try:
        shell = get_ipython().__class__.__name__  # type: ignore[name-defined]
        return shell in {"ZMQInteractiveShell", "Shell"}
    except Exception:
        return False


def _module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except Exception:
        return False


def _gpu_visible() -> bool:
    if os.getenv("CUDA_VISIBLE_DEVICES") == "":
        return False
    if os.getenv("COLAB_GPU"):
        return True
    if shutil.which("nvidia-smi"):
        try:
            completed = subprocess.run(
                ["nvidia-smi", "-L"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
                timeout=5,
            )
            return completed.returncode == 0 and bool(completed.stdout.strip())
        except Exception:
            return True
    return False


def probe_capabilities() -> CapabilityReport:
    colab_available = _module_available("google.colab")
    userdata_available = False
    if colab_available:
        try:
            from google.colab import userdata  # type: ignore
            userdata_available = hasattr(userdata, "get")
        except Exception:
            userdata_available = False

    kaggle_home = (
        os.getenv("KAGHOME")
        or os.getenv("KAGGLE_HOME")
        or str(Path.home() / ".kaggle")
    )
    kaggle_json_candidates = [
        Path(kaggle_home) / "kaggle.json",
        Path.home() / ".kaggle" / "kaggle.json",
        Path.home() / ".config" / "kaggle" / "kaggle.json",
    ]
    kaggle_present = any(p.is_file() for p in kaggle_json_candidates)

    return CapabilityReport(
        in_ipython=_in_ipython(),
        in_notebook=_in_notebook(),
        colab_module_available=colab_available,
        colab_userdata_available=userdata_available,
        drive_mount_available=colab_available,
        kaggle_credentials_present=kaggle_present,
        nvidia_smi_available=shutil.which("nvidia-smi") is not None,
        gpu_visible=_gpu_visible(),
        tensorflow_importable=_module_available("tensorflow"),
        torch_importable=_module_available("torch"),
        seaborn_importable=_module_available("seaborn"),
        git_available=shutil.which("git") is not None,
    )


def _apply_early_env(runtime: RuntimeConfig) -> None:
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
    tf_cpp_map = {
        RuntimeMode.PRODUCTION: "3",
        RuntimeMode.DEVELOPMENT: "2",
        RuntimeMode.VERBOSE: "0",
    }
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", tf_cpp_map[runtime.mode])
    os.environ.setdefault("PYTHONHASHSEED", os.getenv("PYTHONHASHSEED", "42"))



def _normalize_pathlike(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    raw = os.path.expandvars(str(value))
    return Path(raw).expanduser()


def _path_looks_like_colab_drive(value: str | Path | None) -> bool:
    p = _normalize_pathlike(value)
    if p is None:
        return False
    s = p.as_posix()
    return s == "/content/drive" or s.startswith("/content/drive/")


def _load_overlay(
    path: str | Path | None,
    policy: str = FilePolicy.OPTIONAL,
) -> tuple[dict[str, Any], list[str]]:
    notes: list[str] = []
    if policy == FilePolicy.DISABLED or path is None:
        return {}, notes

    p = _normalize_pathlike(path)
    assert p is not None

    if not p.exists():
        msg = f"overlay_json_path not found: {p}"
        if policy == FilePolicy.REQUIRED:
            raise FileNotFoundError(msg)
        notes.append(msg)
        return {}, notes

    if p.suffix.lower() != ".json":
        msg = f"overlay_json_path must point to a JSON file: {p}"
        if policy == FilePolicy.REQUIRED:
            raise ValueError(msg)
        notes.append(msg)
        return {}, notes

    with p.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("overlay_json_path must contain a JSON object")
    return payload, notes


def _path_from_any(value: str | Path | None) -> Path | None:
    return _normalize_pathlike(value)


def _default_roots(
    settings: BootstrapSettings,
    exec_env: ExecutionEnvironment,
) -> tuple[Path, Path]:
    if settings.persistent_root is not None:
        persistent_root = Path(settings.persistent_root).expanduser()
    elif exec_env == ExecutionEnvironment.COLAB:
        persistent_root = Path("/content/drive/MyDrive/Projects")
    else:
        persistent_root = Path.home() / "projects"

    if settings.workspace_root is not None:
        workspace_root = Path(settings.workspace_root).expanduser()
    elif exec_env == ExecutionEnvironment.COLAB:
        workspace_root = Path("/content") / settings.project_name
    else:
        workspace_root = Path.cwd() / ".work" / settings.project_name

    return persistent_root, workspace_root


def _safe_mkdir(path: Path, mode: int = 0o750) -> Path:
    path = path.expanduser()
    path.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path, mode)
    except Exception:
        pass
    return path.resolve()



def _maybe_mount_drive_for_bootstrap(
    runtime: RuntimeConfig,
    exec_env: ExecutionEnvironment,
    settings: BootstrapSettings,
) -> None:
    if exec_env != ExecutionEnvironment.COLAB or not settings.mount_drive:
        return

    candidates = [
        settings.overlay_json_path,
        settings.persistent_root,
        settings.workspace_root,
        settings.common_code_root,
        settings.kaggle_home,
    ]
    should_mount = any(_path_looks_like_colab_drive(item) for item in candidates)

    # If caller explicitly wants drive mounting in Colab, mount even when no path
    # currently points inside /content/drive, so downstream code can rely on it.
    if should_mount or settings.mount_drive:
        _maybe_mount_drive(runtime, exec_env, mount_drive=True)


def _maybe_mount_drive(runtime: RuntimeConfig, exec_env: ExecutionEnvironment, mount_drive: bool) -> None:
    if exec_env != ExecutionEnvironment.COLAB or not mount_drive:
        return
    try:
        from google.colab import drive  # type: ignore
    except Exception as exc:
        raise RuntimeError("google.colab.drive could not be imported in Colab environment") from exc

    drive_root = Path("/content/drive")
    if drive_root.is_mount():
        runtime.logger.info("Google Drive already mounted.")
        return
    runtime.logger.info("Mounting Google Drive...")
    drive.mount(str(drive_root), force_remount=False)  # type: ignore[attr-defined]


def _compose_paths(
    settings: BootstrapSettings,
    exec_env: ExecutionEnvironment,
    overlay: Mapping[str, Any],
) -> ProjectPaths:
    persistent_root, workspace_root = _default_roots(settings, exec_env)

    persistent_root = _path_from_any(overlay.get("persistent_root")) or persistent_root
    workspace_root = _path_from_any(overlay.get("workspace_root")) or workspace_root
    common_code_root = (
        _path_from_any(overlay.get("common_code_root"))
        or _path_from_any(settings.common_code_root)
    )
    kaggle_home = (
        _path_from_any(overlay.get("kaggle_home"))
        or _path_from_any(settings.kaggle_home)
    )

    project_persistent = persistent_root / settings.project_name
    project_workspace = workspace_root

    return ProjectPaths(
        persistent_root=persistent_root,
        workspace_root=workspace_root,
        project_persistent=project_persistent,
        project_workspace=project_workspace,
        notebooks_dir=project_persistent / "notebooks",
        src_dir=project_persistent / "src",
        data_raw=project_persistent / "data" / "raw",
        data_external=project_persistent / "data" / "external",
        data_interim=project_persistent / "data" / "interim",
        data_processed=project_persistent / "data" / "processed",
        models_dir=project_persistent / "models",
        models_ckpt=project_persistent / "models" / "checkpoints",
        models_export=project_persistent / "models" / "exports",
        logs_dir=project_persistent / "logs",
        reports_dir=project_persistent / "reports",
        reports_figures=project_persistent / "reports" / "figures",
        reports_tables=project_persistent / "reports" / "tables",
        configs_dir=project_persistent / "configs",
        deps_dir=project_persistent / "deps",
        env_dir=project_persistent / "env",
        cache_dir=project_workspace / ".cache",
        tmp_dir=project_workspace / ".tmp",
        common_code_dir=common_code_root,
        kaggle_home=kaggle_home,
    )


def _prepare_dirs(
    paths: ProjectPaths,
    create_project_dirs: bool,
    strict_readonly_paths: bool,
    extra_dirs: Mapping[str, str | Path],
) -> None:
    if not create_project_dirs:
        return

    writeable = [
        paths.project_persistent,
        paths.project_workspace,
        paths.notebooks_dir,
        paths.src_dir,
        paths.data_raw,
        paths.data_external,
        paths.data_interim,
        paths.data_processed,
        paths.models_dir,
        paths.models_ckpt,
        paths.models_export,
        paths.logs_dir,
        paths.reports_dir,
        paths.reports_figures,
        paths.reports_tables,
        paths.configs_dir,
        paths.deps_dir,
        paths.env_dir,
        paths.cache_dir,
        paths.tmp_dir,
    ]
    for p in writeable:
        _safe_mkdir(p)

    readonly = [paths.common_code_dir, paths.kaggle_home]
    if strict_readonly_paths:
        for p in readonly:
            if p is not None and not p.is_dir():
                raise FileNotFoundError(f"Required readonly path does not exist: {p}")

    for name, p in extra_dirs.items():
        if not str(name).strip():
            raise ValueError("extra_dirs contains an empty key")
        _safe_mkdir(Path(p))


def _set_project_env(settings: BootstrapSettings, runtime: RuntimeConfig, exec_env: ExecutionEnvironment, paths: ProjectPaths) -> dict[str, str]:
    prefix = settings.env_prefix.strip().upper()
    env_map = {
        f"{prefix}_EXEC_ENV": exec_env.value,
        f"{prefix}_RUNLEVEL": str(int(runtime.mode)),
        f"{prefix}_ANALYSIS_DEPTH": settings.analysis_depth.value,
        f"{prefix}_PROJECT_NAME": settings.project_name,
        f"{prefix}_PROJECT_PERSISTENT": str(paths.project_persistent),
        f"{prefix}_PROJECT_WORKSPACE": str(paths.project_workspace),
        f"{prefix}_DATA_RAW": str(paths.data_raw),
        f"{prefix}_DATA_EXTERNAL": str(paths.data_external),
        f"{prefix}_DATA_INTERIM": str(paths.data_interim),
        f"{prefix}_DATA_PROCESSED": str(paths.data_processed),
        f"{prefix}_MODELS_DIR": str(paths.models_dir),
        f"{prefix}_LOGS_DIR": str(paths.logs_dir),
        f"{prefix}_REPORTS_DIR": str(paths.reports_dir),
    }
    if paths.common_code_dir is not None:
        env_map[f"{prefix}_COMMON_CODE_DIR"] = str(paths.common_code_dir)
    if paths.kaggle_home is not None:
        env_map["KAGHOME"] = str(paths.kaggle_home)
        env_map["KAGGLE_HOME"] = str(paths.kaggle_home)

    for key, value in env_map.items():
        os.environ[key] = value
    return env_map


def seed_everything(seed: int, framework_mode: str, runtime: RuntimeConfig) -> list[str]:
    notes: list[str] = []
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

    if framework_mode in {FrameworkMode.TENSORFLOW, FrameworkMode.BOTH, FrameworkMode.AUTO}:
        try:
            import tensorflow as tf  # type: ignore
            tf.random.set_seed(seed)
            notes.append("tensorflow seed set")
        except Exception as exc:
            if framework_mode == FrameworkMode.TENSORFLOW:
                raise RuntimeError("framework_mode='tensorflow' but TensorFlow could not be configured") from exc
            notes.append(f"tensorflow not configured: {exc}")

    if framework_mode in {FrameworkMode.PYTORCH, FrameworkMode.BOTH, FrameworkMode.AUTO}:
        try:
            import torch  # type: ignore
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            notes.append("torch seed set")
        except Exception as exc:
            if framework_mode == FrameworkMode.PYTORCH:
                raise RuntimeError("framework_mode='pytorch' but PyTorch could not be configured") from exc
            notes.append(f"torch not configured: {exc}")

    if runtime.verbose:
        runtime.logger.debug("Random seed configured: %s", seed)
    return notes


def configure_matplotlib(runtime: RuntimeConfig, inline: bool, seaborn_theme: bool) -> list[str]:
    notes: list[str] = []
    try:
        import matplotlib  # noqa: F401
    except Exception as exc:
        runtime.logger.warning("matplotlib not importable: %s", exc)
        return ["matplotlib not available"]

    if inline and _in_notebook():
        try:
            get_ipython().run_line_magic("matplotlib", "inline")  # type: ignore[name-defined]
            notes.append("matplotlib inline enabled")
        except Exception as exc:
            notes.append(f"matplotlib inline not enabled: {exc}")

    if seaborn_theme:
        try:
            import seaborn as sns  # type: ignore
            sns.set_theme(style="whitegrid", palette="muted", context="notebook")
            notes.append("seaborn theme applied")
        except Exception as exc:
            notes.append(f"seaborn theme not applied: {exc}")
    return notes


def ensure_python_packages(requirements: Sequence[str], quiet: bool = True, runtime: RuntimeConfig | None = None) -> tuple[list[str], list[str]]:
    installed: list[str] = []
    failed: list[str] = []
    if not requirements:
        return installed, failed

    for req in requirements:
        if not str(req).strip():
            failed.append(str(req))
            continue
        cmd = [sys.executable, "-m", "pip", "install"]
        if quiet:
            cmd.extend(["--quiet", "--disable-pip-version-check"])
        cmd.append(str(req))
        completed = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        if completed.returncode == 0:
            installed.append(str(req))
            if runtime and runtime.verbose:
                runtime.logger.debug("Installed requirement: %s", req)
        else:
            failed.append(str(req))
            if runtime:
                runtime.logger.error("Failed to install %s: %s", req, completed.stderr.strip())
    return installed, failed


def configure_kaggle(paths: ProjectPaths, policy: str, runtime: RuntimeConfig) -> None:
    if policy == SecretPolicy.DISABLED:
        return

    candidates = []
    if paths.kaggle_home is not None:
        candidates.append(paths.kaggle_home / "kaggle.json")
    candidates.extend([
        Path.home() / ".kaggle" / "kaggle.json",
        Path.home() / ".config" / "kaggle" / "kaggle.json",
    ])
    found = next((p for p in candidates if p.is_file()), None)

    if found is None:
        if policy == SecretPolicy.REQUIRED:
            raise FileNotFoundError(
                "Kaggle credentials required but kaggle.json was not found in expected locations"
            )
        runtime.logger.info("Kaggle credentials not found; continuing.")
        return

    with found.open("r", encoding="utf-8") as f:
        creds = json.load(f)
    if not isinstance(creds, dict) or not {"username", "key"} <= set(creds):
        raise ValueError("kaggle.json must contain 'username' and 'key'")

    os.environ["KAGGLE_USERNAME"] = str(creds["username"])
    os.environ["KAGGLE_KEY"] = str(creds["key"])

    target_paths = [
        Path.home() / ".kaggle" / "kaggle.json",
        Path.home() / ".config" / "kaggle" / "kaggle.json",
    ]
    for p in target_paths:
        _safe_mkdir(p.parent)
        p.write_text(json.dumps(creds), encoding="utf-8")
        os.chmod(p, 0o600)
    runtime.logger.info("Kaggle credentials configured from %s", found)


def configure_tensorflow(runtime: RuntimeConfig) -> list[str]:
    notes: list[str] = []
    try:
        import tensorflow as tf  # type: ignore
    except Exception as exc:
        raise RuntimeError("TensorFlow requested but tensorflow is not importable") from exc

    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as exc:
            notes.append(f"memory growth failed for {gpu}: {exc}")

    if gpus:
        try:
            from tensorflow.keras import mixed_precision  # type: ignore
            mixed_precision.set_global_policy("mixed_float16")
            notes.append("tensorflow mixed precision enabled")
        except Exception as exc:
            notes.append(f"tensorflow mixed precision not enabled: {exc}")
    return notes



def bootstrap_environment(settings: BootstrapSettings) -> BootstrapResult:
    exec_env = _coerce_exec_env(settings.force_exec_env)
    runtime = RuntimeConfig(mode=settings.runlevel, exec_env=exec_env, logger_name="mktools.bootstrap")
    _apply_early_env(runtime)

    warnings_raised: list[str] = []

    # In Colab, mount Drive before attempting to read overlays or user-specified
    # roots that may live under /content/drive.
    _maybe_mount_drive_for_bootstrap(runtime, exec_env, settings)

    overlay, overlay_notes = _load_overlay(
        settings.overlay_json_path,
        policy=settings.overlay_policy,
    )
    warnings_raised.extend(overlay_notes)
    for note in overlay_notes:
        runtime.logger.info("%s", note)

    capabilities = probe_capabilities()

    installed, failures = ensure_python_packages(
        requirements=settings.install_requirements,
        quiet=settings.pip_quiet,
        runtime=runtime,
    )

    paths = _compose_paths(settings, exec_env, overlay)
    _prepare_dirs(
        paths=paths,
        create_project_dirs=settings.create_project_dirs,
        strict_readonly_paths=settings.strict_readonly_paths,
        extra_dirs=settings.extra_dirs,
    )
    env_vars = _set_project_env(settings, runtime, exec_env, paths)

    framework_notes = seed_everything(settings.random_seed, settings.framework_mode, runtime)
    framework_notes.extend(
        configure_matplotlib(
            runtime=runtime,
            inline=settings.configure_inline_matplotlib,
            seaborn_theme=settings.configure_seaborn_theme,
        )
    )

    if settings.configure_kaggle:
        try:
            configure_kaggle(paths=paths, policy=settings.kaggle_policy, runtime=runtime)
        except Exception as exc:
            if settings.kaggle_policy == SecretPolicy.REQUIRED:
                raise
            warnings_raised.append(str(exc))
            runtime.logger.warning("%s", exc)

    if settings.framework_mode in {FrameworkMode.TENSORFLOW, FrameworkMode.BOTH, FrameworkMode.AUTO}:
        try:
            framework_notes.extend(configure_tensorflow(runtime))
        except Exception as exc:
            if settings.framework_mode == FrameworkMode.TENSORFLOW:
                raise
            warnings_raised.append(str(exc))
            runtime.logger.info("%s", exc)

    result = BootstrapResult(
        settings=settings,
        runtime=runtime,
        exec_env=exec_env,
        run_id=uuid.uuid4().hex[:8],
        started_at_utc=_utc_now_iso(),
        capabilities=capabilities,
        paths=paths,
        env_vars_written=env_vars,
        packages_installed=installed,
        package_failures=failures,
        framework_notes=framework_notes,
        warnings_raised=warnings_raised,
    )

    runtime.logger.info(
        "Bootstrap complete | env=%s | runlevel=%s | project=%s",
        exec_env.value,
        runtime.mode.name,
        settings.project_name,
    )
    return result


def print_bootstrap_summary(result: BootstrapResult) -> None:
    print("\n=== mktools bootstrap summary =====================")
    print(f"Project        : {result.settings.project_name}")
    print(f"Run ID         : {result.run_id}")
    print(f"Started (UTC)  : {result.started_at_utc}")
    print(f"Exec env       : {result.exec_env.value}")
    print(f"Runlevel       : {result.runtime.mode.name}")
    print(f"Depth          : {result.settings.analysis_depth.value}")
    print(f"Framework mode : {result.settings.framework_mode}")
    print(f"GPU visible    : {result.capabilities.gpu_visible}")
    print(f"Persistent root: {result.paths.persistent_root}")
    print(f"Workspace root : {result.paths.project_workspace}")
    print(f"Data raw       : {result.paths.data_raw}")
    print(f"Reports dir    : {result.paths.reports_dir}")
    if result.packages_installed:
        print(f"Installed pkgs : {', '.join(result.packages_installed)}")
    if result.package_failures:
        print(f"Pkg failures   : {', '.join(result.package_failures)}")
    if result.framework_notes:
        print("Notes          :")
        for note in result.framework_notes:
            print(f"  - {note}")
    if result.warnings_raised:
        print("Warnings       :")
        for item in result.warnings_raised:
            print(f"  - {item}")
    print("===================================================\n")


__all__ = [
    "BootstrapSettings",
    "BootstrapResult",
    "CapabilityReport",
    "FilePolicy",
    "FrameworkMode",
    "ProjectPaths",
    "SecretPolicy",
    "bootstrap_environment",
    "configure_kaggle",
    "configure_matplotlib",
    "configure_tensorflow",
    "ensure_python_packages",
    "print_bootstrap_summary",
    "probe_capabilities",
    "seed_everything",
]
