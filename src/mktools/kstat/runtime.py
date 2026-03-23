"""Backward-compatible re-exports. Prefer mktools.common.runtime."""
from mktools.common.runtime import AnalysisDepth, ExecutionEnvironment, RuntimeConfig, RuntimeMode, detect_exec_env

__all__ = ["AnalysisDepth", "ExecutionEnvironment", "RuntimeConfig", "RuntimeMode", "detect_exec_env"]
