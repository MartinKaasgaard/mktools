
from __future__ import annotations

from mktools.common.bootstrap import BootstrapSettings, FrameworkMode, bootstrap_environment


def test_bootstrap_smoke(tmp_path):
    settings = BootstrapSettings(
        project_name="demo",
        framework_mode=FrameworkMode.NONE,
        persistent_root=tmp_path / "persistent",
        workspace_root=tmp_path / "workspace",
        create_project_dirs=True,
        configure_inline_matplotlib=False,
        configure_seaborn_theme=False,
    )
    result = bootstrap_environment(settings)
    assert result.paths.project_persistent.is_dir()
    assert result.paths.project_workspace.is_dir()
    assert result.env_vars_written["MKTOOLS_PROJECT_NAME"] == "demo"
