import asyncio
import shutil
from pathlib import Path

import anyio
import pytest

try:
    from mcp_agent_mail.config import clear_settings_cache  # type: ignore[import]
except ModuleNotFoundError:
    def clear_settings_cache() -> None:
        pass

try:
    from mcp_agent_mail.db import reset_database_state  # type: ignore[import]
except ModuleNotFoundError:
    def reset_database_state() -> None:
        pass


@pytest.fixture
def isolated_env(tmp_path, monkeypatch):
    """Provide isolated database settings for tests and reset caches."""
    db_path: Path = tmp_path / "test.sqlite3"
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{db_path}")
    monkeypatch.setenv("HTTP_HOST", "127.0.0.1")
    monkeypatch.setenv("HTTP_PORT", "8765")
    monkeypatch.setenv("HTTP_PATH", "/mcp/")
    monkeypatch.setenv("APP_ENVIRONMENT", "test")
    storage_root = tmp_path / "storage"
    storage_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("STORAGE_ROOT", str(storage_root))
    monkeypatch.setenv("GIT_AUTHOR_NAME", "test-agent")
    monkeypatch.setenv("GIT_AUTHOR_EMAIL", "test@example.com")
    monkeypatch.setenv("INLINE_IMAGE_MAX_BYTES", "128")
    monkeypatch.setenv("LLM_ENABLED", "false")
    monkeypatch.setenv("LLM_DEFAULT_MODEL", "stub-model")
    monkeypatch.setenv("LLM_TEMPERATURE", "0")
    monkeypatch.setenv("LLM_MAX_TOKENS", "0")
    try:
        import litellm  # type: ignore[import]

        monkeypatch.setattr(
            litellm,
            "completion",
            lambda *args, **kwargs: {"choices": [{"message": {"content": ""}}]},
            raising=False,
        )
        monkeypatch.setattr(litellm, "success_callback", [], raising=False)
    except ModuleNotFoundError:
        pass
    clear_settings_cache()
    reset_database_state()
    try:
        yield
    finally:
        clear_settings_cache()
        reset_database_state()

        async def _remove_with_retry(target: Path) -> None:
            """Windows 上で PermissionError が発生しがちな削除をリトライ付きで行う。"""
            for _attempt in range(30):
                try:
                    await asyncio.to_thread(target.unlink)
                    return
                except FileNotFoundError:
                    return
                except PermissionError:
                    await anyio.sleep(0.1)

        async def _cleanup() -> None:
            if db_path.exists():
                await _remove_with_retry(db_path)
            if storage_root.exists():
                await asyncio.to_thread(
                    shutil.rmtree, storage_root, ignore_errors=True
                )

        anyio.run(_cleanup)
