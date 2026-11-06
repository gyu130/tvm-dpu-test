import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from nest.registry import GLOBAL_REGISTRY  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_registry():
    """每个测试结束后恢复全局注册表，避免自定义算子污染其他用例。"""
    snapshot = GLOBAL_REGISTRY.snapshot()
    yield
    GLOBAL_REGISTRY.restore(snapshot)
