import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="Torch 前端为 M0 可选依赖")

from nest.frontend import from_torch  # noqa: E402
from nest.runtime import run_module  # noqa: E402


def test_torch_mlp_imports_and_runs():
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 3),
    ).eval()
    args = (torch.randn(1, 4),)

    result = from_torch(
        model,
        args,
        input_names=["x"],
        output_names=["out"],
        module_name="torch_mlp",
    )
    assert result.module.name == "torch_mlp"
    x = args[0].detach().numpy().astype(np.float32)
    with torch.no_grad():
        expected = model(args[0]).numpy()
    actual = run_module(result.module, {"x": x})["out"]
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
