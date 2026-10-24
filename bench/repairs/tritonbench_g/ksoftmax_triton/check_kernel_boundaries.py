"""New exceptional/masked source controls; CPU interpretation, never GPU evidence."""
import hashlib
import importlib.util
import json
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "ksoftmax_triton_repaired.py"
PIN = "03d1517e3ac3ef0fdab20673a410d248d11461c1ddbc12df9e262119a98bd1e4"


def main():
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == PIN
    spec = importlib.util.spec_from_file_location("ksoftmax_checked_cpu", HERE / "check_contract.py")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    source, launches = helper.load(SOURCE)
    records = []
    for mode in (None, "qk", "bk"):
        for log in (False, True):
            for causal in (False, True):
                for kind in ("partial_negative_infinity", "all_negative_infinity", "positive_infinity", "nan"):
                    x = helper.tensor([[[0., 1., 2.], [0., 1., 2.]]])
                    m = None if mode is None else helper.tensor(torch.zeros((2 if mode == "qk" else 1, 3)))
                    target = x.value if mode is None else m.value
                    if kind == "partial_negative_infinity":
                        target[..., 1:] = -float("inf")
                    elif kind == "all_negative_infinity":
                        target[...] = -float("inf")
                    elif kind == "positive_infinity":
                        target[..., 0] = float("inf")
                    else:
                        target[..., 0] = float("nan")
                    y = helper.tensor(torch.full_like(x.value, 37.))
                    expected = helper.forward_reference(x, m, mode, log, causal)
                    old = len(launches)
                    assert source["softmax"](y, x, m, log, mode, causal) is None
                    assert len(launches) == old + 1
                    torch.testing.assert_close(y.value, expected, equal_nan=True)
                    if kind == "partial_negative_infinity":
                        assert not torch.isnan(y.value).any()
                        assert torch.isneginf(y.value[..., 1:]).all() if log else (y.value[..., 1:] == 0).all()
                    else:
                        assert torch.isnan(y.value).all()
                    records.append({"entry": "forward", "mode": mode, "log": log, "causal": causal,
                                    "case": kind, "matches_independent_torch_reference": True})
                if causal:
                    x = helper.tensor([[[0., float("nan"), float("inf")], [0., 1., float("nan")]]])
                    m = None if mode is None else helper.tensor(torch.zeros((2 if mode == "qk" else 1, 3)))
                    y = helper.tensor(torch.full_like(x.value, 37.))
                    source["softmax"](y, x, m, log, mode, causal)
                    expected = helper.forward_reference(x, m, mode, log, causal)
                    torch.testing.assert_close(y.value, expected)
                    assert not torch.isnan(y.value).any()
                    records.append({"entry": "forward", "mode": mode, "log": log, "causal": causal,
                                    "case": "unread_future_nan_and_infinity_do_not_propagate"})
    for log in (False, True):
        for causal in (False, True):
            g = helper.tensor([[[2., float("nan"), float("inf")]]])
            o = helper.tensor([[[0. if log else 1., -float("inf") if log else 0., float("nan")]]])
            d = helper.tensor(torch.full_like(g.value, 37.))
            source["softmax_backward"](d, g, o, log, causal)
            expected = helper.backward_reference(g, o, log, causal)
            torch.testing.assert_close(d.value, expected, equal_nan=True)
            if causal:
                assert (d.value == 0).all()
            else:
                assert torch.isnan(d.value).all()
            records.append({"entry": "backward", "log": log, "causal": causal,
                            "case": "exceptional_saved_and_gradient_words"})
    print(json.dumps({"source_sha256": PIN, "cases": len(records), "status": "passed",
                      "scope": "Actual canonical source interpreted on CPU; CUDA metadata/launch mocked; no GPU numerical conformance claim",
                      "records": records}, indent=2))


if __name__ == "__main__":
    main()
