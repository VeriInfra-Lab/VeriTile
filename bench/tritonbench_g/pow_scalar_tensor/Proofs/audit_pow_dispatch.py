"""Replay the pinned Python extern dispatch without importing Triton or a GPU.

Pass the exact pinned core.py and cuda/libdevice.py snapshots. This executes
the extracted production extern_elementwise function with minimal tensor and
semantic stubs. It checks only overload selection, not lowering or execution.
"""

import argparse
import ast
import builtins
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace


class Tensor:
    def __init__(self, dtype):
        self.dtype = dtype
        self.type = self

    def is_block(self):
        return True

    def with_element_ty(self, dtype):
        return dtype


class Semantic:
    builder = SimpleNamespace(create_extern_elementwise=None)

    def to_tensor(self, value):
        return value

    def binary_op_type_checking_impl(self, lhs, rhs, arithmetic_check):
        # Every successful source profile already has an exact extern key;
        # unsupported profiles must fail before reaching this method.
        assert arithmetic_check is False
        return lhs, rhs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--libdevice", type=Path, required=True)
    args = parser.parse_args()
    core_text = args.core.read_text()
    lib_text = args.libdevice.read_text()
    core_tree = ast.parse(core_text)
    lib_tree = ast.parse(lib_text)
    extern = next(n for n in core_tree.body
                  if isinstance(n, ast.FunctionDef)
                  and n.name == "extern_elementwise")
    pow_fn = next(n for n in lib_tree.body
                  if isinstance(n, ast.FunctionDef) and n.name == "pow")
    signatures = next(n for n in ast.walk(pow_fn) if isinstance(n, ast.Dict))
    table = {}
    for key, value in zip(signatures.keys, signatures.values):
        dtypes = tuple(ast.literal_eval(x.args[0]) for x in key.elts)
        symbol = ast.literal_eval(value.elts[0])
        result = ast.literal_eval(value.elts[1].args[0])
        table[dtypes] = symbol, result

    extern.decorator_list = []
    module = ast.Module(body=[extern], type_ignores=[])
    namespace = {
        "builtins": builtins,
        "dispatch": lambda _func, _lib, _path, values, symbols, result,
                           _pure, _semantic:
            {"symbol": symbols[tuple(v.dtype for v in values)][0],
             "result": result},
    }
    exec(compile(module, str(args.core), "exec"), namespace)
    outcomes = {}
    profiles = ["fp32", "int32", "fp64", "fp16", "bf16", "fp8e4nv",
                "fp8e5", "int1", "int8", "int16", "int64", "uint8",
                "uint16", "uint32", "uint64"]
    for exponent in profiles:
        try:
            result = namespace["extern_elementwise"](
                "", "", [Tensor("fp32"), Tensor(exponent)], table,
                is_pure=True, _semantic=Semantic())
            outcomes[exponent] = {"dispatch": "accepted", **result}
        except KeyError as error:
            outcomes[exponent] = {"dispatch": "KeyError", "key": list(error.args[0])}
    accepted = [p for p, result in outcomes.items()
                if result["dispatch"] == "accepted"]
    assert accepted == ["fp32", "int32"], accepted
    assert outcomes["fp32"]["symbol"] == "__nv_powf"
    assert outcomes["int32"]["symbol"] == "__nv_powif"
    receipt = {
        "scope": "production extern overload dispatch only; no GPU or lowering claim",
        "core_sha256": hashlib.sha256(core_text.encode()).hexdigest(),
        "libdevice_sha256": hashlib.sha256(lib_text.encode()).hexdigest(),
        "source_base_after_explicit_cast": "fp32",
        "outcomes": outcomes,
    }
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
