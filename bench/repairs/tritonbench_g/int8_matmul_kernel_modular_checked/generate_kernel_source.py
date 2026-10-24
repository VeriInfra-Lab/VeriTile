"""Exact AST-driven readable quotation for the separate INT32 word-add candidate.

Only frontend syntax is changed: Lean antiquotations bind source parameters,
tuple shape lists use the existing DSL list spelling, Python loop indentation
becomes braces, and Lean's reserved identifier partial is escaped without a name
change. The inline-assembly instruction is checked exactly, then represented by its
strict unsigned32/signed32 word-add projection in the ordinary quotation. The
actual Python source retains its inline-assembly compiler barrier unchanged.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = HERE / "int8_matmul_kernel_modular_checked.py"
OUTPUT = HERE / "Kernels.lean"
SOURCE_SHA = "5fb5ed4e23f10e9e65aba367755153d8562ef7812ca8c64ee753ca9569af5059"
POINTERS = {"a_ptr": ".fp32", "b_ptr": ".nat", "c_ptr": ".int"}
SCALARS = {"M", "N", "K", "stride_am", "stride_ak", "stride_bk", "stride_bn",
           "stride_cm", "stride_cn", "BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "GROUP_SIZE_M"}
BINARY = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.FloorDiv: "//",
          ast.Mod: "%", ast.LShift: "<<", ast.RShift: ">>", ast.BitAnd: "&"}
COMPARE = {ast.Eq: "==", ast.Lt: "<"}


def name(value):
    return "«partial»" if value == "partial" else value


def render(node, literal=False):
    if isinstance(node, ast.Name):
        if node.id in POINTERS:
            return f"$(({node.id} : Region {POINTERS[node.id]}))"
        return f"$({node.id})" if node.id in SCALARS else name(node.id)
    if isinstance(node, ast.Constant):
        if node.value is None:
            return "None"
        if isinstance(node.value, bool):
            return "True" if node.value else "False"
        if isinstance(node.value, str):
            return json.dumps(node.value)
        assert type(node.value) is int and node.value >= 0
        return str(node.value) if literal else f"$({node.value})"
    if isinstance(node, ast.Attribute):
        value = render(node.value)
        if not isinstance(node.value, (ast.Name, ast.Attribute)):
            value = f"({value})"
        return value + "." + node.attr
    if isinstance(node, ast.BinOp):
        return f"({render(node.left)} {BINARY[type(node.op)]} {render(node.right)})"
    if isinstance(node, ast.Compare):
        assert len(node.ops) == len(node.comparators) == 1
        return f"({render(node.left)} {COMPARE[type(node.ops[0])]} {render(node.comparators[0])})"
    if isinstance(node, ast.Subscript):
        indices = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
        parts = []
        for item in indices:
            if isinstance(item, ast.Slice):
                assert item.lower is item.upper is item.step is None
                parts.append(":")
            else:
                parts.append(render(item))
        return render(node.value) + "[" + ", ".join(parts) + "]"
    if isinstance(node, (ast.Tuple, ast.List)):
        return "[" + ", ".join(render(item, literal=True) for item in node.elts) + "]"
    if isinstance(node, ast.Call):
        function = render(node.func)
        is_asm = function == "tl.inline_asm_elementwise"
        if is_asm:
            expected = ast.parse('tl.inline_asm_elementwise("add.u32 $0, $1, $2;", constraints="=r,r,r", args=[accumulator, partial], dtype=tl.int32, is_pure=True, pack=1)', mode='eval').body
            assert ast.dump(node, include_attributes=False) == ast.dump(expected, include_attributes=False)
            return "((accumulator + «partial»).to(tl.uint32)).to(tl.int32)"
        args = [render(value, literal=((function == "tl.full" and i == 1) or
                                      (function == "tl.arange" and i == 0)))
                for i, value in enumerate(node.args)]
        for keyword in node.keywords:
            assert keyword.arg is not None
            args.append(keyword.arg + "=" + render(keyword.value, literal=keyword.arg in {"axis", "other", "pack"}))
        return function + "(" + ", ".join(args) + ("," if is_asm else "") + ")"
    raise AssertionError(ast.dump(node))


def statement(node, indent=2):
    prefix = " " * indent
    if isinstance(node, ast.Assign):
        assert len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
        result = [prefix + name(node.targets[0].id) + " = " + render(node.value)]
        if isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == 'tl.inline_asm_elementwise':
            result.insert(0, prefix + '-- Exact source add.u32 ASM ABI above; strict word-add projection, not a replacement Python expression.')
        return result
    if isinstance(node, ast.AugAssign):
        assert isinstance(node.target, ast.Name) and isinstance(node.op, ast.Add)
        return [prefix + name(node.target.id) + " += " + render(node.value)]
    if isinstance(node, ast.Expr):
        return [prefix + render(node.value)]
    if isinstance(node, ast.For):
        assert isinstance(node.target, ast.Name) and not node.orelse
        assert isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name) and node.iter.func.id == "range"
        result = [prefix + "for " + name(node.target.id) + " in " + render(node.iter) + " {"]
        for child in node.body:
            result.extend(statement(child, indent+2))
        result.append(prefix + "}")
        return result
    raise AssertionError(ast.dump(node))


def generate():
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SOURCE_SHA
    tree = ast.parse(SOURCE.read_text())
    kernel = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "matmul_kernel")
    assert {arg.arg for arg in kernel.args.args} == set(POINTERS) | SCALARS
    assertions = [node for node in ast.walk(kernel) if isinstance(node, ast.Call)
                  and isinstance(node.func, ast.Attribute)
                  and isinstance(node.func.value, ast.Name)
                  and node.func.value.id == "tl" and node.func.attr == "static_assert"]
    assert len(assertions) == 1 and isinstance(kernel.body[0], ast.Expr)
    assert kernel.body[0].value is assertions[0]
    predicate = ast.parse("K % (4 * BLOCK_SIZE_K) == 0", mode="eval").body
    assert ast.dump(assertions[0].args[0]) == ast.dump(predicate)
    body = []
    for node in kernel.body:
        body.extend(statement(node))
    return '''import VeriTile.Triton

namespace VeriTile.Bench.Repairs.Int8MatmulKernelModularChecked
open VeriTile.Triton
noncomputable section

/- Generated from the separately pinned Python kernel by generate_kernel_source.py.
The FP32 input cast, unsigned packed load, exact assembly ABI, source masks,
pointer updates, loop bounds and INT32 output remain in this readable source.
The sole primitive projection below represents the original exact ASM:
  tl.inline_asm_elementwise("add.u32 $0, $1, $2;", constraints="=r,r,r",
    args=[accumulator, partial], dtype=tl.int32, is_pure=True, pack=1).
It uses the existing strict unsigned32/signed32 word-add semantics; the
real Python program retains this ASM barrier. SourcePreservation proves the
entire quotation unchanged from the prior bounded primitive expansion. -/
def kernel (a_ptr : Region .fp32) (b_ptr : Region .nat) (c_ptr : Region .int)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K GROUP_SIZE_M : Nat) : ComputeKernel := triton {
  tl.region a_ptr = tl.float32, b_ptr = tl.uint8, c_ptr = tl.int32
''' + "\n".join(body) + '''
}

end
end VeriTile.Bench.Repairs.Int8MatmulKernelModularChecked
'''


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--update", action="store_true")
    args = parser.parse_args()
    expected = generate()
    if args.check:
        assert OUTPUT.read_text() == expected, "Readable kernel quotation differs from the pinned source AST generator"
    else:
        assert args.update or not OUTPUT.exists(), "Preserve existing quotation; use --check or explicit --update"
        OUTPUT.write_text(expected)
    print(json.dumps({"source": str(SOURCE.relative_to(ROOT)), "source_sha256": SOURCE_SHA,
                      "quotation": str(OUTPUT.relative_to(ROOT)),
                      "quotation_sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
                      "unique_leading_static_assert": "K % (4 * BLOCK_SIZE_K) == 0",
                      "generated_from_exact_source_AST": True}))
