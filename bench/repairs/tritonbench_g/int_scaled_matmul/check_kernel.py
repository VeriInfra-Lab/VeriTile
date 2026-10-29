"""Read-only binding audit of retained current-source INT8 compiler evidence.

This does not compile, launch CUDA, or assert universal compiler conformance.
The existing source checker separately executes the canonical Python on CPU.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = HERE / "revision001/int_scaled_matmul_repaired.py"
SOURCE_SHA = "17bd051b849369e11f3c2b57440052a8c304722b32985499300ab572c553a33b"


def pin(path):
    path = Path(path).resolve()
    return {"path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


class BeforeExplicitAdd(ast.NodeTransformer):
    def visit_Assign(self, node):
        if (isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute)
                and node.value.func.attr == "inline_asm_elementwise"):
            args = next(k.value for k in node.value.keywords if k.arg == "args")
            return ast.copy_location(ast.AugAssign(target=node.targets[0], op=ast.Add(),
                                                   value=args.elts[1]), node)
        return self.generic_visit(node)


def mma_zero_seed(ptx, match):
    """Follow the explicit nonpredicated mov chain into every MMA C register."""
    operands = re.findall(r"\{([^}]+)\}", match.group())
    assert len(operands) == 4, match.group()
    prefix = ptx[:match.start()]
    assignments = {}
    for instruction in re.finditer(
            r"^\s*(\w+(?:\.\w+)+)\s+(%r\d+),\s*([^;]+);", prefix, re.M):
        op, dest, args = instruction.groups()
        assignments[dest] = (op, args.strip())

    def zero(reg, seen):
        assert reg not in seen, (reg, "cyclic seed")
        op, value = assignments[reg]
        assert op in ("mov.b32", "mov.u32", "mov.s32"), (reg, op, value)
        if value in ("0", "0x0", "0x0U"):
            return True
        assert re.fullmatch(r"%r\d+", value), value
        return zero(value, seen | {reg})

    registers = [r.strip() for r in operands[-1].split(",")]
    assert all(zero(r, set()) for r in registers)
    return registers


def run(artifacts, output):
    assert pin(SOURCE)["sha256"] == SOURCE_SHA
    source_tree = ast.parse(SOURCE.read_text())
    before_tree = ast.fix_missing_locations(BeforeExplicitAdd().visit(source_tree))
    before_path = artifacts / "before_explicit_add.py"
    assert ast.dump(before_tree, include_attributes=False) == ast.dump(
        ast.parse(before_path.read_text()), include_attributes=False), "retained source provenance changed"
    profiles = []
    for path in sorted(artifacts.glob("after_*.ttir")):
        name = path.stem
        scaled = "scaled1" in name
        ttir = path.read_text()
        ptx_path = path.with_suffix(".ptx")
        ptx = ptx_path.read_text()
        dots = [line for line in ttir.splitlines() if "= tt.dot " in line]
        assert dots and "tt.elementwise_inline_asm \"add.u32 $0, $1, $2;\"" in ttir
        for line in dots:
            seed = re.search(r"tt.dot %\w+, %\w+, (%\w+)", line).group(1)
            assert re.search(re.escape(seed) + r" = arith.constant dense<0> : tensor<\d+x\d+xi32>", ttir)
        mma = list(re.finditer(r"mma\.sync\.aligned\.m16n8k32\.row\.col\.satfinite\.s32\.s8\.s8\.s32[^;]+;", ptx))
        wgmma = re.findall(r"wgmma\.mma_async\.sync\.aligned\.m64n64k32\.s32\.s8\.s8[^;]+;", ptx)
        assert bool(mma) != bool(wgmma), name
        seeds = [mma_zero_seed(ptx, match) for match in mma]
        assert all(re.search(r",\s*0;$", instruction) for instruction in wgmma), name
        assert "add.u32" in ptx and ".address_size 64" in ptx
        promoted = set(re.findall(r"cvt\.rn\.f32\.s32\s+(%\w+),", ptx))
        multiplications = re.findall(r"mul\.f32\s+(%\w+),\s*(%\w+),\s*(%\w+);", ptx)
        conversions = dict((src, dst) for dst, src in re.findall(
            r"cvt\.rzi\.s32\.f32\s+(%\w+),\s*(%\w+);", ptx))
        if scaled:
            assert promoted and multiplications and "mul.ftz" not in ptx
            assert all(acc in promoted and result in conversions for result, scale, acc in multiplications)
        else:
            assert not promoted and not multiplications
        profiles.append({
            "name": name, "scaled": scaled,
            "family": "mma.sync.m16n8k32.satfinite" if mma else "wgmma.m64n64k32.scale_d_0",
            "dot_sites": len(mma) + len(wgmma), "ttir_dot_seed_zero": True,
            "mma_zero_seed_registers": seeds, "wgmma_scale_d_zero": bool(wgmma),
            "separate_wrapping_add": True, "fp32_multiplications": len(multiplications),
            "scaled_operand_order": "scale_then_promoted_accumulator" if scaled else None,
            "fp32_no_ftz": True if scaled else None,
            "product_to_toward_zero_int32": True if scaled else None,
            "artifacts": [pin(path.with_suffix("." + ext)) for ext in ("ttir", "llir", "ptx")],
        })
    assert len(profiles) == 6
    assert sum(p["fp32_multiplications"] for p in profiles) == 72
    receipt = {
        "task": "int_scaled_matmul", "exit_code": 0, "source": pin(SOURCE),
        "checker": pin(Path(__file__)), "retained_source_witness": pin(before_path),
        "mode": "read_only_audit_existing", "fresh_compilation": False, "GPU_execution": False,
        "profile_scope": "Triton 3.7.1 SM90; BM=BN=16 or 64; BK=32; GROUP_M=8; warps=4; stages=2; both APIs and scaled EVEN_K branches",
        "profiles": profiles,
        "boundary": "Source/IR/PTX structural audit plus selected primitive semantics; not universal compiler, register allocation, NaN payload or hardware conformance.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2) + "\n")
    print("6 retained current profiles / 18 artifacts / 72 scale-first FP32 multiplies passed; no compile or GPU run.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, default=Path("/tmp/veritile-int-scaled-lowering-audit"))
    parser.add_argument("--output", type=Path, default=ROOT / ".veritile-cache/task-deliveries/int_scaled_matmul/compiler-audit.json")
    options = parser.parse_args()
    run(options.artifacts, options.output)
