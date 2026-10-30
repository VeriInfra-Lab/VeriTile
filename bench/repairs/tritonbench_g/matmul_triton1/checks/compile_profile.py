"""Two exact-source default SM90 ABI compiles; no candidate or GPU execution."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys

TASK = Path(__file__).resolve().parents[1]
SOURCE = TASK / "revision001/matmul_triton1_repaired.py"
SOURCE_SHA = "705bfb2cecafd2644cfe60922a0eebce2e110d0221343c7e87b273c14bbfe62a"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def trace(ptx):
    memory, chains, fmas = {}, {}, []
    for line in ptx.splitlines():
        load = re.search(r"ld\.shared(?:\.v([24]))?\.b32\s+(\{[^}]+\}|%r\d+),\s*\[(%r\d+)(?:\+(\d+))?\]", line)
        if load:
            for j, register in enumerate(re.findall(r"%r\d+", load[2])):
                memory[register] = (load[3], int(load[4] or 0) + j * 4)
        fma = re.search(r"fma\.rn\.f32\s+(%r\d+),\s*(%r\d+),\s*(%r\d+),\s*(%r\d+)", line)
        if fma:
            dst, a, b, seed = fma.groups()
            assert a in memory and b in memory, (a, b)
            prior = chains.get(seed, {"seed": seed, "terms": []})
            chains[dst] = {"seed": prior["seed"], "terms": prior["terms"] + [(memory[a], memory[b])]}
            fmas.append(fma.groups())
    roots = sorted({chain["seed"] for chain in chains.values()})
    assert len(roots) == 8 and len(fmas) == 256
    output = []
    for root in roots:
        chain = chains[root]
        assert chain["seed"] == root and len(chain["terms"]) == 32
        (abase, a0), (bbase, b0) = chain["terms"][0]
        for k, (a, b) in enumerate(chain["terms"]):
            assert a == (abase, a0 + 4*k) and b == (bbase, b0 + 128*k), (root, k, a, b)
        output.append({"root": root, "terms": 32, "k_order": list(range(32)),
                       "live_carry": True, "a_byte_step": 4, "b_byte_step": 128})
    assert not re.search(r"mma\.|wgmma|\.tf32|fma\.rn\.ftz\.f32", ptx)
    return output


def main():
    target = Path(sys.argv[1]).resolve()
    target.mkdir(parents=True, exist_ok=False)
    assert sha(SOURCE) == SOURCE_SHA
    os.environ["TRITON_CACHE_DIR"] = str(target / "compiler-cache")
    import triton
    from triton.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
    spec = importlib.util.spec_from_file_location("matmul_triton1_selected_profile", SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    kernel = module.matmul_kernel
    constants = {"m_block_size": 32, "k_block_size": 32, "n_block_size": 32}
    rows = []
    for abi in ["i32", "i64"]:
        signature = {name: ("*fp32" if name.endswith("_ptr") else abi)
                     for name in kernel.arg_names if name not in constants}
        # Omit options exactly as the selected Python launch does.
        compiled = triton.compile(ASTSource(kernel, signature=signature, constexprs=constants),
                                  target=GPUTarget("cuda", 90, 32))
        assert compiled.metadata.num_warps == 4 and compiled.metadata.num_stages == 3
        folder = target / abi
        folder.mkdir()
        pins = {}
        for ext in ["ttir", "ttgir", "llir", "ptx"]:
            path = folder / ("kernel." + ext)
            path.write_text(compiled.asm[ext])
            pins[str(path)] = sha(path)
        dots = [line.strip() for line in compiled.asm["ttir"].splitlines() if "tt.dot" in line]
        assert len(dots) == 1 and "tf32" not in dots[0]
        chains = trace(compiled.asm["ptx"])
        row = {"abi": abi, "signature": signature, "num_warps": 4, "num_stages": 3,
               "options_omitted_like_python_launch": True, "dot": dots,
               "fma_chains": chains, "artifact_sha256": pins}
        rows.append(row)
        print(json.dumps({"abi": abi, "status": "pass", "fma_chains": len(chains),
                          "terms_per_chain": 32, "num_stages": 3}), flush=True)
    assert sha(SOURCE) == SOURCE_SHA
    result = {"status": "pass", "source": str(SOURCE), "source_sha256": SOURCE_SHA,
              "triton_version": triton.__version__, "target": "SM90", "constants": constants,
              "rows": rows, "gpu_execution": False, "numerical_experiment": False,
              "native_instruction_conformance_proved": False,
              "boundary": "Compiler lane/address/synchronization mapping remains local; only the selected scalar zero/FMA numerical ports occur in the Lean conditional theorem."}
    (target / "receipt.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
