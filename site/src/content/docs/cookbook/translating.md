---
title: Translating a kernel
description: Paste the original Triton function body, bind its inputs outside the block, and keep source fidelity separate from proof scope.
---

The source entry is `Kernels.lean`, not the proof file. Compare
[VectorAdd.py](https://github.com/VeriInfra-Lab/VeriTile/blob/main/bench/examples/VectorAdd/VectorAdd.py)
with [Kernels.lean](https://github.com/VeriInfra-Lab/VeriTile/blob/main/bench/examples/VectorAdd/Kernels.lean).
The content of `triton { ... }` preserves the original Python function body:
statements, comments, casts, indexes, masks, branches, helper calls and order.
Input and invocation declarations belong outside the block.

## Bind the inputs, then paste the body

A directly instantiated kernel can use the actual input signature:

```lean
def addKernel (xReg yReg outReg : RegionName) (blockSize : Nat) : ComputeKernel :=
  let x_ptr := xReg
  let y_ptr := yReg
  let out_ptr := outReg
  let BLOCK_SIZE := blockSize
  with_triton_signature callSignature in triton {
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets)
    y = tl.load(y_ptr + offsets)
    output = x + y
    tl.store(out_ptr + offsets, output)
  }
```

`callSignature` in the linked file declares the actual fp32 pointer storage and
constexpr parameter. A function body alone cannot determine the dtype of an
unannotated Python pointer. This information comes from the input/call interface.
Do not insert `tl.region`, extra casts or Lean parameter antiquotes into the body.

For a reusable function, use `def nameSource : SourceProcedure :=
with_triton_params (...) constexpr_args [...] in triton { ... }` in the task's
namespace. Declare parameters in Python order. Add external `fp32_constants`
only when symbolic FP constexpr values need that binding. A standalone
`nameKernel : ComputeKernel` uses `instantiate_source_procedure`; another Triton
body binds the same source with external `source_procedures` and retains the
Python call arguments. See the reusable
[rms_matmul_rbe source](https://github.com/VeriInfra-Lab/VeriTileBench/blob/main/bench/tritonbench_g/rms_matmul_rbe/Kernels.lean)
and the full [authoring rules](https://github.com/VeriInfra-Lab/VeriTile/blob/main/AGENTS.md).
Legacy brace-loop and antiquote syntax remains supported by parts of the DSL;
it is not the standard for a new transcription.

## Keep the complete source

These changes break correspondence even if a real-arithmetic identity could
justify their output:

- Adding or dropping a cast, `tl.where`, mask, load or store.
- Folding a loop into `tl.sum`, unrolling it or reordering statements.
- Changing pointer updates or index arithmetic.
- Replacing a branch with `tl.where` or deleting unselected constexpr branches.
- Replacing `tl.math.rsqrt(x)` with `1 / tl.sqrt(x)`.

If an operation is unsupported, record the DSL gap. A proof may have an explicit
configuration, window or stage restriction, but that restriction does not permit
editing the authoritative source body. A deliberately different implementation
needs its own Python candidate and corresponding Lean body, with a separate
comparison theorem. Never use a prose specialization note to excuse source drift.

## Check source and proofs separately

From the core root:

```bash
python3 -m scripts.bench.generate_jit_correspondence bench/examples/VectorAdd
bash scripts/check-core-examples.sh
```

In VeriTileBench, use the same discovery command with the selected task path.
No handwritten task JSON is required. The tool checks source structure; it does
not replace independent Real correctness, FP equivalence or comparator replay.
A review should identify both sides of any mismatch, and separately report
unproved obligations or restricted theorem scope.

The standard task contains one Python module, `Kernels.lean`, `IO.lean`,
`Contract.lean`, `RealCorrect.lean`, `FPEquiv.lean` and `Proofs/`. Public Real
headlines use `Spec.Real (io ⊨ formula)` with the original conditions. FP
headlines compare the actual programs with `original ≡[R] candidate`, preserving
precision, domains and strict or experimental rule dependencies. Lower-level
`ComputeCorrect` and `ComputeRefine` lemmas support these public statements;
see [Correctness surfaces](/VeriTile/proofs/correctness-surfaces/).
