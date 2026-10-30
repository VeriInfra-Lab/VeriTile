# matmul_triton2

Reference responsibilities (2026-10-23): `KernelOnly/Reference.lean` retains the independent exact K-term Real matrix product. The optional padded native recurrence is in `KernelOnly/Misc/NativeReference.lean`, and its input connection is in `KernelOnly/Misc/NativeInputConnection.lean`, imported explicitly by `KernelOnly/Misc/NativeCorrect.lean`. The main Real and FP proofs do not import these optional entries.

The current checked repair is
[`revision001/matmul_triton2_repaired.py`](revision001/matmul_triton2_repaired.py).
The original benchmark is unchanged. Start with the current kernel-only [`Correct.lean`](KernelOnly/Correct.lean),
[`FPEquiv.lean`](KernelOnly/FPEquiv.lean) and shared [`IO.lean`](KernelOnly/IO.lean).
They select the unchanged revision001 source and reuse its proofs; the old
root entries remain historical compatibility exports. See the
[current interface scope](KernelOnly/README.md).

The supported inputs are compatible contiguous FP32 matrices on one CUDA device,
with resolved negative view bits, signed-64-bit dimensions and four-byte-aligned
nonempty pointers. Empty matrices, contiguous offset/overlapping inputs and all
FP32 values are included. The original eight autotune configurations remain.

In namespace `VeriTile.Bench.Repairs.MatmulTriton2.Revision001`, the strongest
results are `matmul_triton2_autotuned_real`, `matmul_triton2_autotuned_native` and
`matmul_triton2_autotuned_tf32_if_local`. They cover exact admission, finite
indexing, all successful tuning trials, final selected launch, original logical
input values, actual returned modeled allocation and caller frame. The last
result adds explicit local zero/sync-MMA8/warp-group-MMA8 correspondence.
The instruction-family selection records Triton 3.7.1 SM90 lowering;
GPU numerical conformance remains unvalidated.

See the [revision scope and validation](revision001/README.md) for the original
out-of-bounds witness, masks/index repair, full autotune proof, local numerical
domains and validation results. Shared launcher compatibility remains the
separate `bench/TODO.md` item.


## Optional proofs in Misc, 2026-10-21

Optional native observations, Tensor applications and reader/audit entries are
imported explicitly from `Misc/`. The primary `Contract.lean`, `Correct.lean` and
`FPEquiv.lean` keep the same mathematical and program-pair scope. Main proofs do
not re-export optional native conclusions. Declaration names, statements and
proof bodies are preserved; this file organization changes no input domain,
precision, numerical rule or task completion status.
