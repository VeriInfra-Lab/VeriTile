# Paged KV cache filling

The selected Python source is
[kv_cache_filling_repaired.py](kv_cache_filling_repaired.py). It validates
metadata and ownership, fills caller-owned paged caches, and returns `None`.
The original benchmark file is retained separately. This directory verifies
the selected repair, not an unmodified benchmark with different guards.

## Source and proof entries

[Kernels.lean](Kernels.lean) contains the complete `_quant_row` helper and
complete `_fill_rows` function bodies. Each policy-0/8/4 caller profile retains
all source branches, names, loads, casts, reductions, helper calls and stores.
Pointer types, constexpr arguments and caller bindings are outside the pasted
body. Policy is a checked caller constant; it is not an invented source
statement. Copy calls bind the unused metadata pointers to cache pointers,
matching the Python launch.

[candidate_kernels.py](candidate_kernels.py) transcribes the already selected
candidate: only the vector key-load address changes `src * DK` to `DK * src`
in each policy branch. Scalar loads and every other expression are unchanged.
[jit-correspondence.json](jit-correspondence.json) checks the helper and all six
complete original/candidate profiles. The Python source is not generated from
the Lean implementation.

The source helper is expanded with one evaluation of each argument, local
variable isolation, and simultaneous tuple results. Python `_` remains an
ordinary variable, including its repeated assignments. The second INT4 half
still calls the helper and recomputes its scale and zero, as Python does.
Implicit uint8 add/multiply wrapping belongs to frontend lowering. No extra
`.to(tl.uint8)` is inserted into the pasted packing expression.

The primary entries are [KernelOnly/Correct.lean](KernelOnly/Correct.lean) and
[KernelOnly/FPEquiv.lean](KernelOnly/FPEquiv.lean). Their existing public
statements and [Contract.lean](Contract.lean) input domain are retained.
[KernelOnly/IO.lean](KernelOnly/IO.lean) defines their complete typed scheduling,
key/value, cache and metadata channels.

- `Correct.whole_shared_real` covers every validated job, both versions and
  every program ID, with the independent full-head Real quantization formula,
  complete output observations and exterior frame.
- `FPEquiv.kernel_native` compares the actual source and integer-index
  candidate as `original ≡[[]] indexCommuted`. Both executions, all active
  outputs, access safety and exterior frames are included. No floating-point
  arithmetic rewrite or experimental rule is needed for this transformation.
- [Proofs/SourceBlocks.lean](Proofs/SourceBlocks.lean) proves the full raw
  quantized programs execute their selected constexpr branches for arbitrary
  states, including failed executions. Its list decompositions are proof
  tools; they do not construct the authoritative kernels.
- [Proofs/CopyExecution.lean](Proofs/CopyExecution.lean),
  [Proofs/QuantExecution.lean](Proofs/QuantExecution.lean), and
  [Proofs/QuantExecution4.lean](Proofs/QuantExecution4.lean) connect complete
  source execution to row results. The `KernelOnly/Proofs` modules compose
  access safety, typed readback, logical observations and public conclusions.

`Proofs/SourcePreservation.lean` retains the historical manual quotations for
reference. It records the actual new source inventory and semantic branch
connections; it does not claim those different ASTs are identical.

## Input range and mathematical meaning

For batch `b`, token `i < count[b]`, and head `h`, the source row is
`(start[b] + i) * heads + h`. With `position = total[b] - count[b] + i`, the
destination row uses `page[b, position // block]`, `position % block`, and `h`.
The host rejects duplicate destination tokens before writing.

State tensors are FP32. Policy 0 copies into FP32 caches; policy 8 writes
uint8 codes at full width; policy 4 packs two halves into one byte. Quantized
policies also write FP32 scale/zero pairs. Positive head widths and
head/block counts, packing shapes, metadata coverage, allocation separation,
source limits, non-power-of-two padding, storage offsets and empty work
retain the existing contract. Payloads retain all FP32 words, without a new
finite-only restriction. Metadata is signed INT32 and scheduler buffers are
INT64. Unused negative page entries remain allowed.

The Real interpretation uses full-head extrema, scale `(max-min)/levels`
(or one for a constant head), zero `-min/scale`, nonnegative nearest integer
codes, and low-half plus sixteen times high-half packing. This independent
mathematical formula is not a claim that FP32 arithmetic equals real arithmetic.

Native scalar operations and ordered reductions remain the existing explicit
primitive boundaries. `NativeProfile` preserves minnum/maxnum, division,
rounding and special-value behavior. `PtxCast` describes the selected total
PTX conversion followed by a byte store; historical partial-conversion
counterexamples remain valid for that older model. The empty experimental
rule set in the current pair theorem does not assert portable LLVM-poison
behavior or prove compiler/GPU implementations.

## Optional physical Tensor application

[KernelOnly/Misc/TensorBridge.lean](KernelOnly/Misc/TensorBridge.lean) and
[KernelOnly/Misc/NativeCorrect.lean](KernelOnly/Misc/NativeCorrect.lean) remain
optional consumers. [IO.lean](IO.lean) gives the distinct physical-byte view:
caller pointers, actual descriptor/axis stride observations, signed metadata
loads, scheduler buffers, and typed input reads all refer to one initial heap.
Host allocation and launch realization are explicit input ABI boundaries.

The formerly missing [Proofs/WholeBacking.lean](Proofs/WholeBacking.lean) is
restored as tracked source. It proves input-row binding, source/destination
loads, actual-store byte readback, complete byte frame and Tensor-coordinate
readback. The `SourceBacking` projection comes from the executed regional
states, not an assumed final output. The optional schedule binding is now
explicit source; it does not narrow the primary kernel domain. Retained raw
family endpoints use this same actual-state projection and store list.

## Reproduction and evidence

Build from the repository root with the pinned Lean toolchain:

```sh
lake build bench.repairs.tritonbench_g.kv_cache_filling.KernelOnly.Correct \
  bench.repairs.tritonbench_g.kv_cache_filling.KernelOnly.FPEquiv \
  bench.repairs.tritonbench_g.kv_cache_filling.KernelOnly.Misc.TensorBridge
python3 -m unittest bench.tests.test_kv_cache_filling_body
```

`verified-entry.json` is historical evidence, not a receipt for this migration.
The current source correspondence inventory and repository review checkpoint
record the migrated snapshot and completed validation. Historical isolated
build directories are unnecessary for compiling these tracked modules.

`check_contract.py` is a CPU Triton-source interpreter with a labeled
CUDA-metadata facade; it is not native GPU validation. A missing Torch
installation or unavailable external proof comparator is recorded as
unavailable, never as passed. This migration performs no new GPU experiment.
