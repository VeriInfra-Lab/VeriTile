# Checked mixed sparse attention repair

The repaired wrapper supports FP32 Q/K/V and output, FP16 internal operands,
64-by-64 tiles, D in {16, 32, 64, 128}, four warps and two stages. Inputs have
shape `[B,H,T,D]`, positive H, one CUDA device, and canonical contiguous live
strides. The wrapper inspects signed int32 sparse metadata, checks explicit strided layout, complete
storage and formed-address bounds, creates the intended fresh zero output,
and launches the original repaired two-loop kernel. It does not copy floating
input payloads. Read-only input aliases, including FP32/int32 views of the
same backing, remain allowed. Empty B or T returns the fresh empty output
before launch-only checks or metadata content reads.

The retained kernel repairs narrow K and V to FP16, guard all-masked block
centering, guard a zero normalizer, and retain the final FP16 cast followed
by widening into the FP32 output. Inactive sequence tails and empty sparse
support remain zero. Block selections are causal, explicit columns are
noncausal, and duplicates retain their order and multiplicity.

Proof paths below are relative to the task directory.

## Proof entrypoints

- `Kernels.lean` and `Misc/HostSource.lean`: the selected typed source and full source
  execution proof, including both dynamic loops.
- `HostContract.lean`: raw descriptors, the observed signed metadata checks,
  selected strides, scale ABI word, empty/live validation, and derivation of
  the source shape, metadata, allocation, and machine contracts.
- `Misc/HostCorrect.lean`: `Checked.Real.checked_call`, from actual typed-source
  execution to the independent occurrence-list weighted formula. The exact
  ideal formula uses `pow2((scale * 1.44269504) * dot(q,k))`; it preserves the
  source's decimal coefficient rather than replacing it with exact log2(e).
- `Misc/HostFPEquiv.lean`: `Checked.FP.checked_call`, from the staged native typed
  source to the independent initial-word graph under only local instruction
  and comparison transport. The actual marshalled scale word comes from
  `Request.scaleWord`. The theorem gives returned descriptor identity,
  exact output words, all four output bytes, 64-bit byte-address closure,
  the complement byte frame, and unchanged input words/signed metadata.
  `checked_tensor_call` additionally identifies the returned actual Tensor
  object and shape and proves shared `Torch.FP32.logicalValue` readback using
  the ordinary `TensorLaws`/descriptor observation bridge.

All helpers are below `Proofs/`. The Real and FP public roots have independent
import closures. Neither target invokes the source executor or reads its
returned output. Both source adapters do call `Physical.invoke`, whose host
call executes the complete typed AST and fresh zero allocation. There is no
assumed whole-loop or final-output equation. `Reify.prepare` constructs input
cells directly from the current physical word readers. `Reify.checked_access`
combines the proved bounds of every source load family with its exact raw
address; enabled loads cannot take the total reader's fallback.

The raw-memory fold is a projection of the single returned allocation and
single launch, including zero-initialized tails. It is not a Python launch
per output element. Fresh output separation proves every source extraction
observes the initial inputs, regardless of projection order.

## Validation and limits

The CPU Triton interpreter/guard suite contains 65 cases: 62 numerical,
alias, empty, and rejection cases, two targeted strided-layout rejections,
and the separately checked JIT AST pin.
Approximate Real comparisons use `atol=rtol=5e-3` as diagnostics. The suite
also checks exact final FP16-narrowed/FP32-widened storage values and unchanged
inputs. Eight offline SM80 specializations compile (all D, normal and
scalar-one profiles), exposing FP16 m16n8k16 MMA, conversions, exp2 and stores.
CUDA execution was unavailable; none of these are GPU numerical measurements.
The Lean endpoint proofs and axiom audits are separate evidence.

Run from the repository root:

```sh
python bench/tritonbench_g/mixed_sparse_attention/check_contract.py
python bench/tritonbench_g/mixed_sparse_attention/check_lowering.py
python bench/tritonbench_g/mixed_sparse_attention/check_lean.py
```

`SOURCE_BINDING.md` and `numerical-obligations.json` describe the ordinary
runtime/compiler realization boundary and the remaining local numerical
obligations. The eight compiler samples and source hashes do not establish
compiler correctness for every admitted call. Shared launcher grid-product
overflow remains the separately tracked item in `documents/bench/TODO.md`.
