# Dimension argmax rejects a selected NaN index

Status: **source defect pending repair**. The original and currently selected
repair both fail the independent native argmax target on admitted FP32 input.
This is a concrete program defect, not a numerical assumption to validate
later. No Python kernel, input restriction, or historical theorem was changed
by this diagnosis.

## Target and admitted input

For one dimension slice, the independent native target is the least index of a
NaN when a NaN is present; otherwise it is the least index attaining the maximum
FP32 value, including infinities. The NaN cases below agree with the actual
`torch.argmax` reference. The ideal Real target remains the least Real maximizer;
Real values do not encode NaNs, so an ideal Real theorem alone cannot cover this
native input case.

The dimension wrapper admits strided FP32 CUDA tensors and retains the original
`contiguous()` preparation. Its checks in `triton_argmax_repaired.py` and
`Checked.InputContract` in `Contract.lean` impose no finiteness or non-NaN
restriction. Each counterexample is a contiguous one-dimensional input with
`dim=0`, `keepdim=False`, and no intermediate dtype conversion. Its shape and
launch indices satisfy all extent bounds; ordinary aligned, distinct live
allocations satisfy the pointer/storage conditions. The values are therefore
within the admitted CUDA input domain. The evidence run executes the kernel on
CPU through Triton's interpreter and does not claim to execute the CUDA wrapper.

## Actual source execution

Both kernels were run with `M=1`, `N=len(input)`, `K=1`, `BLOCK_M=128`,
`BLOCK_N=128`, `INT64_INDEX=False`, and one program. The repair also receives
`program_base=0`.

| FP32 input | Original kernel | Selected repair | `torch.argmax` |
| --- | ---: | ---: | ---: |
| `[1.0, NaN]` | 0 | 0 | 1 |
| `[-3.0, NaN, 2.0]` | 0 | 0 | 1 |

For the second input, the observed result is also not the finite maximum's
index, which is `2`; treating this as a NaN-ignoring reduction does not explain
the result.

Run:

```bash
python bench/repairs/tritonbench_g/triton_argmax/check_counterexamples.py
```

The script loads only imports and the **unmodified** selected JIT function AST
from each source, so the original module's automatic CUDA benchmark does not
run. All loads, reductions, Boolean operators and stores use installed Triton
`3.7.1` with `TRITON_INTERPRET=1`; no replacement reduction tree is supplied.
Torch is `2.13.0+cu130`. There are four defect observations and six normal/tie
controls across the two sources. Input bytes and output guard elements remain
unchanged. Exact FP32 words, source hashes and all observations are recorded in
`source-defect-validation.json`. This is CPU source execution, not GPU testing
or an independent proof replay.

Original source SHA-256:
`a584c32ee9c78789629db612fa9106617214f1d07cbef9f3a225490ed013dd9d`.

Selected repair SHA-256:
`9ed6edf017075cf0397c72e5714793e8819427d2dc1e66fb7fe8d174b8f4be6c`.

## Cause and exact proof

The original source initializes the running state at lines 66–67 and updates it
at lines 77–79. The selected repair does the same at lines 165–166 and 176–178:

```python
max_values = tl.full([BLOCK_M], dtype=tl.float32, value=float("-inf"))
argmax_values = tl.full([BLOCK_M], dtype=tl.int64, value=0)
# After the coupled local value/index reduction:
update = local_max > max_values
argmax_values = tl.where(update, start_n + local_argmax, argmax_values)
```

For `[1, NaN]` there is one axis tile. If the local reduction selects index `1`,
its coupled value is NaN. The ordered IEEE comparison `NaN > -inf` is false, so
the running index remains `0`. If the local reduction selects any other index,
the committed index cannot be `1`. Thus **no coupled reduction ordering can
produce the required index `1`** with this update rule.

`Proofs/Counterexamples.lean` contains the following supplementary proof entries:

- `input_first_nan`: the independent first-NaN predicate holds at index `1`.
- `nan_comparison_false`: the exact FP32 comparison is false.
- `strict_update_rejects_nan`: the source update cannot commit that NaN index.
- `dimension_nan_counterexample`: the existing source graph differs from the
  required index for every coupled reduction tree, including trees whose leaves
  include padding.

These are exact word and program facts with standard-base axiom checks; no
floating-point experiment assumption is used. All four proof entries elaborate
successfully and pass `#axiomsClean` after the shared AST/DSL rebuild. The graph proof does not pretend
to be a new complete typed-kernel correctness proof. The actual-source CPU run
is separately recorded above.

## Preserved results and next target treatment

The existing Real proofs remain valid for their ideal Real interpretation. The
existing native graph proofs describe the supplied comparison graph, and their
Real/native bridges explicitly require reached comparison agreements. Those
claims are preserved. Agreement with a source-shaped native target cannot serve
as proof of the independent argmax formula, and the current `FPEquiv.lean` is
not yet the required equivalence between two actual kernel candidates.

The independent native target must retain NaN cases rather than adding a
`nonNaN` input guard. A later, separately recorded repair must preserve the first
NaN index through local selection and across axis tiles, then prove that repaired
kernel against the target. There is no repair candidate or correctness-completion
claim in this delivery.

The global branch explicitly casts intermediate maxima to the requested storage
dtype. Its eventual native target must retain those source conversions. A
different winner after an explicit FP16 cast is not by itself a defect against
the source-precision target. This diagnosis does not expand dtype coverage or
require primitive IEEE numerical experiments.

An earlier hypothesis that a global NaN reduction could select an out-of-range
padding index was not reproduced with the actual Triton interpreter. It is not
included as a source defect. No GPU result or compiler-tree conclusion is
inferred from a hand-supplied balanced reduction.

Numerical deferrals introduced by this diagnosis: **none**.
