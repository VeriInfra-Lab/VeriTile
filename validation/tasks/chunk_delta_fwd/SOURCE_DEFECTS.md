# Static dot-domain defect in the checked source

The current checked Python source is **source-defect pending repair**, not a
completed full-domain kernel correctness result. This diagnosis preserves the
existing conditional Real/native proofs and does not change their statements,
the Python source, or the public input contract.

## Intended result and precision

For chunk `c`, let its visited times be
`W_c = [c*BT, c*BT + ceil(BT/BC)*BC) intersect [0,T)`. With optional initial
state `H_0` (otherwise zero), the independent Real target is

```
history[c,k,v] = H_c[k,v]
corrected_c[t,v] = U[t,v] - sum_k W[t,k] * H_c[k,v]
H_(c+1)[k,v] = H_c[k,v] + sum_(t in W_c) K[t,k] * corrected_c[t,v]
```

The returned corrected row uses the last visiting chunk, `floor(t/BT)`;
optional final state is `H_NT`. Rounded windows and their overlap are part of
the source behavior. Inputs, outputs, state and accumulators are FP32. Both
dots specify `allow_tf32=False`. Native semantics retain casts, ordered scalar
FMAs and the existing one-inner-block live-carry fusion; Real arithmetic is a
separate interpretation. No new finite-value restriction is introduced.

## Reproduced counterexamples

All three cases use `B=H=V=1`, ordinary contiguous FP32 inputs filled with one,
no optional states, and a nonempty `(1,1,1)` grid. They pass the actual checked
wrapper's guards. The checker adapts CUDA identity/context and allocation to
CPU-backed Tensor proxies; it retains actual shape, dtype, strides, storage
and pointer metadata. It captures the actual launch arguments and passes the
underlying Tensor arguments through Triton's real
`create_function_from_signature` and `_pack_args` before compiling the exact
canonical JIT function for SM80 with the first original configuration,
`num_warps=1`.

| Case | T | K | BT | BK | BC | Result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Small key width | 16 | 1 | 16 | 1 | 16 | Correction `(16 x 1) @ (1 x 64)` rejected |
| Small chunk width | 1 | 16 | 1 | 16 | 1 | Update `(16 x 1) @ (1 x 64)` rejected |
| Legal control | 16 | 16 | 16 | 16 | 16 | Both dots compile; scalar FP32 FMA PTX |

Both negative diagnostics are:

```
Input shapes should have M >= 1, N >= 1 and K >= 16
```

Here diagnostic `K` means the dot reduction dimension, not always the source
key-width parameter. In the second case the correction dot
`(1 x 16) @ (16 x 64)` passes; the subsequent update dot fails because its
reduction dimension is `BC=1`. Small M alone is not this defect.

The installed Triton 3.7.1 sources establish the static rule:

- `/usr/local/lib/python3.12/site-packages/triton/backends/nvidia/compiler.py`,
  `min_dot_size`, returns `(1,1,16)` for FP32; the function depends on operand
  bitwidth and does not read `num_warps`.
- `/usr/local/lib/python3.12/site-packages/triton/language/semantic.py`,
  `TritonSemantic.dot`, asserts those dimensions before creating the dot IR.

The report pins both installed source files and an exact excerpt of the
minimum rule. Only the first of the six original autotune configurations was
compiled for these controls. Independence of the shape rejection from warp
count follows from the inspected rule, not from six claimed compiler runs.
The legal control is a compiler sanity check, not a GPU execution or a
universal hardware-resource guarantee.

Reproduce from the repository root:

```
python bench/tritonbench_g/chunk_delta_fwd/check_min_dot.py
```

Use `--output /tmp/chunk-delta-min-dot-review` for an independent evidence
directory. The default report and TTIR/TTGIR/LLVM/PTX control artifacts are in
`.veritile-cache/kernel-evidence/chunk_delta_fwd/min-dot/`. The source is pinned
to SHA256 `cb102b78b196326239b44aa330b78f76d46120997a2b152ca94c52d67e74e193`.

## Classification and preserved results

The wrapper admits a statically ill-typed dot specialization. This is a
source/input-contract defect, not numerical uncertainty, allocator behavior,
an ordinary device-resource failure, or the deferred runtime grid-product
overflow work. The current broad input contract cannot be counted as fully
implemented by this source. Prior abstract execution and observation proofs
remain useful conditional results; they do not establish frontend acceptance
of the two rejected specializations. No Lean statement is withdrawn or
modified by this diagnosis, and no new Lean build is claimed.

Existing scalar primitive, ISA/compiler correspondence and ordinary Tensor
API parameters are model boundaries, not numerical experiments. No numerical
experiment can discharge this compilation failure. GPU execution, independent
native numerical replay and an independent cold Lean rebuild were not run for
this defect packet.

## Separate repair candidates (not applied)

1. Explicitly revise the advertised dot-bearing kernel domain to include
   `BK >= 16` and `BC >= 16`, then align the public contract and executable
   guard. The admitted positive key range would start at `K=9` because BK is
   the next power of two; requiring source `K>=16` would be unnecessarily
   stronger. Launch-skipping and zero-time specializations must be checked
   separately before defining the exact revised API domain. This is an
   explicit domain revision, not a proof of the current full contract.
2. Preserve the current domain with a separate small-dimension implementation
   or padded physical tiles. Physical tile sizes must be separated from the
   logical rounded window. Simply replacing `BC` by `max(16,BC)` changes the
   visited times for small BT. Padding also needs an explicit treatment of
   invalid intermediate lanes and NaN/Inf; zero times a nonfinite value is
   not generally zero in FP32. A candidate needs its own source, native
   target, boundary checks and proof before acceptance.

Neither candidate is implemented or used to narrow an existing theorem here.
