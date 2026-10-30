# Kernel revision: admitted small-K native compilation failures

The checked source selected by the progress ledger has SHA-256
`070f4cc6113aa72a4f5cc1b0f8e9a6f3c44f94391c140a08f0c9d263386ce3d2`.
It is unchanged by this diagnosis. No tighter Python guard or source repair is
introduced. The existing host/model proofs retain their stated meaning.

## Independent target and actual public inputs

For chunk `n=floor(t/BT)`, the intended Real output is

```
scale * (
  exp(g[t]) * sum_d q[t,d] * h[n,d,v]
  + sum_{s in chunk n, s <= t}
      exp(g[t]-g[s]) * (sum_d q[t,d] * k[s,d]) * value[s,v]
)
```

Inputs/output and intermediate arithmetic are FP32; all three source dots set
`allow_tf32=False`. Real is the independent mathematical interpretation.
The current checked domain admits K=0 and every positive K with valid shapes,
contiguous layout, device, scalar, address spans and grid sizes.

A concrete witness uses `B=H=T=K=V=1`, `BT=16`, Q/K/V/history all +1, gate 0,
and scale 1.0. History has the required `(1,1,ceil(T/BT)*K,V)` shape. Ordinary
separate FP32 storage satisfies every existing Python guard. The independent
formula yields output 2. The actual wrapper reaches a nonempty launch with
`BK=1`, `BV=1`, grid `(1,1,1)` and the sole four-warp configuration.

The native compiler rejects the first Q-history dot:

```
b_o += tl.dot(b_q, b_h, allow_tf32=False)
       ^
Input shapes should have M >= 1, N >= 1 and K >= 16
```

The dot has shapes `[16,1]` and `[1,1]`, so its K width violates that native
requirement. Keeping V=16 also fails, isolating K from V.

The actual K=8/BK8 boundary fails too; K=9 produces BK16 and compiles with V=1.
K=16/V=1 and K=16/V=16 compile, as does the previously selected tail profile
T17/K129/V17. There is no reason to add a V>=16 restriction from these results.
K=0/V=16 is also admitted with a nonempty output (the independent formula is
zero for the finite witness), but the compiler still checks the body of the
source loop whose zero upper bound was cast to a scalar tensor and rejects its
BK1 dot. A logically zero-trip loop does not establish successful compilation.

## Evidence and exact scope

`check_kernel_admission.py` executes the actual assertion/wrapper AST on real
CPU Tensor shapes, strides, storage, pointers and dtypes, with CUDA ownership,
allocation and submission mocked. It captures the actual canonical scalar
arguments and obtains specialization using `native_specialize_impl` and
`JITFunction._pack_args`. It compiles the unchanged JIT body/signature for CUDA
SM80 using Triton 3.7.1, omitting only the host autotune decorator and selecting
its only original configuration. Unit scalars remain normally specialized.

The delivery packet pins the source, all eight cases, compiler diagnostics,
control TTIR/TTGIR/LLVM/PTX, checker and process result. Four cases fail and four
controls compile. No GPU executes, no full runtime launcher is replayed, and no
all-architecture compiler statement is claimed. These are concrete admitted
calls that fail for the selected native profile, not an empirical numerical
approximation or missing primitive arithmetic experiment.

## Effect on existing results

The prior README explicitly retained setup failure for small native dot shapes;
its whole-host theorems distinguish setup errors from successful returned output.
Those conditional propositions are not contradicted or withdrawn by this test.
Likewise the existing Real/model source proofs remain applicable to their
interpreter. They do not establish that every admitted nonempty public call can
produce the requested Tensor on the observed native profile.

This revision therefore stops at **source defect pending repair** and does not
count as completed kernel correctness. No whole-source execution premise is
introduced to hide the failure, no K domain is silently narrowed, and no
numerical experiment TODO is created. Any source fix belongs in a separately
identified candidate, preserving the original and current checked revisions.
