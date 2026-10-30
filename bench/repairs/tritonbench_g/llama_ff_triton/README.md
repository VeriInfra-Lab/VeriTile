# Checked RMSNorm–SwiGLU feed-forward

`kernel_ff` retains FP16 X and output storage, FP32 accumulators, and both source
weight branches: FP16, or int8 bytes interpreted as E5M2 floating-point payloads.
W1, W3 and RMS weights must have the same storage dtype. The historical repaired
16×16×64 kernel, four warps, two stages, arithmetic graph and tail masks are
unchanged. Original Python is preserved.

The wrapper checks tensor/layout/dtype/rank/device metadata, contiguous
zero-offset non-lazy views, positive matching K/N, signed argument/grid bounds,
all formed signed-64 byte offsets, mixed-width backing separation, and nonempty
16-byte pointer alignment before performing any view or allocation. Same-width
read aliases are supported. It transposes W1/W3 and directly views X, allocates
one FP16 result, passes canonical dense kernel strides, and returns a view with
shape `[batch, sequence, N]`. Empty leading dimensions skip pointer queries and
the device context/launch. No input normalization or arbitrary-layout copy is
performed.

`Contract.python_checks_iff` connects the ordered boolean checks to the public
contract. `Proofs/CheckedHost.domain` derives the complete historical HostDomain,
including grid/index/formed-pointer bounds, capacity bounds and output
injectivity. Its bindings describe only ordinary tensor storage, alias views and
fresh allocation metadata. Actual contiguous singleton strides are retained in
input descriptors; the canonical strides belong to the kernel views.

`Correct.checked_real` proves the independent logical RMSNorm–SwiGLU formula for
both storage branches after erasing rounding. Its explicit local Byte→Real
interpretation makes any treatment of exceptional E5M2 bytes visible.
`FPEquiv.checked_fp` composes the historical full native source proof with the
checked domain and actual returned tensor metadata. It proves both bytes of each
returned half, and the complete complement byte frame. `checked_caller_bytes`
preserves entire caller backings, including bytes outside logical input windows.
`CheckedHost.active_span` bounds absolute active pointer bytes using ordinary
allocation extent/address-space metadata. No HostDomain, index/nonoverlap proof,
whole-dot oracle, or final-output equation is supplied by a caller.

The native result is conditional on the ordinary local instruction semantics
and the structural compiler/runtime binding described in `NUMERICS.md`.
`Proofs/Reader.lean` audits the original and checked proof closure. Public Lean
files are `Kernels`, `Contract`, `Correct`, and `FPEquiv`; implementation helpers
remain under `Proofs/`.

Run `python bench/repairs/tritonbench_g/llama_ff_triton/check_contract.py` from the
repository root. The checker pins source/quotation hashes, verifies the exact
wrapper guard AST and preserved historical kernel AST, then executes the actual
kernel AST with CPU scalar/tensor instruction adapters. It checks both storage
branches, K tails 1/63/64/65/129, partial tiles, empty dimensions, read aliases,
singleton strides, exceptional data, full caller bytes and context restoration
on a partial-write failure. It also reproduces the original unmasked K-tail
load escape. These checks do not establish GPU numerical conformance.
