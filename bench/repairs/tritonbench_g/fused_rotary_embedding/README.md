# fused_rotary_embedding: checked decoding rotation and cache scatter

The checked repair rotates full split-half Q in place, rotates and scatters K,
and copies V to its decoding cache slot. It returns `None`, retaining the nine
original Tensor objects and their full descriptors. K, V, Cos, Sin, block tables,
lengths, unused cache slots, and all other bytes remain unchanged. One token is
one decoding sequence; coefficient rows are already aligned to token rows.

## Supported call

All inputs are ordinary exact `torch.Tensor` objects, strided, canonical
contiguous CUDA tensors on Q's device, with zero storage offset and no negative
or conjugate view flag. Payloads are FP32; metadata is int32. There are no
payload copies or layout normalization.

- Q is `[T,HQ,D]`; K and V are `[T,HK,D]`. HQ and HK are positive, HQ is divisible
  by HK, and D is positive and even. Padded masked lanes admit non-power-of-two
  even widths. The whole D is rotated; there is no partial-rotary parameter.
- Cos and Sin independently have at least T rows and D/2 columns. Their row
  widths can differ. Nonempty tensors have exact canonical strides, including
  singleton axes. Empty tensors retain their unused strides.
- V-cache is `[N,HK,B,D]`. K-cache is either `[N,HK,B,D]` or
  `[N,HK,D/x,B,x]`, with positive N, B, and x dividing D.
- Lengths are `[T]` and block tables `[T,M]`, M positive. Each positive signed
  length chooses an existing table entry. The selected block is in `[0,N)`;
  `(block,(length-1)%B)` is distinct across tokens. Unused table entries may
  contain any int32 word, including negative values.
- Every tensor has at most `(2**63-1)//4` elements. Live calls additionally
  require four-byte aligned allocation starts, pairwise disjoint complete
  storage intervals, and checked signed-offset/uint64 room for every formed
  pointer, including padded masked lanes. CUDA grid axes satisfy HQ < 2**31
  and T <= 65535. The shared Triton launcher grid-product compatibility item
  remains deferred as recorded in `bench/TODO.md`.

T=0 returns without querying pointers, copying metadata, or launching. Live
calls copy only lengths and block tables to CPU to validate destinations, then
launch in Q's device context and restore the caller's context. Runtime lifetime,
absence of concurrent input/metadata changes, successful compilation/resources,
and successful CUDA execution remain ordinary execution premises.

## Source and proof

The original benchmark file remains untouched. Its independent Sin pitch was
ignored, its indices could overflow before pointer addition, and its own
fixture has T=16 but only two metadata rows. The repair supplies independent
Q/K/V and Cos/Sin strides, widens indices before arithmetic, masks padded lanes,
and rejects malformed metadata or colliding cache destinations. It keeps the
original arithmetic expression trees, four/eight/sixteen-warps selection at
D=256/512, default stages, and explicitly enables the existing FP fusion.

The public endpoints are:

- `Correct.fused_rotary_real`: the independent arbitrary-column split-half
  formula, full Q/K/V output coverage, original objects, `None`, context,
  exterior frame, and empty-call identity, under the exact-real projection.
- `FPEquiv.fused_rotary_native`: the same complete call over finite uint64 byte
  memory, actual current metadata reads, exact FP32 words and all four bytes,
  nonwrapping pointers, full descriptors, read-only/exterior frames, and empty
  identity.
- `FPEquiv.fused_rotary_native_any_order`: the same finite-byte result for any
  complete legal work order, lifting the retained source permutation proof.
- `FPEquiv.fused_rotary_native_if_local`: the native result transported using
  only local `mul`, `neg`, and `fma` word correspondences.

The retained geometry proves destination injection for both cache layouts;
vector and schedule proofs connect the full-width V load/store to the paired
proof schedule. Metadata is decoded from current little-endian bytes and
proved stable. `Arena` executes individual finite four-byte stores and proves
its current-memory mirror equals the existing region-byte source execution.
It does not install a computed target tensor as a runtime result. The public
FP dependency closure excludes the task's mathematical/Real endpoints.
`Contract` derives admission from observed descriptors and finite metadata
checks; `Addresses` bounds the actual canonical source address expressions,
including inactive padded lanes. No output equation is an input premise.

The manual Python-to-Lean source interpretation is supported by the unchanged
arithmetic AST, full actual-kernel interpreter checks, and compiler dataflow
inspection. It is not a certified Python/Triton frontend or universal compiler
proof. The selected Triton 3.7.1 profiles use these exact ordered graphs:

- Q low: `fma(q0,c,neg(mul(q1,s)))`; Q high: `fma(q1,c,mul(q0,s))`.
- K low: `fma(c,k0,neg(mul(s,k1)))`; K high: `fma(s,k0,mul(c,k1))`.
- V: raw bits, with no arithmetic conversion.

All raw FP32 words are included: NaNs, infinities, signed zeros and subnormals.
There is no commutativity, unfused-versus-fused equality, exact FP-to-Real claim,
or assumed full-output agreement. Native instruction numerical conformance
for the local `mul`/`neg`/`fma` ports remains deferred to the shared experiment
batch. CUDA is unavailable here; no GPU execution is claimed. The eight saved
profiles are finite compiler evidence, not a supported-width cap.

## Validation

All 24 Lean modules compile with standard axioms only. Public formulas and
host checks pass non-circularity audits. Run the focused checks with:

```bash
python bench/repairs/tritonbench_g/fused_rotary_embedding/check_contract.py
python bench/repairs/tritonbench_g/fused_rotary_embedding/check_cuda_lowering.py
python bench/repairs/tritonbench_g/fused_rotary_embedding/check_precision.py
```

Results: 43 host admissions, 27 rejections, nine complete actual-kernel CPU
interpreter grids with input/cache-hole byte frames, and three original-source
controls. V-copy controls include NaN payloads, negative zero, infinity and a
subnormal word. Eight actual-JIT SM80/SM90 profiles compile, covering D=2, 6, 16,
96, 128, 256 and 512, both cache layouts, independent coefficient widths and
singleton scalar specializations. The hash-bound PTX audit checks all 40 Q/K/V
store dataflows with their exact operand orders. Artifacts are saved under
`/tmp/veritile-fused-rotary-lowering`.

Original SHA256:
`3581c60f75eb5314c472d4e9ccfd2c0cc8c6a089352352247258c8cbabe7f630`.
