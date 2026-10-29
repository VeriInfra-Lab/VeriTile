# parallel_retention_attention

- Source file: `parallel_retention_attention.py` (upstream `data/TritonBench_G_v1/parallel_retention_attention.py`)
- Corpus: TritonBench-G v1
- Size: 398 lines, 4 `@triton.jit` kernels
- Status: **PORTED** (forward + the backward dk/dv helper) —
  `ParallelRetentionAttention.lean`, main theorems `pra_fwd_o_exec_genuine`
  and `pra_bwd_dkv_exec_genuine` (`exec`-level, dimension-general,
  0 `sorry`).

The upstream file implements chunk-parallel *retention* attention: the
score is the plain dot product `q·k` weighted by the per-head exponential
decay `γ^(pos_q − pos_k)` with `γ = 2^b_b`,
`b_b = log2(1 − 2^(−5 − i_h))`, `i_h = i_bh % H`.

- `parallel_retention_fwd_kernel` (the launched forward JIT) streams K/V in
  two phases — the strictly-below-diagonal blocks unmasked with
  `tl.advance`d block pointers and the decayed accumulator recurrence
  `b_o = b_o · 2^(b_b·BTS) + (Q·K_j ⊙ d_h)·V_j`, then the diagonal block
  causally masked with the intra-block decay tile `d_s` — rescales by
  `d_q[:, None] = 2^(a·b_b)` in between, and stores `o` (block pointer,
  boundary-checked).
- `_parallel_retention_bwd_dkv` (the backward dk/dv helper JIT) streams
  Q/dO in the *opposite* order: the strictly-above-diagonal blocks via a
  **descending `-BTS` loop** (the final member of the descending-range
  lever family — the lever is now closed) with the mirrored decay
  recurrence (`b_dk *= d_b`, `b_dv *= d_b`, `b_do` pre-scaled by
  `d_q = 2^(c·b_b)`, `b_kd = b_k ⊙ 2^((BTL−r)·b_b)`), the post-loop
  rescalings `b_dk *= d_h[:, None]·scale` / `b_dv *= scale`, then the
  diagonal block masked. It is verified as a standalone kernel over
  universally-quantified scalar arguments `i_bh, i_c, i_k, i_v, i_h` — the
  values the shell `parallel_retention_bwd_kernel` would pass.

Headlines: `o[a, p]` =
`Σ_{t ≤ i_c·BTL + a} 2^((i_c·BTL + a − t)·b_b) · score(a,t) · v[t,p]` with
`score(a,t) = (scale·q[a]) · k[t]` over the `BK` head window; `dk`/`dv` =
`Σ_{t ≥ i_c·BTL + r} 2^((t − i_c·BTL − r)·b_b) · scale · (v[r]·do[t]) · q[t]`
/ `… · (k[r]·q[t]) · do[t]` over the kernel's exact query sweep ending at
`max(cdiv(T,BTS)·BTS, (i_c+1)·BTL)`. All lanes in these formulas are the
independent guarded input values described below. The formulas cover ragged
tails without a divisibility hypothesis on `T`, including the source's
unspecified values outside the parent window.

The Python block-pointer loads omit `padding_option`. Their lowered padding
is `unspecified`: an out-of-window load reads the initial state's existing
`undef` function at the actual region and address. For a pointer into region
`rg`, with base `b`, offsets `(r₀, c₀)`, strides `(σr, σc)`, and parent shape
`(R, C)`, lane `(r, c)` has address
`A = b + (r₀ + r)·σr + (c₀ + c)·σc`. Its modeled input value is
`if r₀+r < R ∧ c₀+c < C then s.readMem rg A else s.undef rg A`, where `s`
is the initial state. For example, the forward Q address is
`s.pids 2·s_qk_h + (s.pids 1·BTL+a)·s_qk_t + (praIk s V BV·BK+e)·s_qk_d`;
`praQGuarded` selects the memory or `undef` value at that address and then
multiplies it by `scale`. The other six guarded lane functions use their
respective source addresses in the same way. Loop invariants preserve the
initial memory and `undef` function through the load/compute phases.

The repair corrects the numerical reference functions: their former
out-of-window zeros are now the actual initial input choices. Consequently,
ragged head dimensions and backward tail summands can depend on nonzero
`undef`. Neither original headline has a zero-`undef` premise, and no public
premise was added. The existing premises remain:

- Forward: `s_vo_d = 1`, `BV ≤ s_vo_t`, `BTL % BTS = 0`, and `0 < BTS`.
- Backward dk/dv: `dk ≠ dv`, `s_qk_d = 1`, `s_vo_d = 1`,
  `BK ≤ s_qk_t`, `BV ≤ s_vo_t`, `BTL % BTS = 0`, and `0 < BTS`.

The complete-execution regression `pra_unspecified_padding_correct` checks
`T=K=V=BTL=BTS=BV=1`, `BK=2`, zero program IDs, and unit scale/strides.
In-bounds Q and K are zero, in-bounds V is one, and the padding input is
`u`; the actual forward body stores `u*u`. In particular, padding one gives
output one, while padding zero gives output zero. This regression is part
of the module axiom gate and demonstrates why replacing unspecified padding
with zero changes the result.

These are per-program `exec` results for active output lanes in the real
arithmetic model. They provide no frame theorem for other memory, no host
launch/composition or host-side NK/NV reduction proof, and no bit-accurate
IEEE or FP32 refinement. The backward helper is verified independently;
composition with the shell and the dq helper remains outside this proof.

The helper-JIT structure (no cross-`@triton.jit` call surface), the dropped
trailing bare `return`, the descending-loop respelling
(`for j in range(0, cdiv(cdiv(T,BTS)·BTS − (i_c+1)·BTL, BTS))` with
body-first `i = cdiv(T,BTS)·BTS − BTS − j·BTS`; the `±BTS` cancels exactly
in ℤ and ℕ-truncation reproduces Python's empty range — the
`parallel_attention` respelling verbatim), the diagonal unary-minus decay
index `-o_k[:, None] + o_q[None, :]` respelled as the subtraction
`o_q[None, :] - o_k[:, None]` (identical on every lane the causal
`tl.where` mask keeps), and the explicit `tl.toReal(...)` int→float casts
(the `chunk_retention` precedent) are a `Translation-surface blocker:`
registered in `proof_blockers.md`. The backward shell
`parallel_retention_bwd_kernel` and `_parallel_retention_bwd_dq` are the
trusted boundary.
