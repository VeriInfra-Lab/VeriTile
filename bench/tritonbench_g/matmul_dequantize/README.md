# matmul_dequantize

`FPCorrect.lean` now gives independent floating/machine-number execution
contracts for **all three originally accepted stages**, with their actual fp16
storage profile and signed int32 packed words. Both matmul contracts cover
arbitrary K-loop length, and `matmul4_kernel` retains both `NO_GROUPS` branches.
All contracts prove success, active output readback and preservation outside
the actual write set from a specification that reads only the initial state.

- The standalone stage uses the checked source bridge to `matmul_dequant_int4`:
  fixed-width nibble extraction, exact integer promotion and directly rounded
  half multiplication.
- The second matmul stage preserves explicit `a.to(fp16)` and `b.to(fp16)`
  dot operands, `(weight-zero)*scale`, fp32 block accumulation and half store.
  Its legal 16×16, two-K-block software witness produces half `0xd4aa`.
- The first matmul stage preserves its original `dot(a,b)` without adding the
  twin's redundant operand cast, both group-loading branches, separately
  rounded half products and subtraction, the dead local `c`, and the actual
  accumulator-to-destination conversion. Its two-block witnesses produce
  `0xd4ab` for `NO_GROUPS=true` and `0xd8ab` for `false`, which loads a different
  scale in block two. They also exercise a masked row and destination gap.

The matmul dot primitive is explicit. Binding the chosen software reduction
schedule and scalar multiplication/subtraction contraction to compiled GPU
code remains a native implementation obligation. There is no real-number
factorization, associativity, global identity rounding, or assumed output in
these proofs. The 32-bit software carrier uses low 16 bits for half storage.

The following sections document the earlier mathematical model and its
accepted launch/configuration boundary, not the floating-point proof.

- Source file: `matmul_dequantize.py` (upstream `data/TritonBench_G_v1/matmul_dequantize.py`)
- Corpus: TritonBench-G v1
- Size: 358 lines, 3 `@triton.jit` kernels — **all three launched, all three modeled**
- Status: **PORTED** — `MatmulDequantize.lean`, main theorems
  `matmul_dequantize_matmul4_exec_genuine` /
  `matmul_dequantize_matmul_exec_genuine` /
  `matmul_dequantize_dequantize_exec_genuine` (`exec`-level,
  dimension-general, 0 `sorry`).

The upstream file exercises three independent int4-dequantization paths, one
per test case, and each JIT is a light textual variant of a kernel already
ported in this bench — the port mirrors those three proof stacks into one
namespace (the `decay_cumsum` multi-kernel-file precedent), in source order:

| JIT (audit anchor = first) | launcher | template port | delta vs template |
|---|---|---|---|
| `matmul4_kernel` | `matmul_dequantize_int4_gptq` | `matmul_dequantize_int4` | drops the twin's `.to(a.dtype)` on the `tl.dot` operand; dead `c`-cast spelled directly as `tl.float16` |
| `matmul_kernel` | `matmul_dequantize_int4_s2` | `int4_matmul` | the three working casts spelled directly as `tl.float16` (the twin uses inferred dtypes) |
| `dequantize_kernel` | `matmul_dequantize_int4_s1` → `dequantize_int4` | `matmul_dequant_int4` | **byte-identical** kernel body — pure rename mirror |

## `matmul4_kernel` — GPTQ dequantizing GEMM, both `NO_GROUPS` arms

`C = A · dequant(qweight)`: eight 4-bit weights per 32-bit word along K,
eight 4-bit zero-points per word along N, scale first then subtract the
pre-scaled zero-point, group-swizzled pid decomposition. `NO_GROUPS : Bool`
stays a genuine parameter — one headline covers both arms through the
`groupRow` unifier (pre-load at group row 0 vs per-step reload at
`k // (groupsize // BLOCK_SIZE_K)`). `c = accumulator.to(tl.float16)`
compiles to a genuine `Op.castFloat` but remains a **dead binding**: the
store writes `accumulator`, exactly as the source does, so the headline is
about the float32 accumulator. Disclosure: integer literals are antiquoted
`$(n)` (`0xF` = `$(15)`); `bits` / `infearure_per_bits` are transcribed as
ordinary statements.

## `matmul_kernel` — signed-dequant GEMM, `SPLIT_K = 1` arm, fp16 store

The `int4_matmul` disclosures verbatim: **(1)** `SPLIT_K` fixed to `1` (the
`tl.store` arm; the `tl.atomic_add` arm drops with the constexpr; the
headline carries `s.pids 1 = 0`); **(2)** the loop bound
`tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)` is the antiquoted binder `numKBlocks`
with side condition `K = BLOCK_SIZE_K · numKBlocks`; **(3)** the signed
nibble difference runs on the `.int` channel via
`tl.cast(·, tl.int32) - tl.cast(·, tl.int32)` and promotes through
`Op.intToReal`; `hBK8 : BK % 8 = 0` identifies the
`(BLOCK_SIZE_K * SPLIT_K * stride_bk // 8)` pointer advance with the
packed-word address. Unlike the twin, the three casts survive **faithfully**:
`b = ((int_b - int_bzp) * bs).to(tl.float16)` puts `b` on the `.fp16`
register channel, the dot's `a.to(tl.float16)` / `b.to(tl.float16)` are
preserved as the inputs to `Op.dotTyped fp16 fp16 fp32`, followed by a
`fp32 → real` cast for the accumulator. The legacy evaluator erases widths
and reduces this to the real dot; no IEEE rounding equivalence is claimed.
The stored
`c = accumulator.to(tl.float16)` makes the terminal store `.fp16`-typed —
so the headline reads the output at the `MemCell` level,
`C[row, col] = MemCell.of .fp16 (fp16(accSpec))` (the `matmul_tma`
f16-branch precedent, here on a masked `.ptr` store).

## `dequantize_kernel` — standalone tile dequantizer

`fp_b[k, n] = (nib(b) − nib(zp)) · scale`, one shared
`(offs_n < N) & (offs_k < K)` gate on every load and the store; the
`other=0.0` kwargs survive faithfully (on the packed `.nat` channels the
literal reads as nat `0`; masked-off lanes are never consumed). Disclosures
= the `matmul_dequant_int4` pair: the inline `.int`-channel respell of the
dequant statement, and `$(n)` literal antiquoting. `hInj` is the only
hypothesis beyond the dimension variables; `fpbAddr_injective` discharges it
for the host's row-major `fp_b`.

All dimensions, strides, block sizes and `group_size` stay symbolic in every
headline; no output region is ever read back into a spec. The two matmul
kernels share the pid-swizzle definitions (`numPidM` … `pidN`) and the `C`
address map (`cAddr`, `cAddr_injective`) — they are textually identical in
the source.

The complete four source/projection paths (one in the sibling int4 module,
three here) are replayed with exact AST decomposition and full-module axiom
audits. Public theorem premises and conclusions are unchanged. These remain
legacy exec/output-readback statements: they do not prove a frame, finite
allocation safety, GPU scheduling, or the Python host launch.
