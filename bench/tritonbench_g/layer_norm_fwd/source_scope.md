# layer_norm_fwd source and proof scope

The complete `_layer_norm_fwd_1pass_kernel` body is imported from TritonBench
commit `603e28a`. The whole upstream file, including host wrapper and tests, is
retained verbatim. The proof target is each launched row of the complete JIT,
with symbolic `M`, `N`, block size and row strides.

The JIT's ten flags are `IS_RMS_NORM`, `HAS_RESIDUAL`, `STORE_RESIDUAL_OUT`,
`HAS_BIAS`, `HAS_DROPOUT`, `STORE_DROPOUT_MASK`, `HAS_ROWSCALE`, `HAS_X1`,
`HAS_W1`, and `HAS_B1`. The source macro keeps their complete branch tree. In
particular, the second input's rowscale path is present even though the host
wrapper rejects `x1` together with `rowscale`.

| Output | Write condition | Input-only mathematical specification |
| --- | --- | --- |
| `Y[row, col]` | `col < N` | `Formula.y` |
| `Y1[row, col]` | `HAS_W1` and `col < N` | `Formula.y1` |
| `Mean[row]` | not `IS_RMS_NORM` | `Formula.mean` |
| `Rstd[row]` | always | `Formula.rstd` |
| `RESIDUAL_OUT[row, col]` | `STORE_RESIDUAL_OUT` and `col < N` | `Formula.combined` |
| first `DROPOUT_MASK[row, col]` | dropout and mask flags, `col < N` | `Formula.keep seeds[row]` |
| second `DROPOUT_MASK[M + row, col]` | additionally `HAS_X1` | `Formula.keep seeds[M + row]` |

`Contract.Post` specifies all memory cells, including exact tagged cells
outside those writes. It uses only the initial input snapshot. It does not
assume the target correctness result. Both complete source versions establish this postcondition and successful
completion. The candidate proof executes its changed prefix independently and
composes the complete common source suffix. Both versions also have successful
finite-memory execution proofs, with every physical cell specified and a full
frame outside the actual source writes.

The host creates `SEEDS` with `torch.int64` and values in `[0, 2^32)`. The JIT
loads int64, performs `.to(tl.uint32)`, and calls canonical `tl.rand` with seven
rounds. `Formula.seedU32` records modulo `2^32`; its host-range lemma proves
value preservation on the host-created seed domain. Neither a uint32 pointer
declaration nor a ten-round seeded-dropout theorem proves this source.

The candidate exchanges `row * stride_x_row` with
`stride_x_row * row`. It leaves all floating expressions, eager operands,
branch conditions, reduction order, casts and stores in place. `Proofs/FP.lean` proves equality of the whole typed fp32 execution for every
structural floating algebra, including failures and every state field.
`Proofs/TypedExecution.lean` separately proves complete successful runs in an
explicit `WithBot ℝ` consistency algebra, starting from empty registers at PID 1
with negative int64 seeds, `N=2`, `BLOCK_N=4`, all optional features and both
normalization modes. This witness has 14 clean audit gates and asserts no IEEE
or native execution. Other storage profiles remain separate obligations.

Current native evidence samples six complete-source profiles at `BLOCK_N=128`
on Triton 3.8.0 / SM90. It covers both normalization branches, two random
streams and all seven observable output paths. It does not prove all dtype
combinations, symbolic block sizes, runtime-scalar ABIs, GPU scheduling,
numerical results or equivalence.

The selected fp32 source profile passes the strict complete AST audit on the
actual port entry, retaining all ten symbolic flags. The fp32 source-lowering
checks and full-state FP theorem separately pass Lean and `#axiomsClean`.
The complete original real execution and independent tagged-memory contract
pass 36 audit gates; candidate execution and the same independent contract pass
seven further gates. Finite flat-memory safety, successful execution, all seven
outputs and full frame pass 40 further gates. The remaining scope includes
additional storage profiles, the host-to-native floating scalar ABI, and the
fixed-width index arithmetic bridge even for the selected fp32 storage profile.

In the pinned Python source, `eps`, `dropout_p`, the six strides, `M` and `N`
are unannotated runtime arguments. The installed Triton 3.8.0 runtime classifies
Python floats as fp32 and selects integer widths from their values. Its NVIDIA
launcher converts a Python float through a C float for fp32 arguments. All six
offline receipts explicitly choose fp32 for `eps`/`dropout_p` and i32 for the
integer arguments; they do not prove that every host call has this ABI.
The Lean source instead uses real-valued scalar parameters and natural-valued
strides and dimensions. No theorem currently connects host float conversion
to those values or proves native integer intermediates avoid overflow.
The selected AST audit verifies the source expressions and binder schema, not
these encoding and arithmetic bridges. Real formulas and universal typed
equivalence retain their stated interpreter scope.

The complete second-input body and residual load/store phases have successful
real execution proofs. Input preservation uses `Domain.ReadWriteSeparated`:
every active input region differs from every written region, matching the
wrapper's fresh output allocations. Read-only inputs may alias one another;
inactive pointers are unrestricted. Output-to-output aliases are retained in
the mathematical last-store order. The second-stream audit has 18 clean gates,
and residual/input-preservation audit has 24. Prefix composition passes 21
gates, and the complete affine tail is included in the original real contract.
These results cover symbolic flags, row and block size, with `N ≤ BLOCK_N` and
active read/write separation. The finite allocation theorem additionally
requires a valid launch row, explicit extents for active buffers, disjoint
physical allocation regions and coverage, and zero `undef` for the current
flatten bridge. It derives complete trace safety from these concrete bounds;
neither trace safety nor any intermediate execution is assumed publicly.
