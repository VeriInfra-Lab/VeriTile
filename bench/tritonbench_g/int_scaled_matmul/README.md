# int_scaled_matmul

Both source programs have complete execution and active-output theorems in
`IntScaledMatmul.lean`. Their scope is exact integer/real algorithm semantics;
fixed-width overflow, host launch, finite-allocation safety, a complete memory
frame, and GPU numerical refinement are outside these claims.

The first kernel omits `padding_option` on its boundary-checked block-pointer
loads. These lanes are unspecified; boundary checks alone do not zero them.
The previous claim that every ragged-K launch computes ordinary GEMM was
incorrect for an arbitrary unspecified-value oracle. The repaired public theorem
keeps the same input premises and proves the actual source formula:

```
F = ceil(K / BLOCK_K) * BLOCK_K
C[row, col] = sum(k < K, A[row,k] * B[k,col])
            + sum(K <= k < F, U_A(row,k) * U_B(k,col))
```

`U_A` and `U_B` are `Int.floor` of the initial state's oracle at the two actual
block-pointer addresses. The proof carries the oracle through the loop and
proves the exact final K boundary. It adds no zero-oracle assumption and no
K-divisibility requirement. The existing positive `BLOCK_K` and output-address
injectivity premises remain. The Python source and both DSL source functions
are unchanged; the manual statement decomposition now uses the source's actual
`PaddingOption.unspecified`.

`IntScaledMatmulPadding.lean` checks a complete source execution with K=1 and
BLOCK_K=2: inputs 2 and 3 and padding values 5 and 7 produce 41, while ordinary
GEMM is 6. It also proves that aligned K has no tail for every oracle.

The second kernel already uses explicit `other=0.0` for its masked integer
loads. Its source and entire scaled-GEMM proof are unchanged:

```
C[col + N*row] = (sum(k < K, A[row,k] * B[k,col]) : Real) * s1[row]
```

Both `EVEN_K` arms remain. The existing `hCeil` covers K with `numKBlocks`;
`hEven` requires an exact K multiple for the unmasked arm. The scale address is
`row`; its passed stride parameters are unused in the Python body. The model's
stored cell is real-valued, so the host's float-to-int32 container truncation is
outside the theorem.

Disclosed surface conventions remain: integer widths are erased, typed base
regions select the integer channel, tuple dimensions/advance deltas use
brackets, block-pointer order is erased, and broadcast attributes use known
literal shapes. The second kernel's descending loop retains its existing
ascending counter representation and explicit trip-count binder.

Local validation includes both complete public entry points, the new padding
regressions, and module axiom audits. Exact inputs and logs are recorded in the
dated acceptance checkpoint. Independent comparator replay remains a separate
environment-dependent gate.
