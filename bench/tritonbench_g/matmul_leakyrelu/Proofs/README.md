# matmul_leakyrelu

[The repaired Python source](../matmul_leakyrelu.py) and its activation helper are copied into [Kernels.lean](../Kernels.lean). Each complete kernel body has 24 top-level Python statements, including its docstring; each helper has one. Caller dtypes, scalar arguments and constexpr values remain outside the quotations. Source statements, two-argument dot followed by addition, pointer updates, masks, activation call and FP16 conversion are retained.

[The dotAcc candidate](../matmul_leakyrelu.py) is a new Python realization of the retained comparison graph: it passes the live accumulator to dot and multiplies the activation slope on the right. Its own helper is copied separately. The comparison keeps the existing `matmul_leakyrelu_fp32_k32_dot_add_to_dotacc` rule; numerical admission of that rule remains pending. No new numerical assumption was introduced.

[Correct.lean](../RealCorrect.lean) connects the complete selected grid to the independent matrix-product and Leaky ReLU formula. [FPEquiv.lean](../FPEquiv.lean) compares the two complete programs under the explicit rule. Execution, output, frame, both source safety judgments and all five retained input profiles have been rechecked locally. Fresh current independent proof replay is still required; see [validation status](../../../../validation/body-migration-progress.json).

[Contract](../Contract.lean) · [IO](../IO.lean) · [Proof helpers](.) · [Typed call and mathematical adapters](Sources).
