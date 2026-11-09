# apply_penalty

[Python source](../apply_penalty.py) · [copied Lean bodies](../Kernels.lean) · [input contract](../Contract.lean) · [Real correctness](../RealCorrect.lean) · [FP equivalence](../FPEquiv.lean).

`Kernels.lean` contains the original and existing reciprocal candidate, each with the complete 17-statement Python body. Invocation metadata supplies FP32 payloads and signed int32 metadata outside the quotation. The program-id, cumsum, token-ID, arange and stride int64 conversions remain in the body. Counts remain signed until the source multiplication converts them at FP32 precision.

The independent Real formula is `(if 0 < logit then logit / repetition else logit * repetition) - count * frequency - presence`. Its IO pins nonnegative mathematical metadata to actual integer cells. The proof includes source execution, access safety, flat allocation transport, active output values under the required scatter injectivity, and the exterior frame. The retained host proof connects the checked int32 words to these inputs and composes the complete admitted grid.

FP equivalence uses the actual two typed programs and the existing `div_mul_rcp` rule only on active lanes whose positive arm is selected. Its finite/nonzero operand domain is unchanged; inactive and unchanged-arm lanes add no numerical conditions. Repeated token IDs remain allowed for equivalence because both sources retain the same store order. The `ProgramIO` view also reads signed metadata cells.

[Proof helpers](.) contain execution and mathematical connections, not alternate kernel implementations. [Validation progress](../../../../validation/body-migration-progress.json) distinguishes local verification from accepted independent replay; compilation and body comparison alone do not mark a task complete.
