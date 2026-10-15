# Repaired public entrypoints and frontend regressions — passed

All 25 requested targets and their dependencies completed: 3,857 Lake jobs,
exit code 0, 568 captured inputs unchanged. `receipt.json` lists the exact
targets and source hashes; `inputs/` preserves their local source closure.

Both Correct and FPEquiv passed for context_attn_bloom, dropout_triton,
diag_ssm_triton, chunk_bwd_dqkg, chunk_delta_fwd, chunked_cumsum_fwd,
chunk_gla_fwd and chunk_gated_attention. Correct passed for context_attn_fwd
and context_attn_llama; their known failing FPEquiv entrypoints are explicitly
excluded from this passing selection and remain required by the full bundle.
Three complete legacy files passed: matmul_triton1, int8_matmul_kernel,
and int4_matmul. All registered FPFrontendRegressions passed, including the
new signed load/subtraction, fp32 where guard and constant branch regressions.

The gated recurrence proof follows the actual branch-local continuation and
proves the execution/safety bridge. Its FP finite-operand guard now occurs
inside the selected gate arm after the actual prefix. That corrects a misplaced
guard; it is not definitionally identical to the old domain. All gate/storage/
seed/final selections remain quantified. The chunk GLA where guards are
semantically unchanged for their actual operands, as checked by FPWhereContext.

This proves local Lean elaboration and the declared axiom gates for the listed
statements. It does not establish independent comparator replay, unconditional
IEEE/native correctness, or acceptance of all 184 tasks. See the full current
review bundle for the remaining failures and exact contract scopes.
