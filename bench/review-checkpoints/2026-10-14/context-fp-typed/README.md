# Complete context attention FP closure

Passing isolated build on 2026-10-14: 3,607 Lake jobs, 318 captured inputs,
exit 0, no changed inputs. Targets: registered FP frontend regressions and
both context_attn_fwd/context_attn_llama Correct, FPEquiv, TypedExecution closures.
The source inputs in this directory are exact copies of the passing closure.

All fp16/bf16/fp32 storage branches remain. Public FP assumptions now explicitly
include a separately validated typed-fp32 MUL-COMMUTE binding; the old scalar
atom does not imply it. See ../typed-fp32-mul-binding.md for precise scope,
provenance, the negative countermodel, and the actual typed operand guards.
No GPU execution, IEEE equality, or official independent replay is claimed.
