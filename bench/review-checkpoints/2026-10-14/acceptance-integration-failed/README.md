# Expanded public-entry integration — failed

Lake completed all 3,798 scheduled jobs with exit 1 and unchanged captured
inputs. The complete diagonal SSM correctness and FP entries now compile,
as do the three repaired context-attention real correctness entries. The
remaining failures are:

- context_attn_fwd.Proofs.FP and context_attn_llama.Proofs.FP: the half/bfloat
  paths now retain multiplication on the typed fp32 channel inside a cast;
  the old real-channel scalar rule and operand guard do not prove that goal.
- chunk_gla_fwd.FPEquiv: the public duplicated site definitions need to match
  the repaired explicit-fp32 internal sites. Their operand domains are checked
  by FPWhereContext.same_guard and do not require a new input condition.
- chunk_gated_attention.Proofs.RealHProgram: the frontend places the common
  continuation inside both gate arms, while the manual body places it after
  the conditional. Its syntax-equality claim requires a faithful decomposition
  and execution/safety bridges. This also blocks chunk_gla_fwd.Correct.

No additional assumptions or restricted input regime was inserted to hide
these failures. Later passing subsets must not be read as a passing full corpus.
Independent comparator replay remains unavailable on this host.
