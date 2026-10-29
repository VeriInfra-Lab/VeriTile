# parallel_attention

This port proves complete forward execution (`pa_fwd_o_exec_genuine`) and the
standalone backward dk/dv helper (`pa_bwd_dkv_exec_genuine`) against independent
input formulas. The Python source is unchanged. Arithmetic is real-valued;
this is not a bit-accurate native floating-point refinement claim.

The source's block-pointer loads omit `padding_option`. Their out-of-window
lanes therefore use the initial state's address-indexed `undef` inputs. The
masked `dz` load similarly retains its unspecified inputs. All eight guarded
mathematical references now express these values directly, and the loops
preserve both memory and `undef`. The previous zero-padding reference formulas
were incorrect for arbitrary source states.

The forward theorem's hypotheses remain output-buffer separation, contiguous
last-dimension stride/store injectivity, `BTL % BTS = 0`, and `0 < BTS`. The
backward theorem retains the analogous original layout and trip-count
hypotheses, and removes the old global `∀ rg off, s.undef rg off = 0` premise.
Its reference formulas now include the actual masked `dz` padding input. This
is an explicit change to the backward public contract and both input reference
families, not a claim that the prior theorem statements remain unchanged.

The actual source casts are retained in q scaling, dO loads and dot operands;
every explicit `allow_tf32=false` option remains in the lowered AST. A local
proved real-to-real cast identity connects these operations to the real
execution proof. No new numerical assumption, oracle or desired-output premise
is introduced.

`pa_unspecified_padding_correct` is an audited complete-source execution
regression. With a one-element real head and `BK=2`, in-window Q/K values zero,
V one and arbitrary unspecified input `u`, actual forward execution stores
`o[0] = z[0] = (u*u)*(u*u)`. Thus nonzero unspecified input affects retained
output lanes; replacing it with zero is observably wrong.

The forward computation is causal quadratic attention: `score²` contributes
to both the unnormalized value sum `o` and normalizer `z`. Host normalization,
cross-program composition and the backward shell/dq helper remain outside
this port. The existing named-correctness-surface/frame and helper/descending
source-translation review boundaries remain recorded in `proof_blockers.md`.
The retained ragged source inputs do not require divisibility of `T`.

The 2026-10-14 acceptance checkpoint records the complete Lean target and
module axiom gate, exact frozen inputs, the prior/final public statements and
reference definitions, and staged failure/interruption logs. It does not
claim GPU execution or an independent numerical comparison replay.

Acceptance evidence: [complete build and axiom audit](../../review-checkpoints/2026-10-14/parallel-attention-final/README.md),
[exact contract/reference changes](../../review-checkpoints/2026-10-14/parallel-attention-contracts/README.md).
