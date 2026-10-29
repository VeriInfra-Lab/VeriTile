"""Render the two active progress documents; old full texts are exact archives."""
from common import SPEC,require

def render_documents(original,acceptance):
 checkpoint='review-checkpoints/'+SPEC['intended_checkpoint']
 current='['+SPEC['intended_checkpoint']+']('+checkpoint+'/README.md)'
 # Retain the existing precise acceptance/checking policy, not obsolete progress claims.
 begin=original.index('The [finite acceptance boundary]')
 end=original.index('The initial exact numerical adapter')
 policy=original[begin:end]
 scope='''# Original-program scope review

The requested standard covers **all declared stages, dtypes and branches of
183 original programs**. `fp4_to_bf16_conversion` is excluded because its existing
contract covers a repaired candidate; the separate `fp4_to_bf16` original task
remains in the requested set.

The active [`original_program_scope.json.gz`](original_program_scope.json.gz)
records **8 source scopes accepted and 175 pending**. `matrix_transpose` is
verified under explicit primitive contracts. `quantize_global`, `vector_addition`,
`quant_transpose_kernel`, `triton_mul2`, `relu_triton_kernel`,
`vector_addition_custom` and `add_example` are verified with exactly registered
numerical deferrals. Unintegrated producer candidates do not change these counts.

'''+policy+'''The current plain JSON numerical queue has **239 rules, 2,366 exact bindings,
22,317 cases and 12 declarations**. Empty, all-masked and rejected cases require
no unused numerical callbacks. Prior 153 rules, 1,642 exact bindings and eight
callback tables remain unchanged. New numerical relations remain unvalidated.

The current catalog contains 184 tasks, 961 public declarations, 3,666 task
modules and 4,097 registered modules. Its supplement contains 68 declarations:
62 execution and six protocol contracts. These inventory counts are separate
from the 8/183 source acceptance count.

The workflow now uses the '''+current+''' catalog and supplement:

- [Current catalog]('''+checkpoint+'''/current-source-catalog.json.gz)
- [Current supplement]('''+checkpoint+'''/catalog-supplement.json.gz)
- [Combined source review]('''+checkpoint+'''/root-review.json.gz)
- [Integration checks and retained attempts]('''+checkpoint+'''/README.md)

The original metadata baseline and preceding final-source manifest stay frozen.
The global audit checks exact catalog reproduction, whole-task structural
coverage, source-scope reviews, legacy selected sources, value metadata, numerical
bindings, comparator preflight and library replay. Its scope-completeness gate
continues to fail for the 175 pending tasks. Build, test and comparator results
are recorded from actual runs in the current checkpoint; this progress document
makes no independent build, comparator, GPU or numerical-conformance claim.

```sh
python3 -B bench/check_original_program_scope.py \\
  --root . \\
  --catalog bench/'''+checkpoint+'''/current-source-catalog.json.gz \\
  --ledger bench/original_program_scope.json.gz \\
  --output /tmp/original-scope-result.json
```

## Historical progress snapshots

The [previous scope progress document]('''+checkpoint+'''/evidence/baseline-progress.md)
and [previous development summary]('''+checkpoint+'''/evidence/baseline-current-source-acceptance.md)
retain their exact original bytes. Their older counts, pending conclusions and
catalog paths describe those historical snapshots. Earlier bounded vector/ReLU
contracts, scope reopenings, withdrawn quant-transpose recommendations, partial
extensions and numerical registrations remain in their frozen checkpoints.
The new reviewed contracts supersede the older pending conclusions for the
four tasks accepted in this batch; historical evidence itself is unchanged.
'''
 summary='''# Current development and acceptance

The requested scope is **183 original programs, covering all declared stages,
dtypes and branches**. **8 source scopes are accepted and 175 remain pending.**
`matrix_transpose` is verified under explicit primitive contracts. Seven tasks
are verified with exactly registered numerical deferrals: `quantize_global`,
`vector_addition`, `quant_transpose_kernel`, `triton_mul2`, `relu_triton_kernel`,
`vector_addition_custom` and `add_example`.

The active [scope ledger](original_program_scope.json.gz) and
[acceptance policy and current scope review](original_program_scope.md) define
that count. `fp4_to_bf16_conversion` remains the separate repaired-candidate
entry. Numerical deferrals do not waive missing source, host, dtype, branch,
address, memory or failure-order coverage. Producer candidates awaiting
integration do not count as accepted programs.

The current cumulative integration is '''+current+'''. Its
[current catalog]('''+checkpoint+'''/current-source-catalog.json.gz)
contains **184 tasks, 961 public declarations, 3,666 task modules and 4,097
registered modules**. The
[supplement]('''+checkpoint+'''/catalog-supplement.json.gz)
has 62 execution and six protocol declarations. These are proof-inventory
counts, separate from original-program acceptance.

The [plain numerical queue](tritonbench_g/deferred_numerical_experiments.json)
contains 239 rules and 2,366 exact bindings across 22,317 cases in 12 declarations.
The queue remains fail-closed against the reviewed source-derived authority;
these registrations do not establish native scalar conformance. Existing
refutations and all numerical-validation obligations remain preserved.

The workflow selects the current explicit `.json.gz` catalog, supplement and
active scope ledger. It retains the unchanged historical metadata baseline and
final-source review manifest. The current checkpoint's
[validation and attempt records]('''+checkpoint+'''/README.md)
report actual builds, tests and global gates. The 175 pending scopes remain a
completeness failure. Historical comparator successes apply only to their
pinned trees; no current comparator, whole-corpus, GPU or numerical result is
inferred from source acceptance or inventory counts.

## Historical progress snapshots

The [previous development summary]('''+checkpoint+'''/evidence/baseline-current-source-acceptance.md)
and [previous scope progress]('''+checkpoint+'''/evidence/baseline-progress.md)
are preserved byte-for-byte. Their 0/183 or 4/183 summaries, host-and-LoRA catalog,
previous build/test counts, partial-scope tables and historical comparator
results describe earlier states. Their proof evidence and checkpoint links
remain available; the current summary above supersedes their progress claims.
'''
 return {'bench/original_program_scope.md':scope,'bench/current_source_acceptance.md':summary}
