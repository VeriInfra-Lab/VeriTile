"""Keep primary-version goals distinct from the cumulative legacy source ledger."""
from common import SPEC, require

def render_documents(original, acceptance):
 cp='review-checkpoints/'+SPEC['intended_checkpoint']
 policy='review-checkpoints/'+SPEC['predecessor']+'/primary-version-policy.md'
 begin=original.index('The [finite acceptance boundary]')
 tail=original[begin:];end=tail.index('\nThe current ')
 retained_policy=tail[:end]
 lead=f'''The current target is **Real, FP32 and BF16 for 183 original programs, plus
source-required types**, under the approved [primary-version policy]({policy}).
Existing wider proofs remain available. This integration changes the legacy
full-source ledger; it does not establish completion of the three primary versions.
The [new three-task applicability inventory]({cp}/primary-version-applicability.md)
records source-case evidence and leaves version completion pending.

The active [scope ledger](original_program_scope.json.gz) records **11 legacy
full-source scopes accepted and 172 pending**. `matrix_transpose` is verified
under explicit primitive contracts. Ten tasks retain registered numerical
deferrals: `quantize_global`, `vector_addition`, `quant_transpose_kernel`,
`triton_mul2`, `relu_triton_kernel`, `vector_addition_custom`, `add_example`,
`add_value`, `sin_kernel` and `cosine_compute`. `fp4_to_bf16_conversion` remains
the separate repaired candidate; the original `fp4_to_bf16` stays in the requested set.
'''
 inventory=f'''The [compressed numerical queue](tritonbench_g/deferred_numerical_experiments.json.gz)
contains **273 rules, 2,943 exact bindings, 22,875 cases and 15 declarations**.
The prior 239 rules, 2,366 bindings and twelve callback tables retain their exact
contents. Every numerical port remains bound to its actual source case. In
particular, cosine host arithmetic has obligations on post-allocation paths;
empty or masked labels do not generically erase host numerical requirements.
Native scalar conformance remains deferred.

The [current catalog]({cp}/current-source-catalog.json.gz) contains 184 tasks,
964 public declarations, 3,690 task modules and 4,121 registered modules.
The [supplement]({cp}/catalog-supplement.json.gz) contains 65 execution and six
protocol contracts. These are inventory counts, separate from primary-version
completion and legacy acceptance.

The workflow uses the current gzip catalog, supplement and active ledger.
The [combined source review]({cp}/root-review.json.gz) and
[recorded integration attempts]({cp}/README.md) retain exact evidence.
The global scope-completeness gate still reports 172 pending tasks. Build,
test, comparator and numerical outcomes are reported only from actual runs;
no GPU or whole-corpus success follows from these counts.
'''
 history=f'''The [previous scope document]({cp}/evidence/baseline-progress.md) and
[previous development summary]({cp}/evidence/baseline-current-source-acceptance.md)
are archived byte-for-byte. All earlier checkpoints, wider proofs, reopenings,
numerical registrations and failure records are retained. This batch authorizes
no historical cleanup or deletion.
'''
 scope='# Original-program scope review\n\n'+lead+'\n'+retained_policy+'\n\n'+inventory+'\n'+history
 summary='# Current development and acceptance\n\n'+lead+'\n'+inventory+'\n'+history
 return {'bench/original_program_scope.md':scope,'bench/current_source_acceptance.md':summary}
