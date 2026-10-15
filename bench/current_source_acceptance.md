# Current development and acceptance

The current goal is **183 original programs, covering all declared stages,
dtypes and branches**. Unverified numerical assumptions may be recorded for
later validation. Missing source translations, host composition, dtype or
branch coverage, address semantics and known source bugs are not numerical
deferrals. `fp4_to_bf16_conversion` remains the separate repaired-candidate task.

**Whole-program acceptance in this review round: 0/183; 183 remain pending.**
This does not invalidate the existing proofs. The earlier 183 original-source
contracts cover their stated profiles, which are narrower than the newly
requested whole-program standard. The old “six remaining” count concerned
software closure in the original 41-task batch, not these 183 whole programs.

The [current cumulative extension batch](review-checkpoints/2026-10-15/host-and-lora-scope-extensions/README.md)
records 13 public execution contracts across seven tasks. The first eleven were
committed in `93d38e39`; the two new entries add LoRA branches and matrix-transpose
logical host composition:

| Task | Newly proved scope | Still prevents whole-task acceptance |
|---|---|---|
| `fp4_to_bf16` | Packed uint8 input, complete BLOCK=512 grid, output shape and exact bf16 values | Unrestricted caller dtype/error domain; native allocation/address and concurrency correspondence |
| `multinomial_sampling` | Four score formats, complete grid, signed64-index wrapper and token-one return | Other supplied-index/caller dtypes; full error/native domain |
| `kcache_copy_triton` | Arbitrary-token signed copy, five floating raw formats and nine discrete formats | Host reshape/layout/grid composition; native metadata/address domain |
| `triton_conv2d_fwd` | Both fp16 flags and both tf32 flags, complete triple loop over fp32 storage | Other storage dtypes; host grid; signed/native address correspondence |
| `attention_kernel_aligned` | Arbitrary lengths and independent Nat strides in fp16; complete scatter/frame | Original bf16 compile failure, unchecked tails and head-stride bugs; host/native composition |
| `lora_expand_gemv` | Six input/weight pairs × three outputs, all flags, sentinel and complete output loop; exact current-memory reference/frame | Other output/index dtypes; host/grid; native shape, signed addresses and concurrency |
| `matrix_transpose` | Original fp16 logical allocation and single program; exact contiguous strides including empty shapes, transpose bits and full frame | Caller/native rejection, returned Tensor metadata, allocation/ABI and device correspondence |

The first batch passed the full registered Lean build with **7,231 incremental
jobs**, 210 benchmark-tool tests and 25 audit-sharding tests; its
[validation record](review-checkpoints/2026-10-15/original-scope-extensions/validation.json)
retains the actual gate results. The two additional contracts passed independent
source/mathematical reviews and the complete registered integration build with
**7,244 jobs**; the updated workflow passed 25 sharding tests. Their
[validation record](review-checkpoints/2026-10-15/host-and-lora-scope-extensions/validation.json)
records the current source/metadata gates and the expected scope/comparator
failures. The earlier 210 benchmark-tool tests were not rerun in this batch;
their tool and test sources are unchanged.

The isolated official library comparator passed **118/118 targets in eight
modules** against the exact 7,177-file Git tree of `93d38e39`; its
[original result and complete evidence](review-checkpoints/2026-10-15/comparator-original-scope-library/README.md)
are retained. The separate
targeted replay of that commit's six new public files has no final result yet. Neither
run includes the thirteen modules in this second batch; this library result is
not a full current-corpus or whole-program acceptance receipt.

The [scope ledger](original_program_scope.md) tracks the new acceptance standard.
Pending rows deliberately fail its gate. Current catalog reproduction, source
checks, compiled proof replay, numerical bindings and native behavior remain
separate checks; a green inventory alone does not establish completion.
The [current catalog](review-checkpoints/2026-10-15/host-and-lora-scope-extensions/current-source-catalog.json)
contains 184 tasks, 906 public declarations, 3,452 task modules and 3,883
registered modules, with no unreviewed source bindings. These are inventory
counts, not the whole-program completion count.

The [preceding frozen catalog](review-checkpoints/2026-10-15/retention-query-source-proof/current-source-catalog.json)
records 184 tasks, 893 public declarations and 3,840 registered modules before
this batch. Its 58 software-bit, 109 relative-equivalence, 16 backend-value and
one candidate profiles retain their exact scope. Historical comparator
[library](review-checkpoints/2026-10-15/comparator-canonical-library/README.md)
and [corpus snapshot](review-checkpoints/2026-10-15/comparator-storage-maintenance/README.md)
results apply to their pinned earlier sources; they are not current replay.

The [existing numerical queue](tritonbench_g/deferred_numerical_experiments.md)
and the current batch's [numerical follow-up](review-checkpoints/2026-10-15/host-and-lora-scope-extensions/numerical-followup.json) preserve unverified numerical work. The new LoRA entry specifies
rounding, conversions, reduction-tree/fusion and native configuration work.
These follow-ups are not acceptance queue scope bindings yet; matrix transpose
adds no numerical callback. The unrestricted fp16-rounding-is-identity rule is refuted and cannot be
approved by postponing experiments. The Conv2d storage annotation correction
is explicit and reviewed; frozen evidence is unchanged.
