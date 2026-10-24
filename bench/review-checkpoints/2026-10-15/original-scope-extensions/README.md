# Original-program scope extensions

This batch adds **11 public execution contracts across five tasks**, with the
exact declared profiles in [catalog-supplement.json](catalog-supplement.json)
and the source/statement review in [root-review.json](root-review.json).
It is partial progress toward the 183-program standard, not whole-task
acceptance. All 183 full-scope reviews remain pending.

| Task | Extension |
|---|---|
| `fp4_to_bf16` | Complete logical BLOCK=512 launch, nonscalar output shape, exact signed-nibble decoding and full frame for packed uint8 input |
| `multinomial_sampling` | Four score formats over the complete grid; signed64-index wrapper initialization and both return shapes, including early token-one zeros |
| `kcache_copy_triton` | Arbitrary-token signed source, all three coordinates, five floating raw payload formats and nine discrete formats |
| `triton_conv2d_fwd` | Original default `fp16=False` and retained `True`, both tf32 flags and complete KH/KW/channel loops over fp32 storage |
| `attention_kernel_aligned` | fp16 arbitrary length and individual Nat strides; exact unchecked tails and scatter/frame; original bf16 rejection regression |

Each task README gives its remaining structural scope. In particular, a
single-program theorem does not prove the Python host grid, allocation or
concurrent execution. The new FP4 and multinomial logical host proofs retain
their explicit storage and source-domain boundaries. LoRA branch work is not
included in this completed batch.

The Conv2d frozen registry incorrectly described fp16 input/weight/output
storage. The old source uses fp32 storage with binary16 *intermediate* operands
when `fp16=True`. The supplement carries a separately reviewed annotation
correction and leaves the frozen registry intact. The new false-branch proof
is a separate contract.

Attention's original bf16 dot output is rejected by the frontend and by offline
Triton 3.7.1/sm90 compilation. The fp16 source also retains unchecked partial
tiles and V/output head bases computed with other tensors' strides. The Lean
regressions and exact-wrapper CPU metadata probe record these source bugs;
neither compilation nor the probe executed a GPU kernel. They cannot be
classified as deferred numerical assumptions.

[numerical-followup.json](numerical-followup.json) records the exact source,
callback, dtype and configuration work still needing numerical validation.
It grants no acceptance: future `verified_with_deferred_numerics` rows must
also register matching bindings in the unified numerical queue.

Validation of the registered source set passed the full incremental build:

```sh
lake build VeriTile VeriTileMath VeriTileFull TritonBenchGVerified \
  TritonBenchGInProgress TritonBenchGLegacy TritonBenchSpecExamples \
  FPFrontendRegressions
```

The result was **7,231 jobs, exit 0**, including public axiom checks, module
audits and independent-reference checks in the new modules. Targeted proof
receipts and semantic reviews are retained under [evidence](evidence).
Full logs and exploratory failures remain in the local review archive under
`/tmp/veritile-standardization-20261015`; generated caches/build products are
not part of this checkpoint. Receipt file hashes preserve which local artifacts
were observed; not every original compiler artifact is retained here.

The complete benchmark-tool suite passed 210 tests; audit sharding passed
25 tests. The scope gate now checks original AST IDs, explicit dtype/config
cases, and registered callback/rule/statement bindings. Those checks prevent
profile substitution; substantive review still establishes that the stated
cases exhaust the original program and correspond to the actual theorem.

[validation.json](validation.json) binds the final source/tool hashes and the
observed validation logs. The retained [global receipt](evidence/global-gates/receipt.json)
passes source tools, structure, baseline, catalog reproduction, legacy source,
value metadata, deferred numerical records and unchanged-input checks. It
fails scope review (183 pending) and comparator gates (tool unavailable on that
runner). Those failures remain visible; this is not a successful full audit.

Reproduce the current global checks using a fresh output directory:

```sh
python3 -B bench/current_source_audit.py --root . \
  --catalog bench/review-checkpoints/2026-10-15/original-scope-extensions/current-source-catalog.json \
  --baseline-package bench/review-checkpoints/2026-10-15/original-scope-extensions/metadata-baseline \
  --source-map bench/current_source_selections.json \
  --legacy-inventory bench/tritonbench_g/current_completion_inventory.json \
  --evidence-root bench/review-checkpoints \
  --review-manifest bench/review-checkpoints/2026-10-15/raw-copy-software-metadata/catalog-review-rebased.json \
  --supplement bench/review-checkpoints/2026-10-15/original-scope-extensions/catalog-supplement.json \
  --scope-ledger bench/original_program_scope.json \
  --output /tmp/original-scope-global-replay
```

This command is expected to remain nonzero until all 183 scope reviews and the
official comparator gates pass. The value-metadata adapter replays the full
frozen checker in its pinned tree, then checks all current supplemental public
declarations; running the historical checker directly over new live public
files is not the current acceptance procedure.

The new [metadata baseline receipt](metadata-baseline/receipt.json) pins 3,750
files from commit `9327d49ade58b9895672d462e230fe686626ebd2`. It includes the
already reviewed Retention extension and all 893 pre-existing public claims.
Its unchanged complete checker passes in the archived tree. The preceding
raw-copy baseline predates that extension and cannot serve as the byte-for-byte
authority for the current live metadata. No old evidence is rewritten, and no
new public file is hidden from the current catalog. Retention's two contracts
now come from the baseline, so the supplement does not count them again.

The portable attention probes use fresh output directories:

```sh
python3 -B bench/review-checkpoints/2026-10-15/original-scope-extensions/replay_attention_source.py \
  --output /tmp/aligned-source-replay
python3 -B bench/review-checkpoints/2026-10-15/original-scope-extensions/replay_attention_compile.py \
  --output /tmp/aligned-compile-replay
```

The first requires Torch for CPU tensor metadata; the second requires Triton
and its offline CUDA compilation tools. Neither launches a GPU kernel. The
saved producer receipts identify the original scripts; these portable wrappers
preserve the original kernel and assert the recorded source/compile outcomes.

Independent official comparator replay of these new modules is still pending:
the recorded global run could not find the official comparator. A separate
isolated environment is being restored for a new replay. Historical comparator
receipts are not reused as current success. The full-scope gate also
intentionally fails while reviews are pending.
