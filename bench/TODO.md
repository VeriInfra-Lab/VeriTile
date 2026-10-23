# TritonBench-G 后续事项

- [ ] **证明和验收完成后，再统一清理历史文件与重复实现。**
  - 当前优先推进证明和验收；只做推进任务必需的整理，不开展批量清理。
  - 保留并复用仍适用的旧定理，明确其前提和结论；将辅助证明归入 `Proofs/`，统一当前公开入口。
  - 后续核对引用与构建依赖，再处理重复版本、不再使用的兼容文件、过期检查点及临时产物。
  - 必要历史证据先确认已在 Git 或归档中保存；已证明的有效结果不得因目录整理或验收范围扩大而丢失。
  - 清理后检查相关构建、验收引用和工作树状态，单独提交清理变更。

记录：2026-10-17，用户确认“先完成证明和验收，后续再清理”。
关联工作聊天：`01a14539-f72c-7c83-aa71-3a0014d6e60a`。

- [ ] **程序证明主线完成后，统一处理 Triton 启动器的大 grid 乘积溢出兼容性。**
  - 本轮安排（2026-10-18）：将这个共享运行时问题后置，不要求各题现在重复开发兼容方案。
  - 已核对当前安装的 Triton 3.7.1：`triton/backends/nvidia/driver.c` 使用有符号 `int` 计算 `gridX * gridY * gridZ > 0`；特定大 grid 会使该乘法溢出。现有审查未实测该大尺寸的 GPU 运行结果。
  - 保留已有条件性程序证明的前提、范围和完成记录。该运行时适用性单独登记；受影响配置在问题解决并验证前，不宣称已获得原生运行正确性保证。
  - 后续集中评估启动器修复或经核实的修复版本；需要兼容未修复运行时时，再复用适用的分段启动候选。核对空 grid、普通 grid 和溢出边界，以及分段覆盖、偏移和执行依赖。
  - 本安排不延期题目源码自身的索引、stride、mask、偏移溢出或输出覆盖缺陷，也不自动改写另一工作聊天的验收记录。

- [ ] **集中核对 wrapper 的原生单轴 grid 上限。**
  - 2026-10-21 复查发现：`l2_norm_bwd` 的现有输入检查允许形状 `[2**31, 1]`，wrapper 随后提交 `grid=(2**31,)`，超过 CUDA `grid.x <= 2**31-1`。这是单轴启动边界，与上一项启动器内部的 grid 乘积溢出不同。
  - 本轮通过提取实际 wrapper AST、代入仅含元数据的对象捕获该 grid；未分配大张量、未运行 GPU。证据：`.veritile-cache/final-standard-review-20261021/group08-l2-grid-capture.json`。
  - 后续统一决定合理的 API 上界或分段启动方案。现有逐行 kernel 定理的数学范围不变；本项不是 numerical assumption，不计作该 kernel 证明缺失，也不宣称该超界配置已能原生运行。

## Kernel 源码入口整理

- [x] **处理 2026-10-22 复查确认的 92 题源码组织项。**
  已移除单 kernel 外层重建，将实际源码移出 `Proofs/`、将程序分解助手归入 `Proofs/`；保留透明共享入口与真实多阶段组合。共享 DSL 必要时用 `tl.ports inputs=[...], outputs=[...]` 保留完整有序端口，不改变计算或安全前提。92 题完整选定程序保持性与针对性源码构建通过，92 个源码入口联合导入通过；218 个选定及共享源码文件复扫，本轮所查模式剩余 0。修复前位置保留在[历史扫描](review-checkpoints/2026-10-22/direct-triton-scan/README.md)，最终证据见[修复报告](review-checkpoints/2026-10-22/direct-triton-fix/README.md)，状态只维护在 `tritonbench_g/task_progress.json`。
  90 题当前 `Correct`／`FPEquiv` 整文件 fresh 检查通过；`chunk_gla_fwd` 五个主声明逐字源码检查通过，附带审计主动停止、未宣称完成；`int_scaled_matmul` 的 fresh `FPEquiv` 保留既有 `Algebra.minimum` ABI 限制，完整源码保持和既有适用主证明证据分别记录。整题正确性验收及数值范围不变，没有新增 host、硬件或数值实验要求。

## 主入口公式、条件与范围

- [x] **2026-10-22 逐题检查全部 183 个选定主入口。**
  168 题调整数学展示、域记号或说明，15 题保留；367 个 reader、449 个主声明的
  真实阶段和分支保留。辅助定理不留在 reader；必要的域、观察及调度条件不因统一
  记号而删减。规则已写入 `AGENTS.md` 和 `MAIN_THEOREM_CONVENTIONS.md`；
  逐题目标、条件、完整范围及证据见[主入口检查](review-checkpoints/2026-10-22/main-formulas/README.md)。
  完成统计和数值延期不变；持续检查使用 `python bench/check_main_formula_review.py`。
- [x] **修正抽查指出的 `mean_reduction` 入口可读性问题。**
  改为逻辑矩阵 `xs row col`，由 IO 携带原完整调度，覆盖前提仍公开；输入、输出
  坐标双射及新旧完整判断等价均有证明。此前可读性结论过宽，本次只定点修正该题，
  不声称全部 183 题已按这一细化标准复核。见[修正报告](review-checkpoints/2026-10-22/mean-logical-io/README.md)。
- [ ] **后续统一整理共享构建代的重放与跨题兼容性。**
  本轮保留 `chunk_gla_fwd` 附带长审计、`int_scaled_matmul` 当前统一缓存 ABI、
  `kv_cache_filling` 选定归档的深 FP 审计三项具体限制，以及额外跨题总导入的旧生成
  声明冲突。逐题已证成果、精确声明检查和适用历史证书分别记录；不把旧缓存可导入
  当作当前声明已验证，也不将这些构建限制归为 numerical assumption。
  本轮不扩展成重编全部传递依赖或逐题硬件验证，详细边界见上述报告。

## Input-contract review and scope control

Recorded on 2026-10-18 at the user's request. Related working chat:
`01a14539-f72c-7c83-aa71-3a0014d6e60a`.

- [ ] **Review and consolidate input assumptions after the current proof work.**
  For each task, distinguish the existing documented or checked API contract,
  the intended supported use, assumptions required by the theorem, and proposed
  new wrapper guards. Record shape, dtype, layout/stride, storage offset,
  alignment, aliasing, logical view flags, and integer-range requirements only
  where relevant. Python accepting an object is not by itself a promise to
  support every layout or view that object can represent.
- [ ] **Use `add_example` as the first contract-review example.**
  Prefer an explicit same-shape, same-device, contiguous FP32 CUDA contract over
  unconditional full input copies solely to support arbitrary strides. Review
  empty inputs, kernel integer bounds, and unresolved logical negation
  separately; contiguity alone does not settle them. Make any newly required
  guard explicit in the API and theorem. The original wrapper did not check
  contiguity, so a revised contract must not retroactively certify the unchanged
  original program. General-stride support and a faster general implementation
  are optional extensions, not prerequisites for this narrower contract.
  A reviewable source revision now lives at
  `bench/repairs/tritonbench_g/add_example/add_example_repaired.py` in this
  workspace: contiguous FP32 inputs, no unresolved negative view bits,
  `numel < 2**31`, checked 4-byte pointer alignment for nonempty inputs,
  no input copies, and one nonempty launch. Empty inputs have no pointer-alignment
  requirement. CPU byte-offset storage cases motivate the alignment guard;
  their CUDA construction and execution have not been tested. The 2026-10-19
  `LOAD_STORE_AUDIT.md` records the PTX 8.7 alignment rule and Triton 3.7.1
  scalar `.b32` lowering on SM80/SM90, establishing the guard's instruction-level
  basis without a GPU run. Its exact guard/domain
  equivalence, guarded Real/FP32 logical-grid theorems, physical pointer/readback
  binding and returned-Tensor source theorem now pass Lean under ordinary Tensor,
  allocator and successful-runtime contracts. Numerical GPU validation and
  replacement of the other checkout's accepted repair remain pending. Existing
  acceptance counts are unchanged.
- [ ] **Audit other repair candidates for unnecessary generalization.**
  Identify copies, generic wrappers, layout/dtype expansion, cross-device
  support, and runtime compatibility work introduced beyond the intended API.
  Review a minimal contract or source repair for each case. Preserve applicable
  old proofs and exact historical acceptance records; this TODO does not itself
  change any candidate, theorem, numerical status, or completion count.

### Scope for subsequent tasks, effective immediately

The user clarified on 2026-10-19: **the required proof object is the kernel;
the Python wrapper supplies input assertions.** `Correct.lean` exposes the
kernel's independent mathematical specification and correctness proof;
`FPEquiv.lean` exposes equivalence of the displayed kernel versions. Keep
assertions and the declared kernel input contract consistent, including the
shared connection from admitted layouts to memory meaning. Do not add wrapper
execution, allocator behavior, device-context orchestration, grid submission,
or returned-Tensor identity as per-task acceptance requirements. Kernel index
arithmetic, masks, load/store conditions and the specified output coverage
remain required under the declared launch geometry. Previously proved host
results remain available as optional helpers; their absence from the public
kernel entries is an intentional scope change, not a failed kernel proof.
Apply this boundary as tasks are edited; do not duplicate historical evidence.
`add_value` was the first updated entry. The subsequent explicit request to
check all selected public formats is an interface audit, not a new prerequisite
audit of every source-correctness proof.

The earlier 2026-10-20 strict split separated raw kernel results from Tensor
applications. Those proofs remain valid, but that split did not complete the
requested public IO format. The current pass checks all 183 selected task
entries. `Correct.lean` and `FPEquiv.lean` expose shared IO judgments and
independent Real formulas or actual two-program `≡[R]` equivalence;
raw State/run/readback/frame assembly needed by those results belongs in
`Proofs/`. Optional single-source native observations, Tensor/host applications
and audit readers belong in `Misc/`, including `Misc/NativeCorrect.lean` and
`Misc/TensorBridge.lean`. They are not additional FP32-correctness or host-proof
acceptance requirements.
Shared task `IO.lean` data shows actual source/ABI, channels, masks and full
schedules. Preserve old input domains, outputs and physical frame results; do
not discard an existing full-grid result when presenting kernel-stage pairs.
New candidate whole-grid composition is not required beyond the agreed kernel
scope. Source-access
safety must be proved, not hidden in an IO-name alias or numerical assumption.
The single task checklist separately counts IO-format completion and prior
correctness acceptance, with concrete per-task adapter gaps. No source/contract
expansion or numerical experiments are added by this pass. See the
[file responsibilities](repairs/README.md) and the single task checklist.

Shared implementation note for later focused maintenance: `ProfiledRewrite`
and `ProfiledMemory` currently obtain program signatures through
`ComputeKernel.inputs/outputs`, which project to Algorithm syntax. Direct `.mk`
ports avoid costly projection and its unsupported-syntax fallback. Uniform's
attempted signature proof exposed this cost; its IO work uses actual source
schedule equivalence instead. This is a shared signature issue, not a numerical
assumption or a reason to change benchmark sources.

The current `cross_entropy1` and `cross_entropy_ops` checked revisions select
the default single-process API and retain local class splitting. Their unchanged
conditional distributed sources and peer-coherence theorems remain available
separately. A checked
multi-process admission protocol is an explicit later contract-review item,
not a numerical deferral and not a claim of checked distributed completion.
`softmax_flaggems` likewise explicitly requires contiguous upstream gradients;
support for broadcast/autograd strided gradients is outside its current checked
domain. These scope restrictions are visible in the single task checklist.

Continue the source repairs and mathematical/source-precision proofs needed for
each task's explicit supported input contract. First state that contract and
the concrete missing result, then implement the smallest justified repair and
reuse applicable proofs.

- Do not expand a task to arbitrary strides, aliases, dtypes, devices, tensor
  view semantics, API redesign, or general library/runtime verification unless
  the intended contract or a concrete in-scope failure requires it. Prefer an
  explicit justified precondition and an appropriate guard to a generic
  compatibility layer where the API permits that choice.
- Do not add assumptions merely to exclude a counterexample inside the agreed
  domain, assume the desired output, or make the premises inconsistent. Any
  genuinely unresolved API-scope choice is a recorded contract-review item;
  retain conditional results with their exact premises and continue work that
  does not depend on that choice.
- Record unvalidated numerical transformation rules, their precision,
  applicability domains, and required evidence for the later shared experiment
  batch: for example, reassociation, reduction-order changes, division to
  reciprocal multiplication, and exp/log rewrites. Basic operation semantics,
  dtype, rounding and FTZ belong to the semantic model and its documented
  implementation boundary; do not turn native-to-software operation conformance
  into a per-task numerical experiment. Addressing, masks, output coverage, and
  contradictory model interfaces are structural obligations, not numerical
  deferrals. A rule's finite-operand domain is checked at its rewrite site; it
  does not add a Python input guard to an otherwise valid source program.
- The refined 2026-10-21 FP rule policy preserves exact model proofs and separates
  them from empirical admissions. Concrete models and their strict lemmas may
  remain in main results; arbitrary opaque-interface parameterization is optional.
  Extract generic scalar laws to the shared numerical library and reuse them.
  Do not replace an existing strict proof with an experiment or equate an
  experiment's acceptance with universal IEEE equality. The single task ledger
  records actual rule-evidence gaps separately from numerical deferrals; the
  former 43 concrete-model dependencies are not mandatory refactoring tasks.
  See the [rule rescan](review-checkpoints/2026-10-21/fp-rule-rescan/README.md).
- Keep the previously deferred shared Triton launcher compatibility work and
  historical cleanup separate. Within the supported source contract, retain
  the necessary kernel index, pointer, and output-correctness obligations.
- Reuse existing build, axiom, source-binding, and acceptance checks. Add or
  repeat checks when a concrete change or unresolved failure warrants them;
  avoid per-task proliferation of wrappers, approval layers, and unchanged
  evidence scans. Necessary proof-validity checks remain required.
- Measure progress by accepted in-scope program/proof results. Installation,
  receipt generation, wrapper preparation, and optional support expansion do
  not by themselves complete a task. Already admitted rules are not pending
  experiments. Track pending experiments for auxiliary results separately;
  do not report native conformance or wider API coverage without evidence.

### Current numerical experiment queue (2026-10-21)

- The selected 183 primary kernel results and 19 examples have no identified
  missing numerical admission. Their existing rule premises remain explicit;
  the current rule pool is distinct from the instances actually used by proofs.
- No auxiliary numerical experiment remains in the current queue. The optional
  `apply_penalty` compiler-native equivalence and its dedicated experiment were
  removed at the user's request; the source-level primary equivalence retains
  its existing admitted rule.
- The former softmax operand-coverage TODO is removed: correlation alone does
  not require another experiment for an already applicable admitted rule.
  Whole-kernel GPU runs, proof replay and historical native-conformance work
  remain separate; none was performed by this cleanup.

This file records the user's scope decision. It does not itself update the
working chat's `/goal`; the revised objective must be applied in that chat.

### Focused launch-device binding audit (2026-10-19)

- [x] Complete the focused audit of accepted caller-tensor wrappers lacking an
  effective owning-device context or executed current-device precondition.
  Corrected fourteen tasks: `fused_recurrent_delta`, `multinomial_sampling`,
  `spinning_lock_reduction`, `attention_forward_triton`, `attention_fwd_triton1`,
  `attention_fwd_triton2`, `attention_llama`, `attn_fwd_triton`, `fp4_to_bf16`,
  `layer_norm_liger`, `matmul_kernel`, `matmul_leakyrelu`,
  `matmul_persistent_triton`, and `matmul_triton_autotune`.
  Observed in installed Triton 3.7.1's `triton.runtime.jit.JITFunction.run`: it calls
  `driver.active.get_current_device()`, then
  `driver.active.get_current_stream(device)`, selects
  `self.device_caches[device]`, and passes that stream to `kernel.run`.
  Equal tensor devices alone therefore do not select the launch device.
  Each actual submission now selects its owning tensor's device and restores
  the caller's device. Delegated launch sites were checked before correction.
  Excluded existing contexts: the preceding `fused_recurrent_retention` correction;
  `_DeviceGuard` callers `fifth_order_sph_harmonics`, `isfinite_kernel`,
  `pow_scalar_tensor`, and `relu_strided_buffer`; and the explicit device-0
  guards/selection in `fused_layernorm_triton` and `layer_norm_welfold`.
  `cross_entropy_ops` delegates to a guarded source. The subsequently accepted
  `chunk_gate_recurrence` has an executed current-device precondition.
  Internal-allocation-only `batched_vecmat_mult`, `matrix_reduction`,
  `matrix_transpose`, `nested_loops_processing`, and `matmul_tma` did not meet
  this caller-input mismatch trigger. Other preexisting device-context callers
  are not enumerated here; this was not a repository-wide runtime audit.
  Targeted CPU suites and affected Lean builds passed; logical cuda:1 / ambient
  cuda:0 controls check actual submission and restoration on success and failure.
  CUDA contexts were mocked: no GPU/runtime or numerical conformance is claimed.
  Kernel ASTs and numerical graphs are unchanged. Exact old/current source hashes,
  scope and evidence are in [the durable audit note](../documents/LaunchDeviceAudit.md).
  This completion does not change historical acceptance counts. The shared
  launcher grid-product compatibility TODO above remains deferred and open.

### Cross-task Lean import compatibility (2026-10-20)

- [ ] Resolve the duplicate automatically generated
  `VeriTile.Triton.FP.Structural.evalExpr.eq_4` declaration when the unrelated
  `bench.examples.SoftmaxReciprocal.FPEquiv` and historical SGMV
  `Proofs.SourcePrelude` closures are imported into one Lean environment.
  The SGMV public kernel closure builds independently; its local byte-lemma
  extraction avoids an unnecessary dependency but does not repair this shared
  generated-name collision. Preserve current kernel proof receipts and diagnose
  shared source/toolchain/cache compatibility separately. This is neither a
  numerical experiment nor evidence of a false kernel result. Do not claim the
  combined import succeeds: its failed check and the subsequent actual-target
  checks are retained in `.veritile-cache/kernel-batch028/`.
  Rechecked on 2026-10-21 against the current frozen IO-format objects: the
  same pair still fails on `Structural.evalExpr.eq_4` (the other owner is
  `VeriTile.Triton.Float.Reciprocal`). This remains open. The 183 successful
  per-task public-entry checks do not claim compatibility with this example
  and historical-proof union. Current evidence:
  `.veritile-cache/public-io-format-20261020/post-acceptance-review-20261021/softmax_sgmv.log`.
- [x] Resolve the additional generated-name collision
  `VeriTile.Triton.FP.Structural.Algebra.withDefaultPrecision.eq_1` between
  `chunk_gate_recurrence.Proofs.KernelPrefix` and
  `context_attn_fwd.Proofs.FPModel` in the batch031 combined import.
  The exact two-module union passed a fresh import check on 2026-10-21 using
  the current frozen objects. Current evidence:
  `.veritile-cache/public-io-format-20261020/post-acceptance-review-20261021/chunk_context.log`.
  Preserve the earlier failed union and separately scoped public-target checks
  in `.veritile-cache/kernel-batch031/` as historical evidence. This closes
  that specific import conflict; it is neither an independent proof replay nor
  a numerical experiment.
- [ ] Finish shared-generation import compatibility after the remaining-105
  campaign. All 183 tasks have explicit individual acceptance (182 complete,
  one with deferred numerical evidence), but the attempted 182-task union in
  the isolated minimumFloat/current-Welford generation fails when
  `attention_fwd_triton1_tile_checked.Proofs.Step` and
  `VeriTile.Triton.Float.Reciprocal` introduce the same generated
  `VeriTile.Triton.FP.Structural.evalExpr.eq_6` declaration. Preserve the actual
  failure in
  `.veritile-cache/finish-pending-105-20261021/combined-minimum-welford-20261022/all-main-182-final-joint.log`.
  This is a shared import-compatibility issue, not a numerical deferral or a
  failed individual kernel theorem. Required task-helper refreshes are recorded
  separately in that generation's manifest; in particular, the integer-dot
  helper used by `int_scaled_matmul/Proofs/CandidateUnscaled.lean` must preserve
  the new `Algebra.minimum` field when rebuilding in this generation. Its
  individually accepted strict generation is retained. This union is not a
  source replay.
- [ ] Consolidate the selected `kv_cache_filling` minimum/reduceMin generation
  with the current shared generation as a separate integration task. The new
  source-only selection passed all 47 necessary module builds, complete-source
  preservation and fresh public/optional joint checks with its explicit driver
  at `.veritile-cache/finish-pending-105-20261021/source-only/kv_cache_filling/build.py`.
  The current combined generation lacks its `Op.reduceMin`, `Algebra.reduceMin`
  and selected `tl.min` promotion. Its attempted 183-task import therefore cannot
  substitute for that independent selected-generation check. Preserve both
  generations and their stated scopes; do not count this as missing numerical
  evidence or silently overlay a different shared model.

### Optional-reader import compatibility found during IO review (2026-10-22)

- [x] Resolve the duplicate aliases
  `VeriTile.Bench.Repairs.ChunkBwdDqkg.Reader.realApplication` and
  `nativeApplication` in `chunk_bwd_dqkg/Misc/Reader.lean` and
  `Misc/HostReader.lean`. The host aliases now use
  `ChunkBwdDqkg.HostReader.realApplication/nativeApplication`, while the Tensor
  reader names remain unchanged. The fresh primary plus all task `Misc/` import
  now passes, retaining both reader interfaces and the prior complete-IO `rfl`
  checks. Historical failure evidence remains in
  `.veritile-cache/io-consolidation-20261022/queue07/chunk_bwd_dqkg/`.
- [x] Resolve the name collision
  `VeriTile.Bench.Repairs.MatmulKernel.correct` between
  `matmul_kernel/Correct.lean` and `Misc/Legacy/HostCorrect.lean` before claiming
  their joint import. The host result now uses
  `MatmulKernel.Legacy.HostCorrect.correct`. Removing the first collision also
  exposed a duplicate `MatmulKernel.fp_equiv` in the native and legacy host
  entries; the host result now uses `MatmulKernel.Legacy.HostFPEquiv.fp_equiv`.
  The fresh primary plus all task `Misc/` import, including `CandidateCorrect`,
  passes. Historical first-collision evidence remains in
  `.veritile-cache/io-consolidation-20261022/queue11/logs/matmul_kernel/`.

The subsequent namespace fix closes these two task-local compatibility items;
see [the source changes and successful joint checks](review-checkpoints/2026-10-22/misc-import-fix/README.md).
Kernel acceptance and numerical evidence are unchanged. The shared-generation
whole-corpus compatibility items above remain open.
