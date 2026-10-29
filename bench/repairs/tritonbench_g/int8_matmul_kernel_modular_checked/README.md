# Explicit word-add int8 matmul candidate

Current FP entry (2026-10-21): `FPEquiv.configuration_equiv` compares two
actual retained autotune specializations. Each specialization executes the
complete source grid, with unchanged FP32-to-signed8 input conversion, packed
UINT8 loads, zero-seeded dot and explicit add.u32 accumulation. For example,
config0 uses a 128x256x64 tile while config13 uses 128x32x64, allowing fewer
inactive columns on narrow matrices. These are existing source specializations,
not a synthetic candidate. Both full schedules have proved safety, success,
the same dense signed low32 result and exact exterior frames. The experimental
rule set is empty; no no-overflow condition or narrower payload domain is added.
The public theorem allows every pair whose original K divisibility assertions
hold, so all fifteen retained configurations remain covered.

The original carry-fed dot and selected modular repair remain incompatible on
the documented config14 integer-recurrence witness. That historical program is
not used as this pair's comparison target, and its defect record is preserved.
The diagnostic does not claim a new GPU run. Optional source/native results
remain in `Misc/NativeCorrect.lean` as ordinary auxiliary theorems.

This separate candidate retains the original checked FP32/packed-UINT8 input domain, all 15 autotune configurations, static assertions, indexing, launch structure, and INT32 output. Its only arithmetic change is to compute each dot from zero and carry the result through explicit `add.u32` inline assembly. The wrapper docstring names the resulting signed low32 contract. The prior selected source and its satfinite counterexample remain unchanged.

The native target is the low 32 bits of the dense signed sum of the admitted FP32-to-signed8 conversions and unpacked two-bit weights. If that sum fits signed INT32, a separate corollary returns the exact signed sum. `idealRealTarget` is an independent arbitrary-Real specification; no equality between unrounded Real inputs and truncated/wrapped native output is asserted.

`Kernels.lean` now contains an ordinary direct `triton` quotation and has no
reverse import from `Proofs/`. Its one add.u32 primitive is expressed by the
already used strict unsigned32/signed32 word-add projection, with the complete
real instruction and ABI documented adjacent to the authoritative source.
The actual Python inline ASM barrier is unchanged. This is a semantic projection
of that instruction, not a new Python add/cast expression that the compiler
could contract. `Proofs/SourcePreservation.lean` proves complete AST equality to
the predecessor bounded expansion. The strict `wordAddExpr_eval` certificate
and existing overflow/rejection controls are retained. The generator/checker
checks the exact Python ASM ABI and every surrounding source AST statement;
no shared DSL/core engineering or hardware acceptance obligation is introduced.

The existing frontend emits the unique leading `tl.static_assert(K % (4 * BLOCK_SIZE_K) == 0)` as an algorithm metadata node that the structural evaluator cannot execute. `TileSource.compileAssertionProjection` replaces only that exact leading empty-false metadata node with its executable no-op form. The quoted tail, ports and statement count are preserved. True/nonempty conditionals are rejection controls. The source generator checks the unique assertion predicate, and `TileConfig.all_config_assertions` proves the original per-configuration admission obligation for every `c : Fin 15`. The public IO launch explicitly selects this projection; no second whole-kernel AST is maintained.

The proof follows every actual source statement through the prelude, both dynamic loops, FP32 loads and partial casts, UINT8 extraction, integer dot, explicit word addition, pointer advances, output masks and stores. FP32 cells use the frontend's real carrier with explicit FP32 precision and raw 32-bit storage words. The zero-seeded block bound is at most 32768 in absolute value for every prefix of at most 128 products, so all modeled K32 instruction families agree within a dot, including satfinite. Modular carry addition then gives the independent dense low32 target for all 15 configurations, including zero K.

`GridExecution.GridExec` contains actual complete `FP.Structural.exec` runs against evolving common memory. Output coverage and disjoint program writes connect these executions to the complete grid, including empty M/N. `StoreABI` proves scalar source/backing store commutation and that the final source memory equals the encoding of the final typed common backing. Public execution requires output storage distinct from A/B, supplied by the existing wrapper's `Allocation.separate`. This is the ordinary fresh-output allocation fact, not an added Python guard. The same backing supports output Tensor readback and an exact frame outside C's logical cells. Shared `Torch.FP32.Tensor` API laws connect A's logical values to those same input words; the existing task-local UINT8/INT32 descriptors retain storage identity, offsets and strides. Existing guard, formed-index, pointer and allocation-coverage facts are reused without a new host orchestration model. This is a typed storage model, not a new untyped byte-heap alias analysis.

Primary endpoints are `kernel_low32` in `Correct.lean` and the actual two-program `configuration_equiv` in `FPEquiv.lean`. The optional `native_kernel` theorem remains in `Misc/NativeCorrect.lean`. Both use the shared `ProgramIO.ImplementsOn` judgment over finite logical input/output arrays. `IO.lean` visibly lists the selected source, full grouped launch schedule, FP32 input, packed UINT8 input, signed INT32 output, arbitrary original input strides, no scratch, and unchanged logical cast/byte domain. It also states the independent dense low32 formula and the existing native primitive observation. The main statements contain no raw state or execution predicate. `Proofs/KernelResults.lean` retains the former raw complete-grid statements as ordinary helper theorems. The former complete Tensor results are retained as `TensorBridge.kernel_tensor_low32`, `native_tensor_lane` and `native_tensor_kernel`, with the same backing and every original observation. Optional ordinary corollaries `TensorBridge.kernel_torch_low32` and `TensorBridge.kernel_tensor_representable` are in `Misc/TensorBridge.lean`; neither primary entry imports that interface. Native correspondence is conditional only on primitive FP32-to-signed8, fixed K32 instruction, and two-word `add.u32` leaves. The source execution, casts, packing, loop order and stores are proved compositionally; no whole-kernel lowering equation or output equality is a premise. `Proofs/AccessSafety.lean` proves access safety along the actual evaluated prelude, nested loop iterations and final masked stores. `Proofs/PublicIO.lean` connects each reached address to the displayed logical channels; `Proofs/GridIO.lean` composes the prior actual per-program execution certificates into the complete shared invocation schedule, preserving empty M/N and zero K. `Proofs/NativeIO.lean` reuses the existing primitive correspondence. `Misc/Audit.lean` checks standard axioms and independent specification definitions; public statement audits reject operational detail. This format pass changes no selected Python, kernel, input contract, numerical law or experiment status.

Compiler evidence covers exactly two offline SM90 profiles using the real Triton binder and `_pack_args`, at M=N=1, K=16777728. Config 14 retains zero-seeded satfinite MMA plus eight distinct `add.u32` updates; config 4 retains WGMMA with scale-d zero plus 64 distinct updates. TTIR, TTGIR, LLVM IR and PTX are pinned. `binder-profiles.json` pins the checker bytes executed during compilation, retained as `compiler-driver-at-compilation.py`; the newer inspection-only checker and receipt explicitly record this identity mapping. These profiles support the arithmetic barrier for those specializations. They do not certify compilation/resources for all 15 configurations or constitute GPU execution. No new numerical experiment or historical control rerun was performed.

The preserved defect uses FP32 A=-128, packed byte 15, M=N=1 and K=16777728. Its dense result -1073774592 fits INT32, while the old config-14 satfinite result is -1073709056 (error 65536). Original config 4 wraps to the dense result. A final-sum-fit guard cannot fix that defect.

Inspection-only checks:

```bash
PYTHONDONTWRITEBYTECODE=1 python bench/repairs/tritonbench_g/int8_matmul_kernel_modular_checked/generate_kernel_source.py --check
PYTHONDONTWRITEBYTECODE=1 python bench/repairs/tritonbench_g/int8_matmul_kernel_modular_checked/check_compiler_barrier.py
PYTHONDONTWRITEBYTECODE=1 python bench/repairs/tritonbench_g/int8_matmul_kernel_modular_checked/check_delivery.py
```

The task-local evidence directory contains `delivery.json`, build receipts, repository import/object closure, inspection receipts, and the two compiler profiles. Historical evidence and root-owned ledger/Lake selection are outside this candidate's mutations.

Optional proof layout (2026-10-21): `Misc/Audit.lean`, `Misc/NativeCorrect.lean`, `Misc/TensorBridge.lean` are optional entries in `Misc/`. The required `Contract.lean`, `Correct.lean`, `FPEquiv.lean` and their proof helpers do not import these entries. Declaration names, statements, proof terms and acceptance status are retained; this move adds no mathematical completion or numerical experiment.
