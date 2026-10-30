# Selected source and compiler binding

The source is exactly the `contract_revision.source` in the captured task ledger row, `bench/repairs/tritonbench_g/matmul_triton2/revision001/matmul_triton2_repaired.py`.

The selected revision performs nine explicit `tl.cast(..., tl.uint64)` operations on M/N/K and six dense strides. PID and arange results use `.to(tl.uint64)`. The grouped `min` result is explicitly cast. Runtime specialization of integer one therefore calls `tl.cast` on a constexpr, rather than invoking a nonexistent constexpr `.to` method.

The wrapper passes the original A and B objects with canonical strides K,1,N,1,N,1. This preserves offset contiguous views and singleton layouts. It allocates FP32 C on A's device, returns without a launch for M=0 or N=0, and otherwise invokes the unchanged eight-configuration autotuner in the input device context.

The source quotation is the task-specific readable shared-DSL `Revision001.Core.typed` in the frozen `revision001/Kernels.lean`; `KernelOnly.Kernels.kernel` names this same source. Both input masks and the output mask, grouped row permutation, K loop and padded lanes are retained. The actual compiler fuses the Python `accumulator += tl.dot(a,b)` to accumulator-fed `tt.dot` with TF32 inputs. The independent native target uses the captured SM90 family and fixed k8 instruction ports. No precision override or numerical source repair is introduced.

The Python checker normalizes precisely the reviewed casts and repair masks against the original benchmark AST, checks the complete wrapper's admission guards and canonical argument list, and executes all eight kernels through Triton's CPU interpreter. The new binder probe imports the unmodified selected JIT, captures actual wrapper arguments, calls `create_function_from_signature` and `JITFunction._pack_args`, and passes that exact signature/constants/attributes to `triton.compile`. Its report and all 12 TTIR/PTX pairs are pinned.

All source files remain unchanged. The failed initial cache `rfl` probe and subsequent task-only successful refresh are retained as build provenance. The baseline cache stayed read-only; new receipts identify the unchanged source bytes. The requested pre-refresh receipt capture was made after the first three modules had already refreshed, so the packet records that timing and pins the untouched baseline objects rather than claiming nonexistent earlier receipts.

The 2026-10-21 main program pair uses the complete
`matmul_triton2_mask_hoisted.py` candidate: two invariant vector comparisons are
computed before the K loop and reused in both load masks and the final store.
The original shared quotation is unchanged; `maskHoisted` is a complete ordinary
candidate quotation. `MaskHoist.exec_memory` proves the actual full candidate
execution realizes the same masked ordered-dot recurrence, and `Pair.equivalent`
proves both safe successful full IO schedules, all observed output cells and
both exact raw exterior frames. Python AST reverse-inlining checks preserve
every uint64 cast, all eight autotune configurations/keys and the complete
wrapper/precision statements. No measured performance improvement is asserted.
