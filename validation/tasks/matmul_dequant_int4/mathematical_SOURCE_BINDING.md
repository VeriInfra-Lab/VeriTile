# Source and proof binding

Selected predecessor: `bench/tritonbench_g/matmul_dequant_int4/matmul_dequant_int4.py`, SHA256 `7442f809ad84ab32016abffe4f02a0b6ba1b527bff1555151562fdb3419f636f`.

Candidate: `matmul_dequant_int4_grid_y_checked.py`. Deleting the two added `_packed` guard lines produces the predecessor byte for byte. The candidate guard probe also compares the full decorated JIT ASTs and complete quantizer AST. The source configurations, launch scalar order, singleton stride cases, integer unpacking, conversion placement, mask, store, raw fused loop and group branches are unchanged.

`DSL.Standalone.typed` is definitionally identical to the original benchmark's typed standalone quotation for both floating profiles. `Source.execution` derives successful full shared execution, and `Source.execution_memory` derives its exact stores and frame. `KernelGrid.whole` composes every original two-dimensional grid program with geometric coverage and injectivity. `Spec.result_eq` translates signed shifts of actual INT32 payloads to independent unsigned base-16 digits at actual Tensor coordinates.

`DSL.Fused.typed` is definitionally identical to the complete retained fused quotation. `Raw.configurations` proves all 17 tuning entries agree, and `Raw.noGroups` proves the retained Boolean branch selection. `Raw.valid` is derived directly from the existing `RawAccepted`; no sibling `Checked.InputContract` is required. Complete fused Real/native proofs are reused with the existing contraction/fixed-K16 model and then lifted to actual Tensor metadata/storage/coordinates.

The only numerical interface added for the standalone quotation is the FP16 finite promotion identity on signed integers -15 through 15. It is an explicit local primitive port, supported by the existing exact-conversion lemmas, and is not a numerical experiment. The FP32 theorem has no promotion obligation. Reused raw primitive-model ports and historical Torch-library ports remain explicit. There is no native-FP-equals-Real assertion, output-equality assumption, assumed source success, or new runtime domain restriction behind these ports.
