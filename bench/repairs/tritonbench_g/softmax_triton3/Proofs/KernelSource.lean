import bench.repairs.tritonbench_g.softmax_triton3.Kernels

/-! The raw source retains signed index casts, caller scalar bindings and the
optional-pointer test. It is no longer AST-equal to the historical Nat-index
quotation. Actual Real execution/safety is proved in RealExecution.lean;
FP execution and safety are proved independently in FP/PairExecution.lean. -/
