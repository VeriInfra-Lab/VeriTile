import bench.repairs.tritonbench_g.decay_cumsum.Proofs.KernelCompiler

/-! Kernel-only native FP32 graph equivalence. The selected object's local
instruction graphs and original runtime scale ABI are consumed by source
execution and independent output references. All raw FP32 words are admitted;
FMA, operand order, first-step additions and approximate FTZ exp2 are retained.
Primitive semantics and object decoding are explicit model boundaries, with
no deferred split/fused numerical equality or output oracle. -/
namespace VeriTile.Bench.Repairs.DecayCumsum.KernelNativeResults
noncomputable section
abbrev decay_cumsum_graph_equiv := @KernelNative.complete
abbrev decay_cumsum_kernel_equiv := @KernelCompiler.complete
abbrev decay_cumsum_runtime_scale := @KernelCompiler.runtime_scale_value
#axiomsClean decay_cumsum_graph_equiv
#axiomsClean decay_cumsum_kernel_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.DecayCumsum.KernelNativeResults
