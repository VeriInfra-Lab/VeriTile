import bench.repairs.tritonbench_g.chunk_gate_recurrence.Proofs.KernelCompiler

/-! Kernel-only native FP32 graph equivalence, including every forward,
input-gradient, complete tile contraction and seed-gradient store. Actual
logical Tensor applications are exposed separately in TensorBridge.lean. Selected-object
ABI/decoder binding and primitive bit semantics are explicit model boundaries;
no split/fused equality or primitive-conformance experiment is deferred. -/
namespace VeriTile.Bench.Repairs.ChunkGateRecurrence.RawNative
noncomputable section
alias chunk_gate_recurrence_graph_equiv := KernelNative.complete
alias chunk_gate_recurrence_kernel_equiv := KernelCompiler.complete
#axiomsClean chunk_gate_recurrence_graph_equiv
#axiomsClean chunk_gate_recurrence_kernel_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGateRecurrence.RawNative
