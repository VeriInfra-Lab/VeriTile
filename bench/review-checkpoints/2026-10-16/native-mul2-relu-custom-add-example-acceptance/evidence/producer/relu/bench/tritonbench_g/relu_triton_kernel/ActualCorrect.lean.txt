import bench.tritonbench_g.relu_triton_kernel.Proofs.ActualHost
import bench.tritonbench_g.relu_triton_kernel.FiniteCorrect

namespace VeriTile.Bench.TritonBenchG.ReluTritonKernel
open Actual
noncomputable section

/-- Complete source-derived observation for every host call, canonical dtype,
constexpr promotion branch, actual layout and native setup outcome. Ordinary
primitive contracts describe metadata/allocator/ABI access, never a kernel
result. Invalid source accesses are explicitly unclassified observations.
The inherited bounded FiniteCorrect theorem remains available unchanged. -/
specification relu_actual_source_outcomes {σ : Type} (P : Actual.NativePrimitives σ)
    (numerics : Actual.Numerics) (x : Actual.Tensor) (before : ByteMemory.Memory)
    (state : HostWrapper.HostState σ) :
    Actual.sourceHost P numerics x before state=Actual.hostFormula P numerics x before state :=
  Actual.original_source_all_outcomes P numerics x before state

#specNonCircular Actual.hostFormula avoiding [Actual.sourceHost,Actual.runGrid,Actual.runProgram,
  Actual.executeBody,Actual.step,Actual.applyWrites]
#axiomsClean relu_actual_source_outcomes
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ReluTritonKernel
