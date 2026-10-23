import bench.tritonbench_g.rms_norm_triton.Proofs.KernelIndex
import bench.tritonbench_g.rms_norm_triton.Proofs.NativeKernel

namespace VeriTile.Bench.TritonBenchG.RmsNormTriton.FPEquiv
open VeriTile.Triton FP.Structural Whole Observed
noncomputable section

/-- Native FP32 words. The selected IEEE primitives, FMA contraction and
per-lane reduction tree are explicit compiler/model boundaries. This theorem
uses no numerical experiment assumptions and asserts no FP-to-Real identity.
It proves successful source execution and equality to an independent native
recurrence on every physical memory cell, including the untouched frame. -/
theorem source_native_equivalence
    (M : NativeModel (BitVec 32)) (profile : ReductionFamily) (eps : ℝ)
    (a : Prepared (Term (BitVec 32)) input weight (entryGraph entry) preexisting)
    (domain : InputDomain input weight normalized preexisting)
    (binding : TensorBindings a x w xc wc y) (schedule : Schedule input normalized)
    (finite : FiniteAdmission a normalized) (physical : PhysicalBinding finite x w xc wc y)
    (initial : Memory (BitVec 32)) (heap : a.output.memory = entryGraph initial) :
    KernelPointers xc wc y ∧ IndexCorrespondence a normalized finite ∧
      ActualSourceAddresses a normalized finite xc wc y ∧
    ∃ final, a.entryRun graphAlgebra normalized eps schedule = some final ∧
      KernelResult (nativeFormula M profile (columns normalized) (nextPower (columns normalized)) x w entry)
        y initial (projectMemory M profile (columns normalized) a.output.tensor.base (a.commit final)) ∧
      projectMemory M profile (columns normalized) a.output.tensor.base (a.commit final) =
        referenceMemory (nativeFormula M profile (columns normalized) (nextPower (columns normalized)) x w entry) y initial := by
  exact ⟨kernel_pointers a binding,index_correspondence a normalized finite,
    actual_source_addresses a binding domain finite physical,native_kernel_result M profile eps a domain binding schedule initial heap⟩

#guard_msgs (drop info) in
#specNonCircular nativeFormula avoiding [kernel,lower,interpret,rowValue,sourceRun,Prepared.entryRun]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.RmsNormTriton.FPEquiv
