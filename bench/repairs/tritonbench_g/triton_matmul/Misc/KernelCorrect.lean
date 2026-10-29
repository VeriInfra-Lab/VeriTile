import bench.repairs.tritonbench_g.triton_matmul.Proofs.KernelExecution
import bench.repairs.tritonbench_g.triton_matmul.Proofs.KernelRealInputs

namespace VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.KernelCorrect
open VeriTile.Triton FP.Structural FPValue
open Contract TensorBinding InputAdapter KernelContract KernelRealModel
noncomputable section

/-- Independent full-K GEMM on unrestricted Real values in the original
physical A/B views. This definition does not consult execution or output. -/
abbrev expected := KernelRealInputs.expected

/-- The selected complete readable DSL and actual finite source grid both
execute. Every output entry is the independent Real GEMM; exterior typed
memory is preserved. Empty output runs no programs; K=0 retains live stores. -/
specification kernel_real (read : KernelRealInputs.RealRead) (a b c : Tensor)
    (la : Laws (storage a.metadata) a) (lb : Laws (storage b.metadata) b)
    (h : InputContract a.metadata b.metadata) (initial : State ℝ)
    (output : Output a b c) (bound : KernelRealInputs.Inputs read a b initial) :
    VeriTile.Spec.Real (∃ actual typed,
      FiniteGrid.grid ideal idealDot (request a b) (List.range (programs (request a b))) initial=some actual ∧
      runPrograms ideal idealDot (request a b) (List.range (programs (request a b))) initial=some typed ∧
      actual.mem=typed.mem ∧ CheckedHost.Post a b c initial actual (expected read a b)) := by
  have input : KernelContract.Inputs (fun _ => read) a b initial := ⟨bound.left,bound.right⟩
  obtain ⟨actual,typed,ha,ht,hm,result⟩ := KernelSource.executes ideal idealDot idealNumerics
    (fun _ => read) a b c h initial output input (ideal_primitives _)
  refine ⟨actual,typed,ha,ht,hm,KernelSource.post idealNumerics a b c initial actual output result _ ?_⟩
  intro i
  have logical := KernelRealInputs.source_inputs read a b la lb h initial bound
  exact ideal_reference (request a b) initial _ _ logical.1 logical.2 i.1 i.2

#specNonCircular expected avoiding [Source.typed,FP.Structural.exec,runPrograms,FiniteGrid.grid]
#axiomsClean kernel_real
end
end VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.KernelCorrect
