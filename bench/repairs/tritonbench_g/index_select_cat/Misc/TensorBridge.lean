import bench.repairs.tritonbench_g.index_select_cat.Proofs.KernelNativeResults
import bench.repairs.tritonbench_g.index_select_cat.Proofs.Kernel

/-! Optional Tensor checks and binding consequences. -/
namespace VeriTile.Bench.Repairs.IndexSelectCat.TensorBridge
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy
noncomputable section

theorem python_checks_iff_preconditions (a : Arguments) :
    checkArguments a = Bool.true ↔ AdmittedArguments a := KernelProof.argument_checks_iff a

#axiomsClean python_checks_iff_preconditions

namespace Correct
/-- Per-program correctness of the complete shared typed source over arbitrary
Real payloads. Validated indices are bound to the actual initial index memory;
all active columns use the independent initial-input gather formula. -/
theorem index_select_kernel_correctness (rest : Algebra ℝ) (r : Request)
    (checked : checkInputs r = Bool.true) (s : State ℝ)
    (launch : Launched r s.pids)
    (observed : IndexObservation r (fun i => (s.mem r.layout.index i).read .int)) :
    Spec.Real (∃ t,FP.Structural.exec (copyBackend rest) (kernel r.layout) s = some t ∧
      (∀ d : Fin 512,s.pids 1*512+d.val < r.D →
        t.mem r.layout.output (s.pids 0*r.D+(s.pids 1*512+d.val)) =
          .mk .real (KernelProof.expected r s (s.pids 0) (s.pids 1*512+d.val))) ∧
      (∀ region off,region ≠ r.layout.output ∨ ¬KernelProof.Written r s.pids off →
        t.mem region off = s.mem region off)) :=
  KernelProof.correct rest r ((KernelProof.checks_iff r).mp checked) s launch observed
#axiomsClean index_select_kernel_correctness
end Correct

namespace FPEquiv
open VeriTile.Bench.Repairs.IndexSelectCat.KernelNativeResults
/-- Both actual FP32/int64 sources execute successfully, agree on every typed
memory cell, gather the initial values, and preserve the exterior frame.
The copy backend admits every 32-bit payload and introduces no numerical law. -/
theorem index_select_successful_equivalence (rest : Algebra (BitVec 32))
    (r : Request) (checked : checkInputs r = Bool.true) (s : State (BitVec 32))
    (launch : Launched r s.pids)
    (observed : IndexObservation r (fun i => (s.mem r.layout.index i).read .int)) :
    ∃ left right,
      FP.Structural.exec (copyBackend rest) (original r.layout).kernel s = some left ∧
      FP.Structural.exec (copyBackend rest) (indexCommuted r.layout).kernel s = some right ∧
      left.mem = right.mem ∧
      (∀ d : Fin 512,s.pids 1*512+d.val < r.D →
        left.mem r.layout.output (s.pids 0*r.D+(s.pids 1*512+d.val)) =
          .mk .real (KernelProof.expected r s (s.pids 0) (s.pids 1*512+d.val))) ∧
      (∀ region off,region ≠ r.layout.output ∨ ¬KernelProof.Written r s.pids off →
        left.mem region off = s.mem region off) := by
  obtain ⟨t,run,values,frame⟩ := KernelProof.correct rest r
    ((KernelProof.checks_iff r).mp checked) s launch observed
  refine ⟨t,t,run,?_,rfl,values,frame⟩
  change FP.Structural.exec (copyBackend rest) (indexCommutedKernel r.layout) s = some t
  rw [←Rewrite.execution_eq]
  exact run
#axiomsClean index_select_successful_equivalence
end FPEquiv

theorem accesses {α : Type} [Inhabited α] (r : Request) (h : InputContract r) (s : State α)
    (launch : Launched r s.pids)
    (observed : IndexObservation r (fun i => (s.mem r.layout.index i).read .int)) :
    (∀ i : KernelProof.Lane,TypedGather.indexOffset r.layout s i < r.N) ∧
    (∀ i : KernelProof.Lane,TypedGather.active r.layout s i →
      TypedGather.inputOffset r.layout s i < r.M*r.D ∧
      TypedGather.outputOffset r.layout s i < r.N*r.D) :=
  KernelProof.accesses r h s launch observed

#axiomsClean accesses
end
end VeriTile.Bench.Repairs.IndexSelectCat.TensorBridge
