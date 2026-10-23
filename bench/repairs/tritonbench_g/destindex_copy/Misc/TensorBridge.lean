import bench.repairs.tritonbench_g.destindex_copy.Correct
import bench.repairs.tritonbench_g.destindex_copy.FPEquiv
import bench.repairs.tritonbench_g.destindex_copy.Misc.NativeObservation
import bench.repairs.tritonbench_g.destindex_copy.Proofs.Composition

/-! Optional Tensor/check applications; the primary kernel specifications
remain in Correct.lean and FPEquiv.lean, which do not import this bridge. -/
namespace VeriTile.Bench.Repairs.DestindexCopy.TensorBridge
open VeriTile.Triton FP.Structural
noncomputable section

theorem python_checks_iff (m : Option Inputs) (w : Nat → Word) :
    checkArguments m w=.ok () ↔ Admitted m w := checkArguments_iff m w

/-- Each repaired payload-plane kernel copies its initial source rows and
preserves all other memory. Host ordering of the two planes is optional. -/
theorem destindex_copy_real (rest : Algebra ℝ) (m : Inputs) (w : Nat → Word)
    (h : InputContract m w) (s : Side) (before : Nat → ℝ) (other : State ℝ) :
    Spec.Real (Stage m w s before
      (DestindexCopyKv1.execute rest (plane m s) w before other)) :=
  DestindexCopyKv1.execute_correct rest (plane m s) w (h.common s) before other

/-- Exact physical FP32 copy and frame for each complete repaired plane kernel.
Output object identity and orchestration of the two invocations are not premises
or conclusions of the kernel contract. -/
theorem destindex_copy_fp32 (rest : Algebra Word) (m : Inputs) (w : Nat → Word)
    (h : InputContract m w) (s : Side) (before : Memory)
    (binding : TensorBinding m w before) (other : State Word) :
    DestindexCopyKv1.SourceConformance (plane m s) w before ∧
    Stage m w s before
      (DestindexCopyKv1.Physical.launch (plane m s) (List.range (config m s).count) before) ∧
    (∀ byte, (∀ x, touched m w s x →
      byte<addr (base m s .output) x ∨ addr (base m s .output) x+4≤byte) →
      DestindexCopyKv1.observeByte
        (DestindexCopyKv1.Physical.launch (plane m s) (List.range (config m s).count) before) byte =
      DestindexCopyKv1.observeByte before byte) := by
  have stage := DestindexCopyKv1.Physical.correct rest (plane m s) w
    (h.common s) before (binding.planeBinding s) other
    (List.range (config m s).count) (by simp [config])
  exact ⟨DestindexCopyKv1.source_conformance (plane m s) w (h.common s) before
    (binding.planeBinding s), stage,
    fun byte outside => DestindexCopyKv1.byte_frame (plane m s) w (h.common s)
      before _ stage.2 byte outside⟩

#axiomsClean destindex_copy_real
#axiomsClean destindex_copy_fp32
#axiomsClean python_checks_iff
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.DestindexCopy.TensorBridge
