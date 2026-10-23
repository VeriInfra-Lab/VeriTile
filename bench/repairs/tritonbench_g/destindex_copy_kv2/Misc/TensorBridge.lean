import bench.repairs.tritonbench_g.destindex_copy_kv2.Proofs.Kernel
import VeriTile.Meta.Specification
import bench.repairs.tritonbench_g.destindex_copy_kv2.Proofs.Rewrite
import bench.repairs.tritonbench_g.destindex_copy_kv2.Correct
import bench.repairs.tritonbench_g.destindex_copy_kv2.FPEquiv
import bench.repairs.tritonbench_g.destindex_copy_kv2.Misc.NativeObservation

/-! Optional Python-check correspondence; primary kernel results remain in Correct and FPEquiv. -/
namespace VeriTile.Bench.Repairs.DestindexCopyKv2.TensorBridge
open VeriTile.Triton FP.Structural
noncomputable section

theorem python_checks_iff (m : Option Inputs) (w : Nat → Word) :
    checkArguments m w=.ok () ↔ Admitted m w := checkArguments_iff m w

#axiomsClean python_checks_iff
end
end VeriTile.Bench.Repairs.DestindexCopyKv2.TensorBridge


namespace VeriTile.Bench.Repairs.DestindexCopyKv2.TensorBridge
open VeriTile.Triton FP.Structural
noncomputable section

theorem destindex_copy_kv2_real (rest : Algebra ℝ) (m : Inputs) (w : Nat → Word)
    (h : InputContract m w) (before : Nat → ℝ) (other : State ℝ) :
    Spec.Real (RowCopy (config m) (destination w) (fun x => before (addr (base m .source) x))
      (fun x => before (addr (base m .output) x))
      (fun x => Mathematical.execute rest m w before other (addr (base m .output) x)) ∧
    (∀ cell, (¬∃ x, touched (config m) (destination w) x ∧ cell=addr (base m .output) x) →
      Mathematical.execute rest m w before other cell=before cell)) := Mathematical.correct rest m w h before other

#axiomsClean destindex_copy_kv2_real
#specNonCircular RowCopy avoiding [Mathematical.execute,Physical.launch]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.DestindexCopyKv2.TensorBridge


namespace VeriTile.Bench.Repairs.DestindexCopyKv2.TensorBridge
open VeriTile.Triton FP.Structural FP.ProfiledMemory
open scoped VeriTile.Spec
noncomputable section

/-- Native finite addresses and exact FP32 words for an actual nonempty kernel
invocation. The wrapper's no-launch empty branch is not a proof obligation. -/
theorem destindex_copy_kv2_fp32 (rest : Algebra Word) (m : Inputs) (w : Nat → Word)
    (h : InputContract m w) (ne : Nonempty m) (before : Memory)
    (binding : TensorBinding m w before) (other : State Word) :
    DestindexCopyKv1.SourceConformance m w before ∧
    RowCopy (config m) (destination w) (fun x => before (addr (base m .source) x))
      (fun x => before (addr (base m .output) x))
      (fun x => Physical.launch m (List.range (config m).count) before (addr (base m .output) x)) ∧
    (∀ byte, (∀ x, touched (config m) (destination w) x →
      byte<addr (base m .output) x ∨ addr (base m .output) x+4≤byte) →
      observeByte (Physical.launch m (List.range (config m).count) before) byte=observeByte before byte) := by
  rw [Physical.launch_eq m w h ne]
  obtain ⟨hv, hf⟩ := DestindexCopyKv1.Physical.correct rest m w h.common before binding other (List.range (config m).count) (by simp)
  exact ⟨DestindexCopyKv1.source_conformance m w h.common before binding, hv,
    fun byte outside => DestindexCopyKv1.byte_frame m w h.common before _ hf byte outside⟩


#axiomsClean destindex_copy_kv2_fp32
end
end VeriTile.Bench.Repairs.DestindexCopyKv2.TensorBridge
