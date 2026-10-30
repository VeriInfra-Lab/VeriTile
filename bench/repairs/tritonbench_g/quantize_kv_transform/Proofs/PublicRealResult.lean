import bench.repairs.tritonbench_g.quantize_kv_transform.Proofs.KernelInterface
import bench.repairs.tritonbench_g.quantize_kv_transform.Proofs.KernelRealResult

namespace VeriTile.Bench.Repairs.QuantizeKvTransform.PublicRealResult
open VeriTile.Triton FP.Structural KernelSource
noncomputable section

/-- Complete raw source launch, independent logical-column Real quantization,
and every exterior typed cell. Coverage makes every logical coordinate an
active finite tile coordinate in the result. -/
theorem kernel_real (rest : Algebra ℝ) (p : Parameters)
    (positive : 0 < p.columns) (_headsCovered : p.heads ≤ p.height)
    (columnsCovered : p.columns ≤ p.width) (count : Nat) (dest : Nat → Nat)
    (values : Nat → Nat → Nat → ℝ) (initial : State ℝ)
    (reads : ∀ token head column, token < count → head < p.heads → column < p.columns →
      (initial.mem "input" (KernelResult.inputOffset p token head column)).read .real = values token head column)
    (metadata : ∀ token, token < count →
      (initial.mem "destination" token).read .int = (dest token : Int))
    (unique : ∀ a b, a < count → b < count → dest a = dest b → a = b)
    (order : List Nat) (schedule : order.Perm (List.range count)) :
    ∃ final, KernelGrid.run (KernelExecution.backend (KernelRealModel.algebra rest)
      KernelRealModel.convert KernelRealModel.equal) p order initial = some final ∧
      KernelInterface.Result p count dest
        (fun token i => KernelRealModel.code (values token i.1.val) p.columns i.2.1.val)
        (fun token head => KernelRealModel.scale (values token head.val) p.columns) initial final := by
  obtain ⟨final, ran, codes, scales, frame⟩ := KernelInterface.complete
    (KernelRealModel.algebra rest) KernelRealModel.convert KernelRealModel.equal p
    (lt_of_lt_of_le positive columnsCovered) count dest initial metadata unique order schedule
  refine ⟨final, ran, ?_, ?_, frame⟩
  · intro token i ht hi
    rw [codes token i ht hi]
    congr 1
    apply KernelRealResult.code rest p _ positive columnsCovered i.1 hi.1 (values token i.1.val) _ i.2.1 hi.2
    intro c hc
    simpa [KernelGrid.atProgram] using reads token i.1.val c ht hi.1 hc
  · intro token head ht hh
    rw [scales token head ht hh]
    congr 1
    apply KernelRealResult.scale rest p _ positive columnsCovered head hh (values token head.val)
    intro c hc
    simpa [KernelGrid.atProgram] using reads token head.val c ht hh hc

#axiomsClean kernel_real
end
end VeriTile.Bench.Repairs.QuantizeKvTransform.PublicRealResult
