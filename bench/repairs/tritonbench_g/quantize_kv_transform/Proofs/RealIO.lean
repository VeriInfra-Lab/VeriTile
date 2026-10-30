import bench.repairs.tritonbench_g.quantize_kv_transform.Proofs.PublicIO
import bench.repairs.tritonbench_g.quantize_kv_transform.Proofs.PublicRealResult

namespace VeriTile.Bench.Repairs.QuantizeKvTransform.RealIO
open VeriTile.Triton FP.Structural KernelSource
noncomputable section
set_option maxHeartbeats 6000000

/-- Independent quantization codes and pre-selection scales for every logical
input row; the result exposes both outputs and no block-dependent recurrence. -/
def mathematical (p : Parameters) (count : Nat) (vals : IO.Inputs ℝ p count) :=
  IO.outputValues (p := p) (count := count)
    (fun t h c => KernelRealModel.code (IO.values vals t.val h.val) p.columns c.val)
    (fun t h => KernelRealModel.scale (IO.values vals t.val h.val) p.columns)

theorem implements (rest : Algebra ℝ) (p : Parameters) (positive : 0 < p.columns)
    (headCover : p.heads ≤ p.height) (colCover : p.columns ≤ p.width)
    (count : Nat) (order : List Nat) (coverage : order.Perm (List.range count)) :
    (IO.quantizeIO (KernelExecution.backend (KernelRealModel.algebra rest)
      KernelRealModel.convert KernelRealModel.equal) p count order).ImplementsOn
      (IO.Destinations p count) (mathematical p count) := by
  apply PublicIO.implements (KernelRealModel.algebra rest) KernelRealModel.convert
    KernelRealModel.equal p (lt_of_lt_of_le positive colCover) headCover colCover count order coverage
    (fun vals t h c => KernelRealModel.code (vals t h.val) p.columns c.val)
    (fun vals t h => KernelRealModel.scale (vals t h.val) p.columns)
  intro vals initial dest reads metadata unique
  obtain ⟨final, ran, codes, scales, frame⟩ := PublicRealResult.kernel_real rest p positive
    headCover colCover count dest vals initial reads metadata unique order coverage
  refine ⟨final, ran, ?_, ?_, frame⟩
  · intro t h c ht
    exact codes t (⟨h.val, lt_of_lt_of_le h.isLt headCover⟩,
      ⟨c.val, lt_of_lt_of_le c.isLt colCover⟩, PUnit.unit) ht ⟨h.isLt, c.isLt⟩
  · intro t h ht
    exact scales t ⟨h.val, lt_of_lt_of_le h.isLt headCover⟩ ht h.isLt

#specNonCircular mathematical avoiding [kernel, FP.Structural.exec, KernelGrid.run]
#axiomsClean implements
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.QuantizeKvTransform.RealIO
