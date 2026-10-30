import bench.repairs.tritonbench_g.quantize_kv_transform.Proofs.PublicIO
import bench.repairs.tritonbench_g.quantize_kv_transform.Proofs.PublicNativeResult

namespace VeriTile.Bench.Repairs.QuantizeKvTransform.NativeIO
open VeriTile.Triton FP.Structural KernelSource KernelNativeProfile
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩
noncomputable section
set_option maxHeartbeats 6000000

/-- Exact native ordered reduction, two divisions and selected signed-byte
conversion, on every source logical row and raw FP32 input word. -/
def mathematical (a : Arithmetic) (p : Parameters) (headCover : p.heads ≤ p.height)
    (colCover : p.columns ≤ p.width) (count : Nat) (vals : IO.Inputs Word p count) :=
  IO.outputValues (p := p) (count := count)
    (fun t h c => KernelNativeResult.code a p (IO.values vals t.val)
      (⟨h.val, lt_of_lt_of_le h.isLt headCover⟩,
       ⟨c.val, lt_of_lt_of_le c.isLt colCover⟩, PUnit.unit))
    (fun t h => KernelNativeResult.scale a p (IO.values vals t.val)
      ⟨h.val, lt_of_lt_of_le h.isLt headCover⟩)

theorem implements (a : Arithmetic) (rest : Algebra Word) (p : Parameters)
    (width : 0 < p.width) (headCover : p.heads ≤ p.height) (colCover : p.columns ≤ p.width)
    (count : Nat) (order : List Nat) (coverage : order.Perm (List.range count)) :
    (IO.quantizeIO (KernelNativeProfile.backend a rest) p count order).ImplementsOn
      (IO.Destinations p count) (mathematical a p headCover colCover count) := by
  apply PublicIO.implements (algebra a rest) PtxSigned.signedByte a.equal p width
    headCover colCover count order coverage
    (fun vals t h c => KernelNativeResult.code a p (vals t)
      (⟨h.val, lt_of_lt_of_le h.isLt headCover⟩,
       ⟨c.val, lt_of_lt_of_le c.isLt colCover⟩, PUnit.unit))
    (fun vals t h => KernelNativeResult.scale a p (vals t)
      ⟨h.val, lt_of_lt_of_le h.isLt headCover⟩)
  intro vals initial dest reads metadata unique
  obtain ⟨final, ran, codes, scales, frame⟩ := PublicNativeResult.kernel_native a rest p width
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
end VeriTile.Bench.Repairs.QuantizeKvTransform.NativeIO
