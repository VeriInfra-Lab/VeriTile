import bench.tritonbench_g.quantize_kv_transform.Proofs.KernelInterface
import bench.tritonbench_g.quantize_kv_transform.Proofs.KernelNativeResult

namespace VeriTile.Bench.Repairs.QuantizeKvTransform.PublicNativeResult
open VeriTile.Triton FP.Structural KernelSource KernelNativeProfile
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩
noncomputable section

/-- Complete native source launch and independent ordered reduction/division/
signed-byte target, over raw words, geometry, reads and destination schedule. -/
theorem kernel_native (a : Arithmetic) (rest : Algebra Word) (p : Parameters)
    (width : 0 < p.width) (_headsCovered : p.heads ≤ p.height)
    (_columnsCovered : p.columns ≤ p.width) (count : Nat) (dest : Nat → Nat)
    (values : Nat → Nat → Nat → Word) (initial : State Word)
    (reads : ∀ token head column, token < count → head < p.heads → column < p.columns →
      (initial.mem "input" (KernelResult.inputOffset p token head column)).read .real = values token head column)
    (metadata : ∀ token, token < count →
      (initial.mem "destination" token).read .int = (dest token : Int))
    (unique : ∀ x y, x < count → y < count → dest x = dest y → x = y)
    (order : List Nat) (schedule : order.Perm (List.range count)) :
    ∃ final, KernelGrid.run (KernelNativeProfile.backend a rest) p order initial = some final ∧
      KernelInterface.Result p count dest
        (fun token => KernelNativeResult.code a p (values token))
        (fun token => KernelNativeResult.scale a p (values token)) initial final := by
  obtain ⟨final, ran, codes, scales, frame⟩ := KernelInterface.complete
    (algebra a rest) PtxSigned.signedByte a.equal p width count dest initial metadata unique order schedule
  refine ⟨final, ran, ?_, ?_, frame⟩
  · intro token i ht hi
    rw [codes token i ht hi]
    congr 1
    apply KernelNativeResult.code_eq
    intro head column hh hc
    simpa [KernelGrid.atProgram] using reads token head column ht hh hc
  · intro token head ht hh
    rw [scales token head ht hh]
    congr 1
    apply KernelNativeResult.scale_eq
    intro h c ha hc
    simpa [KernelGrid.atProgram] using reads token h c ht ha hc

#axiomsClean kernel_native
end
end VeriTile.Bench.Repairs.QuantizeKvTransform.PublicNativeResult
