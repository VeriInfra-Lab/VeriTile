import bench.tritonbench_g.quantize_copy_kv.Proofs.RawKernel

namespace VeriTile.Bench.Repairs.QuantizeCopyKv.RawNative
open VeriTile.Triton FP.Structural KernelSource
open VeriTile.Bench.TritonBenchG.QuantizeKvTransform
open KernelNativeProfile (Word)
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩
noncomputable section

/-- Full raw launches agree, including every native int8 code, FP16 scale and
exterior typed cell. The native callbacks retain reduction order, div.full,
half rounding/promotion, exceptional words and total PTX signed-byte conversion. -/
theorem kernel_equiv (a : KernelNativeProfile.Arithmetic) (rest : Algebra Word)
    (p : Parameters) (count : Nat) (dest : Nat → Nat)
    (hw : 0 < p.width) (covered : p.heads ≤ p.height)
    (unique : ∀ a b, a < count → b < count → dest a = dest b → a = b)
    (values : Nat → Nat → Nat → Word) (initial : State Word)
    (metadata : ∀ token, token < count →
      (initial.mem "destination" token).read .int = (dest token : Int))
    (reads : ∀ token head column, token < count → head < p.heads → column < p.width →
      (initial.mem "input" (KernelResult.inputOffset p token head column)).read .real = values token head column) :
    ∃ final,
      KernelGrid.run (KernelNativeProfile.backend a rest) p Bool.false (List.range count) initial = some final ∧
      KernelGrid.run (KernelNativeProfile.backend a rest) p Bool.true (List.range count) initial = some final ∧
      RawKernel.Result p count dest (RawKernel.nativeCode a p values)
        (RawKernel.nativeScale a p values) initial final := by
  apply RawKernel.whole (KernelNativeProfile.algebra a rest) PtxSigned.signedByte a.equal
    p count dest hw covered unique initial metadata
  · intro token i ht hh
    simp only [RawKernel.nativeCode, i.1.isLt, i.2.1.isLt, dite_true, Fin.eta]
    apply KernelNativeResult.code_eq
    intro head column ha hc
    simpa [KernelGrid.atProgram] using reads token head column ht ha hc
  · intro token head ht hh
    simp only [RawKernel.nativeScale, head.isLt, dite_true, Fin.eta]
    apply KernelNativeResult.scale_eq
    intro a c ha hc
    simpa [KernelGrid.atProgram] using reads token a c ht ha hc

#axiomsClean kernel_equiv
end
end VeriTile.Bench.Repairs.QuantizeCopyKv.RawNative
