import bench.repairs.tritonbench_g.quantize_copy_kv.Programs

namespace VeriTile.Bench.Repairs.QuantizeCopyKv.NativeCorrect
open VeriTile.Triton FP.Structural KernelSource
open KernelNativeProfile (Word)
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩
noncomputable section

/-- Retained native code/scale formula for the complete original schedule.
This ordinary auxiliary theorem is separate from the two-program headline. -/
theorem kernel_native (a : KernelNativeProfile.Arithmetic) (rest : Algebra Word)
    (p : Parameters) (count : Nat) (dest : Nat → Nat)
    (width : 0 < p.width) (covered : p.heads ≤ p.height)
    (unique : ∀ a b, a < count → b < count → dest a = dest b → a = b) :
    (FPEquiv.originalIO a rest p count dest).ImplementsOn
      (Interface.metadataMatches p count dest) (FPEquiv.native a p count) :=
  Interface.native_implements a rest p count dest Bool.false width covered unique

#axiomsClean kernel_native
#specNonCircular FPEquiv.native avoiding [KernelSource.kernel, FP.Structural.exec, KernelGrid.run]
end
end VeriTile.Bench.Repairs.QuantizeCopyKv.NativeCorrect
