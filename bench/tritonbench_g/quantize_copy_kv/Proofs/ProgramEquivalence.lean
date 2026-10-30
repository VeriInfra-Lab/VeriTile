import bench.tritonbench_g.quantize_copy_kv.Programs

namespace VeriTile.Bench.Repairs.QuantizeCopyKv.FPEquivProof
open VeriTile.Triton FP.Structural KernelSource
open KernelNativeProfile (Word)
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩
open scoped VeriTile.Spec
noncomputable section

/-- Reuse the existing successful full-token-grid equivalence. Both programs
retain their actual invocations, INT64 metadata and exact output cell frame. -/
theorem equivalent (a : KernelNativeProfile.Arithmetic) (rest : Algebra Word)
    (p : Parameters) (count : Nat) (dest : Nat → Nat)
    (width : 0 < p.width) (covered : p.heads ≤ p.height)
    (unique : ∀ a b, a < count → b < count → dest a = dest b → a = b) :
    FPEquiv.original a rest p count dest ≡[[]] FPEquiv.indexCommuted a rest p count dest :=
  ProgramIO.Implementation.of_equivalentOn _
    (Interface.native_equivalent a rest p count dest width covered unique)

#axiomsClean equivalent
end
end VeriTile.Bench.Repairs.QuantizeCopyKv.FPEquivProof
