import bench.repairs.tritonbench_g.quantize_copy_kv.Proofs.ProgramEquivalence

namespace VeriTile.Bench.Repairs.QuantizeCopyKv.FPEquiv
open VeriTile.Triton FP.Structural KernelSource
open KernelNativeProfile (Word)
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩
open scoped VeriTile.Spec
noncomputable section

/-- Full original and existing commuted schedules, on the same metadata/input
domain. Only one integer address product changes; every floating operation,
conversion, output cell and exterior frame is preserved without experiments. -/
specification kernel_equiv (a : KernelNativeProfile.Arithmetic) (rest : Algebra Word)
    (p : Parameters) (count : Nat) (dest : Nat → Nat)
    (width : 0 < p.width) (covered : p.heads ≤ p.height)
    (unique : ∀ a b, a < count → b < count → dest a = dest b → a = b) :
    original a rest p count dest ≡[[]] indexCommuted a rest p count dest :=
  FPEquivProof.equivalent a rest p count dest width covered unique

#print_fp_assumptions kernel_equiv
#axiomsClean kernel_equiv
end
end VeriTile.Bench.Repairs.QuantizeCopyKv.FPEquiv
