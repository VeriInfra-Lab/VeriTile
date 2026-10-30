import bench.tritonbench_g.quantize_kv_transform.Proofs.Pair

namespace VeriTile.Bench.Repairs.QuantizeKvTransform.FPEquiv
open VeriTile.Triton FP.Structural KernelSource KernelNativeProfile
open scoped VeriTile.Spec
noncomputable section
local instance instCandidateWord : Inhabited Word := ⟨FP.Value.zero .fp32⟩

/-- Complete original/candidate token-grid equivalence. Pure output pointers
move before floating quantization, preserving FP32 instruction order, INT8
conversion, masks, both scattered outputs, source safety and exterior frames.
The original all-word domain and arbitrary complete token permutation remain. -/
specification address_early_equiv (a : Arithmetic) (rest : Algebra Word) (p : Parameters)
    (width : 0 < p.width) (headCover : p.heads ≤ p.height) (colCover : p.columns ≤ p.width)
    (count : Nat) (order : List Nat) (coverage : order.Perm (List.range count)) :
    Pair.original a rest p count order ≡[[]] Pair.addressEarly a rest p count order :=
  ProgramIO.Implementation.of_equivalentOn (Pair.variant (backend a rest) p count order)
    (Pair.equivalent a rest p width headCover colCover count order coverage)

#axiomsClean address_early_equiv
end
end VeriTile.Bench.Repairs.QuantizeKvTransform.FPEquiv
