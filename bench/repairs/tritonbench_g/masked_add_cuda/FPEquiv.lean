import bench.repairs.tritonbench_g.masked_add_cuda.Proofs.KernelNativeResults

/-! Existing addition-order sources under the selected fused native graph.
Both sources use the same alpha/data/grad FMA operand order. -/
namespace VeriTile.Bench.Repairs.MaskedAddCuda.FPEquiv
open VeriTile.Triton FP.Structural
open scoped VeriTile.Spec
noncomputable section

abbrev original := KernelNativeResults.original
abbrev commuted := KernelNativeResults.commuted
def assumptions := KernelNativeResults.assumptions

/-- Both complete sources succeed and agree on all typed memory cells.
Only active gradient inputs must be word leaves; every word and arbitrary data
graph remain admitted. The selected fused primitive is the existing model. -/
specification masked_add_commuted_equiv (storage : MaskStorage) (n : Nat) (token : ℝ) :
    original storage n token ≡[assumptions] commuted storage n token :=
  KernelNativeResults.masked_add_commuted_equiv storage n token

#print_fp_assumptions masked_add_commuted_equiv
#axiomsClean masked_add_commuted_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MaskedAddCuda.FPEquiv
