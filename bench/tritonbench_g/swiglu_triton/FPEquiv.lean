import bench.tritonbench_g.swiglu_triton.Proofs.IndexEquivalence

/-! Both actual FP32 source stages compared with their existing row-index
candidate. All floating arithmetic, masks, conversions, backward FMA and
ordered in-place stores remain fixed. Native formula observations are ordinary
auxiliary theorems in NativeCorrect.lean. -/
namespace VeriTile.Bench.Repairs.SwigluTriton
open VeriTile Triton
open scoped VeriTile.Spec
noncomputable section

specification forward_index_equiv (block cols : Nat) :
    Pair.original Bool.false block cols ≡[([] : Spec.Assumptions FP.GuardedFragment)]
      Pair.indexCommuted Bool.false block cols :=
  PairProof.equivalent Bool.false block cols

specification backward_index_equiv (block cols : Nat) :
    Pair.original Bool.true block cols ≡[([] : Spec.Assumptions FP.GuardedFragment)]
      Pair.indexCommuted Bool.true block cols :=
  PairProof.equivalent Bool.true block cols

#print_fp_assumptions forward_index_equiv
#print_fp_assumptions backward_index_equiv
#axiomsClean forward_index_equiv
#axiomsClean backward_index_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SwigluTriton
