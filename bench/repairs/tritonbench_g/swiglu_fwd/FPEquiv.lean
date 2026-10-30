import bench.repairs.tritonbench_g.swiglu_fwd.Proofs.SourceEquivalence

namespace VeriTile.Bench.Repairs.SwigluFwd
open VeriTile.Triton
open scoped VeriTile.Spec
noncomputable section

abbrev original := SourceEquivalence.original
abbrev indexCommuted := SourceEquivalence.indexCommuted
def assumptions := SourceEquivalence.assumptions

/-- The two existing complete FP32 sources differ only in three integer
row-stride products. All floating operations, casts and stores stay ordered. -/
specification swiglu_index_commuted_equiv (X Y OUT : RegionName) (sx sy so N B : Nat) :
    original X Y OUT sx sy so N B ≡[assumptions]
      indexCommuted X Y OUT sx sy so N B :=
  SourceEquivalence.equivalent X Y OUT sx sy so N B

#print_fp_assumptions swiglu_index_commuted_equiv
#axiomsClean swiglu_index_commuted_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SwigluFwd
