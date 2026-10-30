import bench.tritonbench_g.softmax_triton1.Proofs.Rewrite

namespace VeriTile.Bench.Repairs.SoftmaxTriton1.FPEquiv
open VeriTile Triton FP.Structural
open scoped VeriTile.Spec
noncomputable section

abbrev Rules := FP.ScalarArithmetic.Rules

/-- Actual common-prefix operands: only stored numerator lanes must be finite;
the shared denominator must remain finite and nonzero. -/
abbrev divisionSite := Rewrite.divisionSite

def original (out inp : RegionName) (ib ob ir ic orow oc n B : Nat) : Rewrite.Program where
  kernel := softmax_kernel out inp ib ob ir ic orow oc n B
  domain := divisionSite out inp ib ob ir ic orow oc n B

def reciprocal (out inp : RegionName) (ib ob ir ic orow oc n B : Nat) : Rewrite.Program :=
  { original out inp ib ob ir ic orow oc n B with
    kernel := reciprocal_kernel out inp ib ob ir ic orow oc n B }

/-- Both complete typed sources succeed and agree on every memory cell under
DIV-MUL-RCP at their computed operands. The admitted numerical relation is
conditional; this is not a universal IEEE bit-equality theorem. -/
specification softmax_reciprocal_equiv (R : Rules) (out inp : RegionName)
    (ib ob ir ic orow oc n B : Nat) (hB : 0 < B) :
    original out inp ib ob ir ic orow oc n B ≡[R]
      reciprocal out inp ib ob ir ic orow oc n B :=
  Rewrite.equivalence R out inp ib ob ir ic orow oc n B hB


#print_fp_assumptions softmax_reciprocal_equiv
#axiomsClean softmax_reciprocal_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SoftmaxTriton1.FPEquiv
