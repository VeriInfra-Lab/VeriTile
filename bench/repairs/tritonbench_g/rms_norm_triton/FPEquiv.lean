import bench.repairs.tritonbench_g.rms_norm_triton.Proofs.Pair

namespace VeriTile.Bench.Repairs.RmsNormTriton.FPEquiv
open VeriTile.Triton FP.Structural FloatingIO
open VeriTile.Bench.TritonBenchG.RmsNormTriton Whole Observed
open scoped VeriTile.Spec
noncomputable section

/-- The existing candidate commutes only the integer X/Y row-base products.
Both complete grids are safe, succeed, agree under the same native observer
and preserve the complete exterior frame, including every exceptional input. -/
specification index_commuted_equiv (c : Layout) (eps : ℝ)
    (M : NativeModel (BitVec 32)) (epsilonWord : BitVec 32)
    (_epsilonArgument : M.epsilon=epsilonWord) (profile : ReductionFamily)
    (xSeparate : c.X≠c.Y) (wSeparate : c.W≠c.Y)
    (rows : List Nat) (schedule : rows.Perm (List.range c.M)) :
    original c eps M profile rows ≡[[]] indexCommuted c eps M profile rows :=
  ProgramIO.Implementation.of_equivalent (candidate c eps M profile rows)
    (Pair.equivalent c eps M profile xSeparate wSeparate rows schedule)

#axiomsClean index_commuted_equiv
#print_fp_assumptions index_commuted_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RmsNormTriton.FPEquiv
