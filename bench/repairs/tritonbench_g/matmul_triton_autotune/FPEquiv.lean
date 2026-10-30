import bench.repairs.tritonbench_g.matmul_triton_autotune.Proofs.Equivalence

namespace VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Repair.Kernel
open VeriTile.Triton FP.Structural Whole
open scoped VeriTile.Spec
noncomputable section

abbrev original := FloatingIO.original
abbrev safeAddresses := FloatingIO.safe

def assumptions : Spec.Assumptions ComputeStmt := []

/-- The original source and existing safe-address candidate both execute the
complete selected grid and agree on every typed memory cell. Only integer
address/control expressions change; FP32 dot carry and FP16 stores are retained. -/
specification matmul_equiv (instructions : Floating.Instructions) (g : Geometry)
    (ids : List Nat) (order : ids.Perm (List.range (programs g.request))) :
    original instructions g ids order ≡[assumptions] safeAddresses instructions g ids order := by
  apply Spec.FloatingPoint.ofStructural
    (lhs := original instructions g ids order)
    (rhs := safeAddresses instructions g ids order)
    (structural := FloatingIO.Equivalent) rfl rfl
  exact EquivalenceProof.equivalent instructions g ids order

#print_fp_assumptions matmul_equiv
#axiomsClean matmul_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Repair.Kernel
