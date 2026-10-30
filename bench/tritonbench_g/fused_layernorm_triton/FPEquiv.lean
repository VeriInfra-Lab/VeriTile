import bench.tritonbench_g.fused_layernorm_triton.Proofs.PairEquivalence

namespace VeriTile.Bench.Repairs.FusedLayernormTriton
open VeriTile Triton
open scoped VeriTile.Spec
noncomputable section

/-- Hoist the invariant row scalar projections out of the second loop.
Both complete grids safely succeed, agree on all typed Mean/Rstd/Y word
observations and preserve both exterior frames over the full accepted domain.
This is an exact structural transformation with no experimental FP rules. -/
specification whole_fp (p : Pair.Parameters) :
    Pair.original p ≡[Pair.assumptions] Pair.candidate p :=
  Pair.equivalent p

#axiomsClean whole_fp
#stmtSurfaceSubset whole_fp ⊆
  [Spec.Equiv,Pair.Program,Pair.instProgramSyntaxProgram,
   Pair.original,Pair.candidate,Pair.assumptions,Pair.Parameters,ComputeStmt]
#print_fp_assumptions whole_fp
end
end VeriTile.Bench.Repairs.FusedLayernormTriton
