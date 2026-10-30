import bench.repairs.tritonbench_g.flash_attn_scalar_abi_checked.Proofs.PairEquivalence

namespace VeriTile.Bench.Repairs.FlashAttnScalarABIChecked
open VeriTile Triton FP.Structural FlashAttn
open scoped VeriTile.Spec
noncomputable section

/-- The retained integer head-offset normalization, applied to the repaired
independent Q/K/V/O pitches. Both complete grids safely succeed, agree on all
typed O/L observations and preserve their entire exterior frames. The original
all-word scalar ABI, layouts, FP32 storage and FP16 conversions are retained. -/
specification whole_fp (x : Layout) :
    PairIO.original x ≡[PairIO.assumptions] PairIO.candidate x :=
  PairIO.equivalent x

#axiomsClean whole_fp
#stmtSurfaceSubset whole_fp ⊆
  [Spec.Equiv,PairIO.Program,PairIO.instProgramSyntaxProgram,
   PairIO.original,PairIO.candidate,PairIO.assumptions,Layout,ComputeStmt]
#print_fp_assumptions whole_fp
end
end VeriTile.Bench.Repairs.FlashAttnScalarABIChecked
