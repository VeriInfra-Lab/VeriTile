import bench.tritonbench_g.chunk_delta_fwd.Proofs.PairCorrect

namespace VeriTile.Bench.Repairs.ChunkDeltaFwdStaticDot.FPEquiv
open VeriTile Triton FP.Structural ChunkDeltaFwd
open VeriTile.Bench.Repairs.ChunkGlaSimple.ScalarGraph (WordPrimitives)
open scoped VeriTile.Triton.FP.Structural.ProgramIO
open scoped VeriTile.Spec
noncomputable section

/-- Constant-fold state/history address products in the complete selected
FP32 kernel. All arbitrary input words, optional states, ordered dot/FMA
semantics, the full CTA permutation, source safety, successful execution,
all three output channels and both exterior frames are retained. No
floating arithmetic rewrite or experimental rule is used. -/
specification chunk_delta_addressFold_equiv (I : WordPrimitives) (p : Parameters)
    (shape : Geometry.ShapeValid p) (schedule : List PID) (order : schedule.Perm (grid p)) :
    PairIO.original I p schedule ≡[[]] PairIO.addressFold I p schedule :=
  ProgramIO.Implementation.of_equivalent _ (PairCorrect.equivalent I p shape schedule order)

#axiomsClean chunk_delta_addressFold_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkDeltaFwdStaticDot.FPEquiv
