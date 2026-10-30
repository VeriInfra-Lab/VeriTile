import bench.repairs.tritonbench_g.rope_embedding.Proofs.Pair
/-! Two actual complete FP32 sources. The existing candidate commutes only
an exact integer head-group product. The native formula result remains an
ordinary optional theorem in Misc/NativeCorrect.lean. -/
namespace VeriTile.Bench.Repairs.RopeEmbedding
open VeriTile.Triton PairIO
open scoped VeriTile.Spec

specification rope_embedding_index_commuted_equiv (p : Source.Parameters) (back : Bool) :
    original p back ≡[assumptions] indexCommuted p back :=
  PairProof.equivalent p back

#print_fp_assumptions rope_embedding_index_commuted_equiv
#axiomsClean rope_embedding_index_commuted_equiv
#stmtSurfaceSubset rope_embedding_index_commuted_equiv ⊆
  [Source.Parameters,PairIO.original,PairIO.indexCommuted,PairIO.assumptions,
    VeriTile.Spec.Equiv,FP.ProfiledRewrite.Program,FP.ProfiledRewrite.instProgramSyntaxProgram]
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.RopeEmbedding
