import bench.tritonbench_g.embedding_triton_kernel.Proofs.Rewrite

namespace VeriTile.Bench.Repairs.EmbeddingTritonKernel.FPEquiv
open VeriTile Triton FP.ProfiledRewrite
open scoped VeriTile.Spec
noncomputable section

abbrev assumptions := Rewrite.assumptions
abbrev original := Rewrite.original
abbrev indexCommuted := Rewrite.indexCommuted

/-- Exact integer commutativity in the actual FP32/int32 sources, preserving
the entire 64-iteration execution. No floating transformation law is required. -/
specification embedding_index_commuted_equiv (p : Parameters) :
    original p ≡[assumptions] indexCommuted p := Rewrite.equivalent p

#print_fp_assumptions embedding_index_commuted_equiv
#axiomsClean embedding_index_commuted_equiv
#stmtSurfaceSubset embedding_index_commuted_equiv ⊆
  [assumptions,indexCommuted,original,Parameters,Spec.Equiv,
   FP.ProfiledRewrite.Program,FP.ProfiledRewrite.instProgramSyntaxProgram]
end
end VeriTile.Bench.Repairs.EmbeddingTritonKernel.FPEquiv
