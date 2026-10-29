import bench.tritonbench_g.mixed_sparse_attention.Proofs.FPConfig
namespace VeriTile.Bench.TritonBenchG.MixedSparseAttention.FPStages
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

def parsedBlockBody (c : Config) : List ComputeStmt :=
  let code : List ComputeStmt := parsedActiveBody c
  match code[21]? with
  | some (ComputeStmt.forRangeDyn _ _ _ _ body) => body
  | _ => []
def parsedColumnBody (c : Config) : List ComputeStmt :=
  let code : List ComputeStmt := parsedActiveBody c
  match code[23]? with
  | some (ComputeStmt.forRangeDyn _ _ _ _ body) => body
  | _ => []

/-- Exact primitive nodes in both original loops, retaining precision,
shapes, operands and reduction axis; no local AST replacement remains. -/
theorem parser_maximum_nodes (c : Config) :
    (parsedBlockBody c)[10]? = some (maximumAssignment c.BM c.BN) ∧
    (parsedColumnBody c)[8]? = some (maximumAssignment c.BM c.BN) := by
  constructor <;> simp [parsedBlockBody,parsedColumnBody,parsedActiveBody,parsedKernel,source,
    ComputeKernel.surfaceBody,maximumAssignment]

theorem local_expansion_exact (c : Config) :
    blockBody c = parsedBlockBody c ∧ columnBody c = parsedColumnBody c := by
  constructor <;> rfl

#axiomsClean parser_maximum_nodes
#axiomsClean local_expansion_exact
end
end VeriTile.Bench.TritonBenchG.MixedSparseAttention.FPStages
