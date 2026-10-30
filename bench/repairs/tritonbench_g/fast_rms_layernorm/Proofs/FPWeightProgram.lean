import bench.repairs.tritonbench_g.fast_rms_layernorm.Proofs.FPWeightReference

namespace VeriTile.Bench.Repairs.FastRmsLayernorm.FPWeightProgram
open VeriTile.Triton
noncomputable section

def prefixCode (h : Call) (K : Nat) := (Kernels.weightBackward h K).surfaceBody.take 3
def loopBody (h : Call) (K : Nat) : List ComputeStmt :=
  match ((Kernels.weightBackward h K).surfaceBody.drop 3).head? with
  | some (.forLoop _ _ body) => body
  | some (.forRangeDyn _ _ _ _ body) => body
  | some (.forRangeInt _ _ _ _ body) => body
  | _ => []
def tailCode (h : Call) (K : Nat) := (Kernels.weightBackward h K).surfaceBody.drop 4

theorem source_body (h : Call) (K : Nat) :
    (Kernels.weightBackward h K).surfaceBody = prefixCode h K ++
      [.forRangeInt "row" (.constInt 0) (.castNatToInt (.constNat h.rows)) (.constNat 1) (loopBody h K)] ++ tailCode h K := by
  rfl

#axiomsClean source_body
end
end VeriTile.Bench.Repairs.FastRmsLayernorm.FPWeightProgram
