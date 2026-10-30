import bench.tritonbench_g.matmul_triton_autotune.Kernels

/-! Proof-only decomposition of the complete typed quotation in Kernels.lean.
The original helper names and definitions are retained for existing proofs. -/
namespace VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Repair.Source
open VeriTile.Triton FP.Structural
open FPValue
noncomputable section

def prelude (p : Parameters) := (kernel p).surfaceBody.take 13
def body (p : Parameters) : List ComputeStmt :=
  match (kernel p).surfaceBody[13]? with
  | some (ComputeStmt.forRange _ _ _ _ code) => code
  | _ => []
def epilogue (p : Parameters) := (kernel p).surfaceBody.drop 14

theorem decomposition (p : Parameters) :
    (kernel p).surfaceBody=prelude p ++ [.forRange "kk" 0 p.numKBlocks 1 (body p)] ++ epilogue p := rfl


end
end VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Repair.Source
