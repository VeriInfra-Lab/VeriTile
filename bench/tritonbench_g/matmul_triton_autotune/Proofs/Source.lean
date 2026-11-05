import bench.tritonbench_g.matmul_triton_autotune.Proofs.Support.SourceBindings

/-! Proof-only decomposition of the complete typed quotation in Kernels.lean.
The original helper names and definitions are retained for existing proofs. -/
namespace VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Repair.Source
open VeriTile.Triton FP.Structural
open FPValue
noncomputable section

def prelude (p : Parameters) := (kernel p).surfaceBody.take 22
def body (p : Parameters) : List ComputeStmt :=
  match (kernel p).surfaceBody[22]? with
  | some (ComputeStmt.forRangeInt _ _ _ _ code) => code
  | _ => []
def epilogue (p : Parameters) := (kernel p).surfaceBody.drop 23

def loopStatement (p : Parameters) : ComputeStmt :=
  ((kernel p).surfaceBody[22]?).getD (.effectMarker "missing source loop")
theorem decomposition (p : Parameters) :
    (kernel p).surfaceBody=prelude p ++ [loopStatement p] ++ epilogue p := rfl


end
end VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Repair.Source
