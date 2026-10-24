import bench.repairs.tritonbench_g.attn_fwd_triton_tile_checked.NativeIO
import bench.repairs.tritonbench_g.attn_fwd_triton_tile_checked.Kernels
import VeriTile.Triton.Float.ProgramEquivalenceSyntax

namespace VeriTile.Bench.Repairs.AttnFwdTritonTileChecked
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open AttnFwdTriton.TypedExecution Graph
noncomputable section

namespace PairConnection
def kernel (p : Config) (BD : Nat) : ComputeKernel :=
  quotientRemainder p.Q p.K p.V p.QS p.KS p.Out
    p.queryBatch p.queryHead p.queryRow p.queryDim
    p.keyBatch p.keyHead p.keyRow p.keyDim
    p.valueBatch p.valueHead p.valueRow p.valueDim
    p.outputBatch p.outputHead p.outputRow p.outputDim
    p.batches p.heads p.context p.headDim 128 64 BD p.stage p.validStage

end PairConnection

namespace PairIO
/-- Same actual full attention invocation and fixed native graph. -/
def invocation (p : Config) (pid : ProgramIndex p) : Invocation Expr :=
  { NativeInterface.invocation p pid with
    kernel := PairConnection.kernel p (AttnFwdTriton.HostShape.blockD p.headDim) }

def candidate (p : Config) : ProgramIO.Variant (NativeInterface.programIO p) where
  schedule := fun _ => (NativeInterface.fullGrid p).map (invocation p)
  observe := (NativeInterface.programIO p).observe

end PairIO
end
end VeriTile.Bench.Repairs.AttnFwdTritonTileChecked
