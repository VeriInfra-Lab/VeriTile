import bench.tritonbench_g.attention_fwd_triton3.Proofs.RealModelImplementation

namespace VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.LogicalReader
open VeriTile Triton
open VeriTile.Bench.Repairs.AttentionFwdTriton3.Host
noncomputable section

theorem model_correct (c : Call) (valid : Valid c) (f : LogicalIO.Inputs c → LogicalIO.Outputs c)
    (connection : (LogicalIO.io c).lift f=RealInterface.modelFormula c) :
    (LogicalIO.io c).Implements f := by
  change (RealInterface.modelIO c).Implements ((LogicalIO.io c).lift f)
  rw [connection]
  exact RealInterface.model_implements c valid

#axiomsClean model_correct
end
end VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.LogicalReader
