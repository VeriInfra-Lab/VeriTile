import bench.repairs.tritonbench_g.attn_fwd_causal.IO

namespace VeriTile.Bench.Repairs.AttnFwdCausal.RealIO
open VeriTile Triton FP.Structural Source Whole
noncomputable section

def formula (p : Parameters) (work : List Program) :
    (program p work).Inputs → (program p work).Outputs :=
  fun vals o=>Fin.cases (Reference.mathematical p vals) (fun i=>Fin.elim0 i) o

end
end VeriTile.Bench.Repairs.AttnFwdCausal.RealIO
