import bench.tritonbench_g.attention_fwd_triton3.Proofs.BodyStages
import bench.tritonbench_g.attention_fwd_triton3.Proofs.CSEParts

namespace VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.KeyMaskCSE
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.Reference
open VeriTile.Bench.Repairs.AttentionFwdTriton3.Host
noncomputable section
variable {α : Type} [Inhabited α]

def ScoresReady (c : Call) (o : Ops α) (a : Inputs α) (cur : Int) (old : Carry c α)
    (s : State α) : Prop :=
  Reference.ScoresReady c o a cur old s ∧
  (c.params.evenN=0 → s.regs .bool [c.BN] "__veritile_statement_jit_local_11_key_mask"=
    some (fun i=>decide ((i.1.val : Int)+cur<c.NK)))

end
end VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.KeyMaskCSE
