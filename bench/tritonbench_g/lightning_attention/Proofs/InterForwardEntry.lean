import bench.tritonbench_g.lightning_attention.Proofs.InterModel
namespace VeriTile.Bench.Repairs.LightningAttention
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive withDot)
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]
structure InterFReady (p : Parameters) (head row1 row2 : Nat) (input : Memory α)
    (carry : Matrix α p.E p.D) (s : State α) : Prop where
  inputs : Inputs s.mem input
  qkOffset : s.regs .nat [] "qk_offset"=some (fun _ => head*p.N*p.D)
  vOffset : s.regs .nat [] "v_offset"=some (fun _ => head*p.N*p.E)
  oOffset : s.regs .nat [] "o_offset"=some (fun _ => head*p.N*p.E)
  query : s.regs .ptr [p.CB,p.D] "DQ_block_ptr"=some (fun i => (("queryGradient" : RegionName),head*p.N*p.D+(row1+i.1.val)*p.D+i.2.1.val))
  incoming : s.regs .ptr [p.CB,p.E] "DO_block_ptr"=some (fun i => (("incoming" : RegionName),head*p.N*p.E+(row1+i.1.val)*p.E+i.2.1.val))
  key : s.regs .ptr [p.CB,p.D] "K_block_ptr"=some (fun i => (("key" : RegionName),head*p.N*p.D+(row2+i.1.val)*p.D+i.2.1.val))
  value : s.regs .ptr [p.E,p.CB] "V_trans_block_ptr"=some (fun i => (("value" : RegionName),head*p.N*p.E+(row2+i.2.1.val)*p.E+i.1.val))
  rows1 : s.regs .nat [p.CB] "off_block1"=some (fun i => row1+i.1.val)
  rows2 : s.regs .nat [p.CB] "off_block2"=some (fun i => row2+i.1.val)
  total : s.regs .real [p.E,p.D] "kv_trans"=some carry

def InterFDelta (p : Parameters) (acc : Matrix α p.E p.D) (s : State α) : Prop :=
  s.regs .real [p.E,p.D] "kv_trans_current"=some acc

end
end VeriTile.Bench.Repairs.LightningAttention
