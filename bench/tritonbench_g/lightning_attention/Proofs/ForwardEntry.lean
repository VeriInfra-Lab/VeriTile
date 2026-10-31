import bench.tritonbench_g.lightning_attention.Proofs.Model
namespace VeriTile.Bench.Repairs.LightningAttention
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive withDot)
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]
structure FwdReady (p : Parameters) (pid : PID) (input : Memory α) (s : State α) : Prop where
  inputs : Inputs s.mem input
  query : s.regs .ptr [1,p.D] "Q_block_ptr"=some (fun (i : TileIndex [1,p.D]) => (("query" : RegionName),pid.head*p.N*p.D+i.2.1.val))
  key : s.regs .ptr [p.D,1] "K_trans_block_ptr"=some (fun (i : TileIndex [p.D,1]) => (("key" : RegionName),pid.head*p.N*p.D+i.1.val))
  value : s.regs .ptr [1,p.BM] "V_block_ptr"=some (fun (i : TileIndex [1,p.BM]) => (("value" : RegionName),pid.head*p.N*p.E+pid.channel*p.BM+i.2.1.val))
  output : s.regs .ptr [1,p.BM] "O_block_ptr"=some (fun (i : TileIndex [1,p.BM]) => (("output" : RegionName),pid.head*p.N*p.E+pid.channel*p.BM+i.2.1.val))
  causal : s.regs .bool [p.BT,p.BT] "index"=some (fun (i : TileIndex [p.BT,p.BT]) => decide (i.2.1.val ≤ i.1.val))
def FwdRows (p : Parameters) (block : Nat) (s : State α) : Prop :=
  s.regs .nat [p.BT] "off_block"=some (fun i => block*p.BT+i.1.val)
def FwdCarry (p : Parameters) (carry : Matrix α p.D p.BM) (s : State α) : Prop :=
  s.regs .real [p.D,p.BM] "kv"=some carry

end
end VeriTile.Bench.Repairs.LightningAttention
