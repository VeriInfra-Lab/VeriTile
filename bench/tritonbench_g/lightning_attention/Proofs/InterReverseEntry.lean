import bench.tritonbench_g.lightning_attention.Proofs.InterForwardEntry
namespace VeriTile.Bench.Repairs.LightningAttention
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive withDot)
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]
structure InterRReady (p : Parameters) (head row1 row2 : Nat) (input : Memory α)
    (carry : Matrix α p.D p.E) (s : State α) : Prop where
  inputs : Inputs s.mem input
  key : s.regs .ptr [p.CB,p.D] "K_block_ptr"=some (fun i => (("key" : RegionName),head*p.N*p.D+(row1+i.1.val)*p.D+i.2.1.val))
  value : s.regs .ptr [p.E,p.CB] "V_trans_block_ptr"=some (fun i => (("value" : RegionName),head*p.N*p.E+(row1+i.2.1.val)*p.E+i.1.val))
  dk : s.regs .ptr [p.D,p.CB] "DK_trans_block_ptr"=some (fun i => (("keyGradient" : RegionName),head*p.N*p.D+(row1+i.2.1.val)*p.D+i.1.val))
  dv : s.regs .ptr [p.CB,p.E] "DV_block_ptr"=some (fun i => (("valueGradient" : RegionName),head*p.N*p.E+(row1+i.1.val)*p.E+i.2.1.val))
  query : s.regs .ptr [p.D,p.CB] "Q_trans_block_ptr"=some (fun i => (("query" : RegionName),head*p.N*p.D+(row2+i.2.1.val)*p.D+i.1.val))
  incoming : s.regs .ptr [p.CB,p.E] "DO_block_ptr"=some (fun i => (("incoming" : RegionName),head*p.N*p.E+(row2+i.1.val)*p.E+i.2.1.val))
  rows1 : s.regs .nat [p.CB] "off_block1"=some (fun i => row1+i.1.val)
  rows2 : s.regs .nat [p.CB] "off_block2"=some (fun i => row2+i.1.val)
  total : s.regs .real [p.D,p.E] "dkv"=some carry

def InterRDelta (p : Parameters) (acc : Matrix α p.D p.E) (s : State α) : Prop :=
  s.regs .real [p.D,p.E] "dkv_current"=some acc

/-- Pointer subtraction is only rewritten after the loop invariant exhibits
one complete remaining subblock; this avoids assuming natural subtraction wraps. -/
theorem back_address (base row cb lane width col : Nat) :
    base+(row+cb+lane)*width+col-cb*width=base+(row+lane)*width+col := by
  simp only [Nat.add_mul]
  omega
theorem back_rows (row cb lane : Nat) : row+cb+lane-cb=row+lane := by omega
#axiomsClean back_address
end
end VeriTile.Bench.Repairs.LightningAttention
