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

def interRUpdateParts (p : Parameters) : Source.DynamicRange :=
  (Source.dynamicRange? (interReverseParts p).body "reverse_j").getD ⟨[],.constNat 0,.constNat 0,.constNat 0,[],[]⟩
def interRDeltaParts (p : Parameters) : Source.DynamicRange :=
  (Source.dynamicRange? (interRUpdateParts p).after "reverse_j").getD ⟨[],.constNat 0,.constNat 0,.constNat 0,[],[]⟩
def InterRDelta (p : Parameters) (acc : Matrix α p.D p.E) (s : State α) : Prop :=
  s.regs .real [p.D,p.E] "dkv_current"=some acc

def interREntry (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (head row1 row2 : Nat) (input : Memory α) (carry : Matrix α p.E p.D)
    (s : State α) (ctx : InterFReady p head row1 row2 input carry s) :
    {t : State α // run (engine (withDot M D)) (interReverseParts p).before s=some t} := by
  refine ⟨?_,?_⟩
  swap
  simp [interReverseParts,interForwardParts,inter,ComputeKernel.surfaceBody,Source.dynamicRange?,Option.map,Option.getD,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,engine,withDot,
    Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,
    TileShape.axisDim,Region.cast,State.setReg,ctx.qkOffset,ctx.vOffset,ctx.oOffset]
  rfl

theorem interREntry_memory (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (head row1 row2 : Nat) (input : Memory α) (carry : Matrix α p.E p.D)
    (s : State α) (ctx : InterFReady p head row1 row2 input carry s) :
    (interREntry M D p head row1 row2 input carry s ctx).val.mem=s.mem := by simp [interREntry]
theorem interREntry_ready (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (head row1 row2 : Nat) (input : Memory α) (carry : Matrix α p.E p.D)
    (s : State α) (ctx : InterFReady p head row1 row2 input carry s) :
    InterRReady p head (p.NT*p.BT) (p.NT*p.BT) input (fun _ => zero M)
      (interREntry M D p head row1 row2 input carry s ctx).val := by
  constructor
  · rw [interREntry_memory]; exact ctx.inputs
  all_goals simp [interREntry,zero,FP.Structural.bop,Tile.bop,Broadcast.leftIndex,Broadcast.rightIndex]
  all_goals
    funext i
    simp [FP.Structural.bop,Tile.bop,Broadcast.leftIndex,Broadcast.rightIndex,Nat.add_mul,Nat.add_assoc]

/-- Pointer subtraction is only rewritten after the loop invariant exhibits
one complete remaining subblock; this avoids assuming natural subtraction wraps. -/
theorem back_address (base row cb lane width col : Nat) :
    base+(row+cb+lane)*width+col-cb*width=base+(row+lane)*width+col := by
  simp only [Nat.add_mul]
  omega
theorem back_rows (row cb lane : Nat) : row+cb+lane-cb=row+lane := by omega
#axiomsClean interREntry_ready
#axiomsClean back_address
end
end VeriTile.Bench.Repairs.LightningAttention
