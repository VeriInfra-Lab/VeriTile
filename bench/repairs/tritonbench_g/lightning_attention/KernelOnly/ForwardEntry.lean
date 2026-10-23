import bench.repairs.tritonbench_g.lightning_attention.KernelOnly.Parts
namespace VeriTile.Bench.Repairs.LightningAttention.KernelOnly
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive)
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
  causal : s.regs .int [p.BT,p.BT] "index"=some (fun (i : TileIndex [p.BT,p.BT]) => (i.1.val : Int) - (i.2.1.val : Int))
def FwdRows (p : Parameters) (block : Nat) (s : State α) : Prop :=
  s.regs .nat [p.BT] "off_block"=some (fun i => block*p.BT+i.1.val)
def FwdCarry (p : Parameters) (carry : Matrix α p.D p.BM) (s : State α) : Prop :=
  s.regs .real [p.D,p.BM] "kv"=some carry

def fwdEntry (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (s : State α) :
    {t : State α // run (engine (withDot M D)) (fwdParts p).before s=some t} := by
  refine ⟨?_,?_⟩
  swap
  simp [fwdParts,forward,ComputeKernel.surfaceBody,Source.dynamicRange?,Option.map,Option.getD,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,engine,withDot,
    Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,
    TileShape.axisDim,Region.cast,State.setReg]
  rfl

theorem fwdEntry_memory (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (s : State α) :
    (fwdEntry M D p s).val.mem=s.mem := by simp [fwdEntry]
theorem fwdEntry_ready (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (s : State α) :
    FwdReady p ⟨s.pids 0,s.pids 1⟩ s.mem (fwdEntry M D p s).val ∧
    FwdRows p 0 (fwdEntry M D p s).val ∧
    FwdCarry p (fun _ => zero M) (fwdEntry M D p s).val := by
  refine ⟨?_,?_,?_⟩
  · constructor
    · rw [fwdEntry_memory]; exact Inputs.refl _
    all_goals simp [fwdEntry,FP.Structural.bop,Tile.bop,Broadcast.leftIndex,Broadcast.rightIndex]
    all_goals
      funext i
      simp [FP.Structural.bop,Tile.bop,Broadcast.leftIndex,Broadcast.rightIndex]
  · simp [FwdRows,fwdEntry]
  · simp [FwdCarry,fwdEntry,zero]
#axiomsClean fwdEntry_memory
#axiomsClean fwdEntry_ready
end
end VeriTile.Bench.Repairs.LightningAttention.KernelOnly
