import bench.repairs.tritonbench_g.lightning_attention.KernelOnly.InterForwardEntry
import bench.repairs.tritonbench_g.lightning_attention.KernelOnly.ForwardBody
namespace VeriTile.Bench.Repairs.LightningAttention.KernelOnly
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive)
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]
def querySubDelta (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (head row : Nat)
    (input : Memory α) (acc : Matrix α p.E p.D) : Matrix α p.E p.D :=
  fun i => add M (acc i) (dot D (subVT M p head row input) (subK M p head row input) i)

def interDeltaBody (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (head row1 row2 sub : Nat) (input : Memory α) (carry acc : Matrix α p.E p.D)
    (s : State α) (ctx : InterFReady p head row1 row2 input carry s) (ha : InterFDelta p acc s) :
    {t : State α // run (engine (withDot M D)) (interFDeltaParts p).body
      (s.setReg "j" .nat [] (fun _ => sub))=some t} := by
  have hk := ctx.inputs "key" (by simp)
  have hv := ctx.inputs "value" (by simp)
  simp only [InterFDelta] at ha
  refine ⟨?_,?_⟩
  swap
  simp [interFDeltaParts,interFQueryParts,interForwardParts,inter,ComputeKernel.surfaceBody,Source.dynamicRange?,Option.map,Option.getD,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,engine,withDot,
    Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,
    TileShape.axisDim,Region.cast,State.setReg,ctx.key,ctx.value,ctx.rows2,ha,hk,hv]
  rfl

theorem interDeltaBody_memory (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (head row1 row2 sub : Nat) (input : Memory α) (carry acc : Matrix α p.E p.D)
    (s : State α) (ctx : InterFReady p head row1 row2 input carry s) (ha : InterFDelta p acc s) :
    (interDeltaBody M D p head row1 row2 sub input carry acc s ctx ha).val.mem=s.mem := by
  simp [interDeltaBody]

theorem interDeltaBody_ready (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (head row1 row2 sub : Nat) (input : Memory α) (carry acc : Matrix α p.E p.D)
    (s : State α) (ctx : InterFReady p head row1 row2 input carry s) (ha : InterFDelta p acc s) :
    let t := (interDeltaBody M D p head row1 row2 sub input carry acc s ctx ha).val
    InterFReady p head row1 (row2+p.CB) input carry t ∧
      InterFDelta p (querySubDelta M D p head row2 input acc) t := by
  dsimp only
  refine ⟨?_,?_⟩
  · constructor
    · rw [interDeltaBody_memory]; exact ctx.inputs
    all_goals simp [interDeltaBody,ctx.qkOffset,ctx.vOffset,ctx.oOffset,ctx.query,ctx.incoming,ctx.rows1,ctx.total]
    all_goals funext i
    all_goals simp [FP.Structural.bop,Tile.bop,Broadcast.leftIndex,Broadcast.rightIndex,
      Nat.add_mul,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm]
    all_goals omega
  · unfold InterFDelta querySubDelta subVT subK
    simp [interDeltaBody,read,zero,add,cast,dot,FP.Structural.bop,Tile.bop,Broadcast.leftIndex,Broadcast.rightIndex,pair_eta]
    funext i
    simp [FP.Structural.bop,Tile.bop,Broadcast.leftIndex,Broadcast.rightIndex,pair_eta]
    norm_num
#axiomsClean interDeltaBody_ready
end
end VeriTile.Bench.Repairs.LightningAttention.KernelOnly
