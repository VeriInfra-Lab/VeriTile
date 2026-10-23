import bench.repairs.tritonbench_g.lightning_attention.Proofs.InterReverseEntry
import bench.repairs.tritonbench_g.lightning_attention.Proofs.ForwardBody
namespace VeriTile.Bench.Repairs.LightningAttention
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive withDot)
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]
def keyValueSubDelta (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (head row : Nat)
    (input : Memory α) (acc : Matrix α p.D p.E) : Matrix α p.D p.E :=
  fun i => add M (acc i) (dot D (subQT M p head row input) (subG M p head row input) i)

def interReverseDelta (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (head row1 row2 sub : Nat) (input : Memory α) (carry acc : Matrix α p.D p.E)
    (s : State α) (ctx : InterRReady p head row1 (row2+p.CB) input carry s) (ha : InterRDelta p acc s) :
    {t : State α // run (engine (withDot M D)) (interRDeltaParts p).body
      (s.setReg "reverse_j" .nat [] (fun _ => sub))=some t} := by
  have hq := ctx.inputs "query" (by simp)
  have hg := ctx.inputs "incoming" (by simp)
  simp only [InterRDelta] at ha
  refine ⟨?_,?_⟩
  swap
  simp [interRDeltaParts,interRUpdateParts,interReverseParts,interForwardParts,inter,ComputeKernel.surfaceBody,Source.dynamicRange?,Option.map,Option.getD,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,engine,withDot,
    Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,
    TileShape.axisDim,Region.cast,State.setReg,ctx.query,ctx.incoming,ctx.rows2,ha,hq,hg,
    back_address,back_rows]
  rfl

theorem interReverseDelta_memory (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (head row1 row2 sub : Nat) (input : Memory α) (carry acc : Matrix α p.D p.E)
    (s : State α) (ctx : InterRReady p head row1 (row2+p.CB) input carry s) (ha : InterRDelta p acc s) :
    (interReverseDelta M D p head row1 row2 sub input carry acc s ctx ha).val.mem=s.mem := by
  simp [interReverseDelta]

theorem interReverseDelta_ready (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (head row1 row2 sub : Nat) (input : Memory α) (carry acc : Matrix α p.D p.E)
    (s : State α) (ctx : InterRReady p head row1 (row2+p.CB) input carry s) (ha : InterRDelta p acc s) :
    let t := (interReverseDelta M D p head row1 row2 sub input carry acc s ctx ha).val
    InterRReady p head row1 row2 input carry t ∧
      InterRDelta p (keyValueSubDelta M D p head row2 input acc) t := by
  dsimp only
  refine ⟨?_,?_⟩
  · constructor
    · rw [interReverseDelta_memory]; exact ctx.inputs
    all_goals simp [interReverseDelta,ctx.key,ctx.value,ctx.dk,ctx.dv,ctx.rows1,ctx.total,
      FP.Structural.bop,Tile.bop,Broadcast.leftIndex,Broadcast.rightIndex,back_address,back_rows]
    all_goals funext i
    all_goals simp [FP.Structural.bop,Tile.bop,Broadcast.leftIndex,Broadcast.rightIndex,back_address,back_rows]
  · unfold InterRDelta keyValueSubDelta subQT subG
    simp [interReverseDelta,read,zero,add,cast,dot,FP.Structural.bop,Tile.bop,Broadcast.leftIndex,Broadcast.rightIndex,pair_eta,
      back_address,back_rows]
    funext i
    simp [FP.Structural.bop,Tile.bop,Broadcast.leftIndex,Broadcast.rightIndex,pair_eta,back_address,back_rows]
    norm_num
#axiomsClean interReverseDelta_ready
end
end VeriTile.Bench.Repairs.LightningAttention
