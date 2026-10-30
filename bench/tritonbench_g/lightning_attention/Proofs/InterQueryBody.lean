import bench.tritonbench_g.lightning_attention.Proofs.InterForwardEntry
import bench.tritonbench_g.lightning_attention.Proofs.ForwardBody
namespace VeriTile.Bench.Repairs.LightningAttention
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive withDot)
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy (masked_scatter_memory)
noncomputable section
set_option maxHeartbeats 24000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]
def querySubMemory (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (head row block : Nat)
    (carry : Matrix α p.E p.D) (m : Memory α) : Memory α :=
  if block=0 then m else queryUpdate M D p head row carry m

theorem querySubMemory_inputs (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (head row block : Nat)
    (carry : Matrix α p.E p.D) (m : Memory α) :
    Inputs (querySubMemory M D p head row block carry m) m := by
  unfold querySubMemory
  split
  · exact Inputs.refl _
  · exact writeRows_inputs _ _ (by decide) _ _ _ _ _ _ _

def interQueryBody (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (head row1 row2 block sub : Nat) (input : Memory α) (carry : Matrix α p.E p.D)
    (s : State α) (ctx : InterFReady p head row1 row2 input carry s)
    (hi : s.regs .nat [] "i"=some (fun _ => block)) :
    {t : State α // run (engine (withDot M D)) (interFQueryParts p).body
      (s.setReg "j" .nat [] (fun _ => sub))=some t} := by
  have hg := ctx.inputs "incoming" (by simp)
  by_cases hb : 0<block
  all_goals
    refine ⟨?_,?_⟩
    swap
    simp [interFQueryParts,interForwardParts,inter,ComputeKernel.surfaceBody,Source.dynamicRange?,Option.map,Option.getD,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,engine,withDot,
    Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,
    TileShape.axisDim,Region.cast,State.setReg,FP.Structural.store,
    ctx.query,ctx.incoming,ctx.rows1,ctx.total,hi,hg,hb]
    rfl

theorem interQueryBody_memory (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (head row1 row2 block sub : Nat) (input : Memory α) (carry : Matrix α p.E p.D)
    (s : State α) (ctx : InterFReady p head row1 row2 input carry s)
    (hi : s.regs .nat [] "i"=some (fun _ => block)) :
    (interQueryBody M D p head row1 row2 block sub input carry s ctx hi).val.mem=
      querySubMemory M D p head row1 block carry s.mem := by
  have hg := ctx.inputs "incoming" (by simp)
  unfold querySubMemory queryUpdate writeRows scatter ChunkDeltaFwd.scatter subG gradientRows
  by_cases hb : block=0
  all_goals simp [interQueryBody,hb,show (0<block) ↔ block≠0 by omega,
    masked_scatter_memory,pair_eta,read,zero,add,cast,dot,FP.Structural.bop,
    Tile.bop,Broadcast.leftIndex,Broadcast.rightIndex,hg]
  all_goals norm_num

theorem interQueryBody_ready (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (head row1 row2 block sub : Nat) (input : Memory α) (carry : Matrix α p.E p.D)
    (s : State α) (ctx : InterFReady p head row1 row2 input carry s)
    (hi : s.regs .nat [] "i"=some (fun _ => block)) :
    let t := (interQueryBody M D p head row1 row2 block sub input carry s ctx hi).val
    InterFReady p head (row1+p.CB) row2 input carry t ∧
      t.regs .nat [] "i"=some (fun _ => block) := by
  dsimp only
  refine ⟨?_,?_⟩
  · constructor
    · rw [interQueryBody_memory]
      exact (querySubMemory_inputs _ _ _ _ _ _ _ _).trans ctx.inputs
    all_goals by_cases hb : 0<block
    all_goals simp [interQueryBody,hb,ctx.qkOffset,ctx.vOffset,ctx.oOffset,ctx.key,ctx.value,ctx.rows2,ctx.total]
    all_goals funext i
    all_goals simp [FP.Structural.bop,Tile.bop,Broadcast.leftIndex,Broadcast.rightIndex,
      Nat.add_mul,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm]
    all_goals omega
  · by_cases hb : 0<block <;> simp [interQueryBody,hi,hb]
#axiomsClean interQueryBody_memory
#axiomsClean interQueryBody_ready
end
end VeriTile.Bench.Repairs.LightningAttention
