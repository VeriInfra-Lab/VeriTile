import bench.repairs.tritonbench_g.lightning_attention.Proofs.InterReverseEntry
import bench.repairs.tritonbench_g.lightning_attention.Proofs.ForwardBody
namespace VeriTile.Bench.Repairs.LightningAttention
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive withDot)
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy (masked_scatter_memory)
noncomputable section
set_option maxHeartbeats 24000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]
def keyValueSubMemory (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (head row block : Nat)
    (carry : Matrix α p.D p.E) (m : Memory α) : Memory α :=
  if block=0 then m else keyValueUpdate M D p head row carry m

theorem keyValueSubMemory_inputs (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (head row block : Nat)
    (carry : Matrix α p.D p.E) (m : Memory α) :
    Inputs (keyValueSubMemory M D p head row block carry m) m := by
  unfold keyValueSubMemory
  split
  · exact Inputs.refl _
  · exact (writeRows_inputs _ _ (by decide) _ _ _ _ _ _ _).trans
      (writeTrans_inputs _ _ (by decide) _ _ _ _ _ _ _)

def interReverseUpdate (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (head row1 row2 block sub : Nat) (input : Memory α) (carry : Matrix α p.D p.E)
    (s : State α) (ctx : InterRReady p head (row1+p.CB) row2 input carry s)
    (hi : s.regs .nat [] "reverse_i"=some (fun _ => block)) (hbnd : block<p.NT) :
    {t : State α // run (engine (withDot M D)) (interRUpdateParts p).body
      (s.setReg "reverse_j" .nat [] (fun _ => sub))=some t} := by
  have hk := ctx.inputs "key" (by simp)
  have hv := ctx.inputs "value" (by simp)
  have hc : (0<block) ↔ block≠0 := by omega
  by_cases hb : block=0
  all_goals
    refine ⟨?_,?_⟩
    swap
    simp [interRUpdateParts,interReverseParts,interForwardParts,inter,ComputeKernel.surfaceBody,Source.dynamicRange?,Option.map,Option.getD,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,engine,withDot,
      Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
      numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,
      TileShape.axisDim,Region.cast,State.setReg,FP.Structural.store,
      ctx.key,ctx.value,ctx.dk,ctx.dv,ctx.rows1,ctx.total,hi,hk,hv,hc,hb,back_address,back_rows]
    rfl

theorem interReverseUpdate_memory (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (head row1 row2 block sub : Nat) (input : Memory α) (carry : Matrix α p.D p.E)
    (s : State α) (ctx : InterRReady p head (row1+p.CB) row2 input carry s)
    (hi : s.regs .nat [] "reverse_i"=some (fun _ => block)) (hbnd : block<p.NT) :
    (interReverseUpdate M D p head row1 row2 block sub input carry s ctx hi hbnd).val.mem=
      keyValueSubMemory M D p head row1 block carry s.mem := by
  have hk := ctx.inputs "key" (by simp)
  have hv := ctx.inputs "value" (by simp)
  unfold keyValueSubMemory keyValueUpdate writeRows writeTrans scatter ChunkDeltaFwd.scatter
    subK subVT gradientRows gradientTrans
  by_cases hb : block=0
  all_goals simp [interReverseUpdate,hb,masked_scatter_memory,pair_eta,
    read,zero,add,cast,dot,FP.Structural.bop,Tile.bop,Broadcast.leftIndex,Broadcast.rightIndex,hk,hv,
    back_address,back_rows]
  all_goals norm_num

theorem interReverseUpdate_ready (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (head row1 row2 block sub : Nat) (input : Memory α) (carry : Matrix α p.D p.E)
    (s : State α) (ctx : InterRReady p head (row1+p.CB) row2 input carry s)
    (hi : s.regs .nat [] "reverse_i"=some (fun _ => block)) (hbnd : block<p.NT) :
    let t := (interReverseUpdate M D p head row1 row2 block sub input carry s ctx hi hbnd).val
    InterRReady p head row1 row2 input carry t ∧
      t.regs .nat [] "reverse_i"=some (fun _ => block) := by
  dsimp only
  refine ⟨?_,?_⟩
  · constructor
    · rw [interReverseUpdate_memory]
      exact (keyValueSubMemory_inputs _ _ _ _ _ _ _ _).trans ctx.inputs
    all_goals by_cases hb : block=0
    all_goals simp [interReverseUpdate,hb,ctx.query,ctx.incoming,ctx.rows2,ctx.total,
      FP.Structural.bop,Tile.bop,Broadcast.leftIndex,Broadcast.rightIndex,back_address,back_rows]
    all_goals funext i
    all_goals simp [FP.Structural.bop,Tile.bop,Broadcast.leftIndex,Broadcast.rightIndex,back_address,back_rows]
  · by_cases hb : block=0 <;> simp [interReverseUpdate,hi,hb]
#axiomsClean interReverseUpdate_memory
#axiomsClean interReverseUpdate_ready
end
end VeriTile.Bench.Repairs.LightningAttention
