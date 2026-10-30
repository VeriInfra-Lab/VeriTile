import bench.repairs.tritonbench_g.chunk_bwd_dqkg.Proofs.TypedPostValues
import bench.repairs.tritonbench_g.chunk_bwd_dqkg.Proofs.TypedScatter
namespace VeriTile.Bench.Repairs.ChunkBwdDqkg.Typed
open VeriTile Triton FP.Structural FP.ProfiledRewrite Whole
open Scatter (masked_scatter_memory masked_readback_active)
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 12000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def qActive (c : Config) (s : State α) (i : TileIndex [64,c.BK]) : Prop :=
  s.pids 1*64+i.1.val<c.T ∧ s.pids 0*c.BK+i.2.1.val<c.K
instance (c : Config) (s : State α) : DecidablePred (qActive c s) := fun _=>inferInstanceAs (Decidable (_ ∧ _))
def gActive (c : Config) (s : State α) (i : TileIndex [64]) : Prop := s.pids 1*64+i.1.val<c.T
instance (c : Config) (s : State α) : DecidablePred (gActive c s) := fun _=>inferInstanceAs (Decidable (_ < _))
def qOffset (c : Config) (s : State α) (i : TileIndex [64,c.BK]) : Nat :=
  s.pids 2*c.layout.s_k_h+(s.pids 1*64+i.1.val)*c.layout.s_k_t+(s.pids 0*c.BK+i.2.1.val)
def gOffset (c : Config) (s : State α) (i : TileIndex [64]) : Nat :=
  (s.pids 0*s.numPids 2+s.pids 2)*c.T+(s.pids 1*64+i.1.val)

def writeQK (M : Algebra α) (c : Config) (initial current : State α)
    (region : RegionName) (value : TileIndex [64,c.BK] → α) : State α :=
  (TileShape.allIndices [64,c.BK]).foldl (fun t i=>
    if qActive c initial i then t.write region (qOffset c initial i)
      (.mk .fp32 (M.cast (some .fp32) .real .fp32 (value i))) else t) current

def writeG (M : Algebra α) (c : Config) (initial current : State α)
    (value : TileIndex [64] → α) : State α :=
  (TileShape.allIndices [64]).foldl (fun t i=>
    if gActive c initial i then t.write "dg" (gOffset c initial i)
      (.mk .real (M.cast (some .fp32) .real .real (value i))) else t) current

def written (M : Algebra α) (dot : Dot α) (c : Config) (s : State α) (a : Carries α c) : State α :=
  writeG M c s (writeQK M c s (writeQK M c s s "dq" (answerQ M dot c s a)) "dk" (answerK M dot c s a))
    (answerG M dot c s a)

def contextStores (M : Algebra α) (dot : Dot α) (c : Config)
    (initial current : State α) (a : Carries α c)
    (ctx : AnswerContext M dot c initial a current) :
    {t : State α // run (engine (withDot M dot)) ((parts c).after.drop 17) current=some t} := by
  refine ⟨?_,?_⟩
  swap
  simp only [parts,kernel,TypedSource.original,TypedSource.specialize,TypedSource.Storage.dtype,
    ComputeKernel.surfaceBody,Triton.Source.dynamicRange?,Option.map,Option.getD,
    String.reduceEq,beq_self_eq_true,↓reduceIte,List.take,List.drop]
  simp [run,step.eq_3,evalExpr,evalComputeOp,evalOp_unfold,FP.Structural.store,
    Option.bind,engine,withDot,Algebra.withDefaultPrecision,resolvePrecision,
    ComputeDType.eraseDType,ofFloat,toFloat,State.setReg,ctx.dq,ctx.dk,ctx.dg,ctx.pq,ctx.pk,ctx.pg]
  rfl

theorem contextStores_memory (M : Algebra α) (dot : Dot α) (c : Config)
    (initial current : State α) (a : Carries α c)
    (ctx : AnswerContext M dot c initial a current) :
    (contextStores M dot c initial current a ctx).val.mem=(written M dot c initial a).mem := by
  simp [contextStores,written,writeQK,writeG,qActive,gActive,qOffset,gOffset,qPointer,gPointer,
    masked_scatter_memory,ctx.memory,TileShape.indexToList]

theorem stores_memory (M : Algebra α) (dot : Dot α) (c : Config)
    (initial current : State α) (a : Carries α c)
    (ctx : AnswerContext M dot c initial a current) (ready : OutputReady c current) :
    (stores M dot c current ready).val.mem=(written M dot c initial a).mem := by
  have eq := Option.some.inj ((stores M dot c current ready).property.symm.trans
    (contextStores M dot c initial current a ctx).property)
  rw [eq]
  exact contextStores_memory M dot c initial current a ctx

theorem source_memory (M : Algebra α) (dot : Dot α) (c : Config) (s : State α) :
    (finalState M dot c s).mem=(written M dot c s (accumulated M dot c s (valueBlocks c))).mem := by
  exact stores_memory M dot c s _ _
    (post_context M dot c s (loopState M dot c s).val _ (loop_context M dot c s)
      (loopState M dot c s).property)
    (post_ready M dot c (loopState M dot c s).val (loopState M dot c s).property)

#axiomsClean source_memory

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkBwdDqkg.Typed
