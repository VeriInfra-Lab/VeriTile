import bench.repairs.tritonbench_g.attention_llama_scalar_abi_checked.Proofs.Values
import bench.repairs.tritonbench_g.attention_llama_scalar_abi_checked.Proofs.Memory

namespace VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.Source
open VeriTile Triton FP.Structural FP.ProfiledRewrite
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

def address (c : Config) (s : State (Term α)) (i : TileIndex [64,c.D]) : Nat :=
  base c s c.soz c.soh+rows s i.1*c.som+i.2.1.val*c.son

def finish (c : Config) (s t : State (Term α)) (scale : Term α) (n : Nat)
    (a : Carries α c.D) (h : Context c s scale n a t) :
    {u : State (Term α) // run algebra (parts c).after t=some u} := by
  cases hf : c.fp8 <;> cases hc : c.causal
  all_goals
    refine ⟨?_,?_⟩
    swap
    simp [parts,kernel,hf,hc,TypedSource.half,TypedSource.bytes,
      ComputeKernel.surfaceBody,Triton.Source.dynamicRange?,Option.map,Option.getD,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,algebra,store,
      ComputeDType.eraseDType,numeric,integralFloorDiv,integralMod,
      natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,
      TileShape.axisDim,Region.cast,State.setReg,
      h.scalar,h.maximum,h.denominator,h.accumulator,h.query,h.rows,h.columns,
      h.batch,h.head,h.key,h.value,h.memory,h.pids]
    rfl

def stored (c : Config) (s : State (Term α)) (a : Carries α c.D) : State (Term α) :=
  (TileShape.allIndices [64,c.D]).foldl (fun t i=>
    if rows s i.1<c.H then t.write c.Out (address c s i) (.mk .fp16 (.half (a.accumulator i))) else t) s

theorem finish_memory (c : Config) (s t : State (Term α)) (scale : Term α) (n : Nat)
    (a : Carries α c.D) (h : Context c s scale n a t) :
    (finish c s t scale n a h).val.mem=(stored c s a).mem := by
  rcases c with ⟨Q,K,V,Out,sqz,sqh,sqm,sqk,skz,skh,skn,skk,svz,svh,svk,svn,soz,soh,som,son,NH,H,N,SP,D,causal,fp8⟩
  cases fp8 <;> cases causal
  all_goals simp [finish,stored,masked_memory,address,base,rows,bop,
    Broadcast.leftIndex,Broadcast.rightIndex,Function.comp_def,TileShape.indexToList]

def ActiveInjective (c : Config) (s : State (Term α)) : Prop :=
  ∀ i j : TileIndex [64,c.D], rows s i.1<c.H → rows s j.1<c.H →
    address c s i=address c s j → i=j

theorem stored_read (c : Config) (s : State (Term α)) (a : Carries α c.D)
    (hinj : ActiveInjective c s) (i : TileIndex [64,c.D]) (hi : rows s i.1<c.H) :
    (stored c s a).mem c.Out (address c s i)=.mk .fp16 (.half (a.accumulator i)) := by
  apply masked_readback
  · exact TileShape.mem_allIndices _ i
  · exact hi
  · exact hinj

theorem stored_frame (c : Config) (s : State (Term α)) (a : Carries α c.D)
    (r : RegionName) (o : Nat)
    (hmiss : r≠c.Out ∨ ∀ i : TileIndex [64,c.D], rows s i.1<c.H → o≠address c s i) :
    (stored c s a).mem r o=s.mem r o := by
  apply State.masked_scatter_frame
  rcases hmiss with hr|ho
  · exact Or.inl hr
  · exact Or.inr (fun i _=>ho i)

#axiomsClean finish
#axiomsClean finish_memory
#axiomsClean stored_read
#axiomsClean stored_frame
end
end VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.Source
