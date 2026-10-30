import bench.repairs.tritonbench_g.flash_attn.KernelOnly.RealModel

namespace VeriTile.Bench.Repairs.FlashAttn.KernelReal
open VeriTile Triton FP.Structural ActualSource
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 30000

theorem key_check (r : RegionName) (base N D ds rs row : Nat) (i : TileIndex [D,64]) :
    (BlockPtr.mk r base [D,N] [D,64] [ds,rs] [0,row]).inBounds
      (TileShape.indexToList _ i) [1]=decide (row+i.2.1.val<N) := by
  change (decide (row+i.2.1.val<N) && Bool.true)=_
  simp

theorem value_check (r : RegionName) (base N D ds rs row : Nat) (i : TileIndex [64,D]) :
    (BlockPtr.mk r base [N,D] [64,D] [rs,ds] [row,0]).inBounds
      (TileShape.indexToList _ i) [0]=decide (row+i.1.val<N) := by
  change (decide (row+i.1.val<N) && Bool.true)=_
  simp

theorem key_value (x : Layout) (m : Memory ℝ) (pid head start : Nat)
    (i : TileIndex [(parameters x).DIM,64]) :
    ActualSource.key algebra (parameters x) (state x m pid head) start i=
      some (if start+i.2.1.val<x.shape.length then
        logical m .k x.k (head/x.shape.heads) (head%x.shape.heads) (start+i.2.1.val) i.1.val else 0) := by
  by_cases hi : start+i.2.1.val<x.shape.length
  all_goals simp [ActualSource.key,kptr,key_check,state,parameters,stateAt,base,algebra,
    liftMemory,sourceRegion,Real.lift,zero,Cell.read,logical,FlashAttn.address,hi,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm]

theorem value_value (x : Layout) (m : Memory ℝ) (pid head start : Nat)
    (i : TileIndex [64,(parameters x).DIM]) :
    ActualSource.value algebra (parameters x) (state x m pid head) start i=
      some (if start+i.1.val<x.shape.length then
        logical m .v x.v (head/x.shape.heads) (head%x.shape.heads) (start+i.1.val) i.2.1.val else 0) := by
  by_cases hi : start+i.1.val<x.shape.length
  all_goals simp [ActualSource.value,vptr,value_check,state,parameters,stateAt,base,algebra,
    liftMemory,sourceRegion,Real.lift,zero,Cell.read,logical,FlashAttn.address,hi]

theorem score_value (x : Layout) (m : Memory ℝ) (scale : ℝ) (pid head start : Nat)
    (i : TileIndex [128,64]) (active : pid*128+i.1.val<x.shape.length) :
    scores algebra dot (parameters x) (state x m pid head) (some scale) start i=
      if visible x.shape (pid*128+i.1.val) (start+i.2.1.val) then
        some (Spec.score x m scale (head/x.shape.heads) (head%x.shape.heads) (pid*128+i.1.val) (start+i.2.1.val))
      else none := by
  unfold scores dot
  simp only [ofFloat,toFloat,FloatDType.toTileDType]
  have cast : ∀ z : R,algebra.cast (some .fp32) .fp16 .real z=z := fun _=>rfl
  simp only [cast,query_value,key_value]
  simp only [id_eq]
  rw [dot_some (m:=128) (k:=(parameters x).DIM) (n:=64)
    (fun ri e=>if pid*128+ri.val<x.shape.length then
      scale*1.44269504*logical m .q x.q (head/x.shape.heads) (head%x.shape.heads) (pid*128+ri.val) e else 0)
    (fun e ci=>if start+ci.val<x.shape.length then
      logical m .k x.k (head/x.shape.heads) (head%x.shape.heads) (start+ci.val) e else 0) i]
  simp only [algebra,binary,zero,parameters,state,stateAt]
  by_cases hn : start+i.2.1.val<x.shape.length
  · cases hc : x.shape.causal
    · simp [hc,hn,active,visible,WithBot.realAdd,Spec.score,RealSum.sumReal_list]
    · by_cases hle : start+i.2.1.val≤pid*128+i.1.val
      · simp [hc,hn,hle,active,visible,WithBot.realAdd,Spec.score,RealSum.sumReal_list]
      · simp [hc,hn,hle,active,visible,WithBot.realAdd]
  · simp [hn,visible,WithBot.realAdd]

#axiomsClean key_value
#axiomsClean value_value
#axiomsClean score_value
end
end VeriTile.Bench.Repairs.FlashAttn.KernelReal
