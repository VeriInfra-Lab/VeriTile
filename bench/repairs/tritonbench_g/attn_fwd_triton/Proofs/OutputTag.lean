import bench.repairs.tritonbench_g.attn_fwd_triton.Proofs.NativeGraphHost

namespace VeriTile.Bench.Repairs.AttnFwdTriton.Checked
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open TypedExecution NativeGraph
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

private theorem scatter_tag {α ι : Type} (r : RegionName) (off : ι→Nat) (value : ι→α)
    (active : ι→Prop) [DecidablePred active] (xs : List ι) (s : State α) (target : Nat)
    (h : ∃ i∈xs,active i ∧ off i=target) :
    ∃ v, (xs.foldl (fun t i=>if active i then t.write r (off i) (.mk .bf16 (value i)) else t) s).mem r target=
      Cell.mk .bf16 v := by
  induction xs generalizing s with
  | nil=>simpa using h
  | cons i xs ih=>
    by_cases ht : ∃ j∈xs,active j ∧ off j=target
    · exact ih _ ht
    · obtain ⟨j,hj,ha,he⟩:=h
      have hij : j=i := by
        rcases List.mem_cons.mp hj with heq|hm
        · exact heq
        · exact False.elim (ht ⟨j,hm,ha,he⟩)
      subst j
      simp only [List.foldl_cons,if_pos ha]
      rw [State.masked_scatter_frame r off (fun j=>Cell.mk .bf16 (value j)) active xs r target
        (Or.inr (by intro j hj ha he; exact ht ⟨j,hj,ha,he.symm⟩))]
      subst target
      exact ⟨value i,State.write_same ..⟩

/-- Every active store of the complete tail has the actual BF16 cell tag. -/
theorem tail_bf16 {α : Type} [Inhabited α] (A : Algebra α)
    (p : Config) (BD pm ph : Nat) (s t : State α) (hs : Ready BD s)
    (ha : AddressReady p BD pm ph s) (ht : run (engine A) (tail p BD Bool.false) s=some t)
    (i : TileIndex [128,BD]) (hi : outputActive p BD pm i=Bool.true) :
    ∃ v, t.mem p.Out (outputAddress p BD pm ph i)=Cell.mk .bf16 v := by
  rcases hs with ⟨_,_,_,_,_,_,_,_,_,_,_,_,_,⟨li,hli⟩,⟨acc,hacc⟩⟩
  rcases ha with ⟨hn,ho,hp⟩
  simp only [tail,source,Bool.false_eq_true,↓reduceIte,TypedSource.repaired,
    ComputeKernel.surfaceBody,List.drop_succ_cons,List.drop_zero] at ht
  simp [run,step,store,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,
    Option.bind,ComputeDType.eraseDType,numeric,natLt,numericLt,
    Cell.read,ofFloat,toFloat,Region.cast,State.setReg,hn,ho,hp,hli,hacc] at ht
  subst t
  apply scatter_tag
  refine ⟨i,TileShape.mem_allIndices _ i,?_,rfl⟩
  simpa [outputActive,bop] using hi

/-- The selected graph reference is a BF16 output cell, derived from the
executed tail and exact active source store; no dtype/output premise is used. -/
theorem reference_bf16 (ops : Operations) (p : Config) (BD : Nat)
    (a : Verified.PhysicalCells.Layout)
    (initial : VeriTile.Bench.Repairs.LayerNormFwd.TypedPhysical.Physical Word a.toLayout)
    (template : State Word) (hD : p.headDim≤BD) (c : Coordinate p) :
    ∃ w, NativeGraph.reference ops p BD a initial template c=Cell.mk .bf16 w := by
  let s:=VeriTile.Bench.Repairs.LayerNormFwd.TypedPhysical.pull a.toLayout
    (liftPhysical initial) (liftState template)
  let sp:=atProgram p s (owner p c)
  obtain ⟨u,hu,hr,ha,_⟩:=before_tail backend p BD (support BD) Bool.false sp
  obtain ⟨t,ht⟩:=tail_success backend p BD Bool.false u hr
  have hex : exec (engine backend) (source p BD Bool.false) sp=some t := by
    change run (engine backend) (source p BD Bool.false).surfaceBody sp=some t
    rw [←List.take_append_drop 41 (source p BD Bool.false).surfaceBody,run_append,hu]
    exact ht
  have he : referenceState backend p BD (support BD) s (owner p c)=t :=
    Option.some.inj ((reference_exec backend p BD (support BD) s (owner p c)).symm.trans hex)
  have htag:=tail_bf16 backend p BD (owner p c).1.val (owner p c).2.val u t hr ha ht
    (lane p BD hD c) (lane_active p BD hD c)
  rw [lane_address] at htag
  obtain ⟨v,hv⟩:=htag
  refine ⟨evaluate ops v,?_⟩
  change evaluateCell ops ((referenceState backend p BD (support BD) s (owner p c)).mem p.Out
    (logicalAddress p c))=_
  rw [he,hv]
  rfl

#axiomsClean tail_bf16
#axiomsClean reference_bf16
end VeriTile.Bench.Repairs.AttnFwdTriton.Checked
