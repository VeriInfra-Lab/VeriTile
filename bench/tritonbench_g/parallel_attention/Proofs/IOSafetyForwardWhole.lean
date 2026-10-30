import bench.tritonbench_g.parallel_attention.Proofs.IOSafetyForwardLoops

namespace VeriTile.Bench.Repairs.ParallelAttentionKernel.IOSafety
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 800000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

private theorem trace_singleton (M : Algebra α) (bounds : RegionName → Nat)
    (st : ComputeStmt) (s : State α) :
    Safety.traceSafeList M bounds [st] s ↔ Safety.stepSafe M bounds st s := by
  rw [Safety.traceSafeList_cons]
  simp

private theorem pure_list (M : Algebra α) (bounds : RegionName → Nat)
    (code : List ComputeStmt) (safe : ∀ st∈code,∀ s,Safety.stepSafe M bounds st s)
    (s : State α) : Safety.traceSafeList M bounds code s := by
  induction code generalizing s with
  | nil => exact Safety.traceSafeList_nil _ _ _
  | cons st rest ih =>
    apply Safety.traceSafeList_cons_intro (safe st (by simp) s)
    intro t _
    exact ih (fun st h=>safe st (by simp [h])) t

private theorem suffix_head (M : Algebra α) (D : DotPrimitive α) (c : ForwardConfig)
    (bounds : RegionName → Nat) (s : State α) :
    Safety.traceSafeList (withDot M D) bounds ((forwardSuffix c).take 2) s := by
  apply pure_list
  intro st hs t
  simp [forwardSuffix,forwardKernel,forwardSource,ComputeKernel.surfaceBody] at hs
  rcases hs with rfl|rfl
  all_goals simp only [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold]
  all_goals try simp only [safeBlockPtr2]
  all_goals simp [Safety.safeOp_unfold]

theorem forward_suffix (M : Algebra α) (D : DotPrimitive α) (c : ForwardConfig)
    (bounds : RegionName → Nat) (s t : State α) (last : Nat)
    (acc : ForwardAccumulator α c.BTL c.BV) (h : ForwardReady M c s t last acc)
    (hb : ForwardBounds c s bounds) :
    Safety.traceSafeList (withDot M D) bounds (forwardSuffix c) t := by
  rw [←List.take_append_drop 2 (forwardSuffix c),Safety.traceSafeList_append]
  refine ⟨suffix_head M D c bounds t,?_⟩
  intro u ran
  simp [forwardSuffix,forwardKernel,forwardSource,ComputeKernel.surfaceBody,run,step,
    evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,Option.bind,ComputeDType.eraseDType,
    numeric,bop,Region.cast,withDot,h.head,h.chunk,h.keyBlock,h.valueBlock] at ran
  subst u
  by_cases hv : s.pids 0 % ((c.V+c.BV-1)/c.BV) = 0
  all_goals simp [hv,numericEq,forwardSuffix,forwardKernel,forwardSource,ComputeKernel.surfaceBody,
    Safety.traceSafeList_unfold,trace_singleton,Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,
    Safety.safeOp_unfold,Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe,Safety.maskActive,
    step,evalExpr,evalComputeOp,evalOp_unfold,boundaryZero,Option.bind,ComputeDType.eraseDType,store,
    numeric,bop,natLt,numericLt,Region.cast,withDot,State.setReg,
    h.head,h.chunk,h.keyBlock,h.valueBlock,h.output,h.normalizer,
    TileShape.indexToList,BlockPtr.address_2d_offsets,BlockPtr.inBounds_2d_offsets,
    ofFloat,toFloat,queryChunk,valueChunk,Nat.add_assoc]
  · refine ⟨?_,?_,?_⟩
    · simpa [TileBounds,queryChunk,valueChunk,hv,Nat.add_assoc] using hb.output
    · simpa [queryChunk,Nat.add_assoc] using hb.normalizer hv
    · split <;> trivial
  · simpa [TileBounds,queryChunk,valueChunk,Nat.add_assoc] using hb.output

theorem forward_source (M : Algebra α) (D : DotPrimitive α) (c : ForwardConfig)
    (bounds : RegionName → Nat) (s : State α) (hb : ForwardBounds c s bounds) :
    TraceSafe (withDot M D) bounds (forwardKernel c) s := by
  obtain ⟨initial,hp,hi⟩:=first_prefix_run M D c s
  obtain ⟨first,hr,last,hf⟩:=first_range_run M D c s initial 0 (queryChunk c s) _ hi
  obtain ⟨diagonal,hm,hd⟩:=between_loops_run M D c s first last _ hf
  obtain ⟨second,hs,finish,hz⟩:=second_range_run M D c s diagonal (queryChunk c s)
    ((s.pids 1+1)*c.BTL) _ hd
  change Safety.traceSafeList _ _ _ _
  rw [forward_source_split]
  simp only [List.append_assoc]
  rw [Safety.traceSafeList_append]
  refine ⟨forward_prefix M D c bounds s hb,?_⟩
  intro t ht
  rw [hp] at ht
  cases Option.some.inj ht
  simp only [List.singleton_append]
  apply Safety.traceSafeList_cons_of_step
  · have safe:=forward_first_range M D c bounds s initial 0 (queryChunk c s) _ hi hb
    simpa [firstRangeStatement,Safety.stepSafe_unfold,Safety.safeOp_unfold,
      evalOp_unfold,numeric,bop,Option.bind,hi.chunk,queryChunk] using safe
  · rw [first_range_statement M D c s initial 0 _ hi]
    exact hr
  · rw [Safety.traceSafeList_append]
    refine ⟨forward_between M D c bounds first,?_⟩
    intro t ht
    rw [hm] at ht
    cases Option.some.inj ht
    apply Safety.traceSafeList_cons_of_step
    · have safe:=forward_second_range M D c bounds s diagonal (queryChunk c s)
        ((s.pids 1+1)*c.BTL) _ hd hb
      simpa [secondRangeStatement,Safety.stepSafe_unfold,Safety.safeOp_unfold,
        evalOp_unfold,numeric,bop,Option.bind,hd.chunk,queryChunk] using safe
    · rw [second_range_statement M D c s diagonal (queryChunk c s) _ hd.toForwardReady]
      exact hs
    · exact forward_suffix M D c bounds s second finish _ hz hb

#axiomsClean forward_suffix
#axiomsClean forward_source
end
end VeriTile.Bench.Repairs.ParallelAttentionKernel.IOSafety
