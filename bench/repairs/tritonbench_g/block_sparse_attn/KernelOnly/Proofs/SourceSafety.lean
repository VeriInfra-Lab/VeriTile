import bench.repairs.tritonbench_g.block_sparse_attn.KernelOnly.Proofs.AccessFinish

namespace VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly.SourceSafety
open VeriTile Triton FP.Structural
open ChunkRetention.Whole001
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxRecDepth 24000
set_option maxHeartbeats 1000000

theorem channel_loop_safe (a : Arithmetic α) (native : Bool) (p : Parameters)
    (initial current : State (Replicas α)) (carry : Carry α) (slot : Int)
    (h : ChannelInvariant a p initial carry (csrColumn p initial slot)
      (scoreAccumulation a native p initial (csrColumn p initial slot) 0) current)
    (bounds : RegionName→Nat) (backed : AccessBounds p initial bounds)
    (lo : first p initial≤slot) (hi : slot<last p initial) :
    Safety.loopSafe a.model bounds "score_block" 0 p.blocks (channelBody native p) current := by
  apply Safety.loopSafe_of_invariant a.model bounds "score_block" p.blocks (channelBody native p)
    (fun n s=>ChannelInvariant a p initial carry (csrColumn p initial slot)
      (scoreAccumulation a native p initial (csrColumn p initial slot) n) s) _ _ 0 current h
  · intro n hn s hs
    exact channel_safe a native p initial s carry slot _ n hs bounds backed lo hi hn
  · intro n hn s hs t ran
    let next := channelExecution a native p initial s carry (csrColumn p initial slot)
      (scoreAccumulation a native p initial (csrColumn p initial slot) n) n hs
    have result := channelInvariant a native p initial s carry (csrColumn p initial slot)
      (scoreAccumulation a native p initial (csrColumn p initial slot) n) n hs
    rw [next.property] at ran
    cases ran
    simpa only [scoreAccumulation_succ] using result

theorem visit_safe (a : Arithmetic α) (native : Bool) (p : Parameters)
    (initial current : State (Replicas α)) (carry : Carry α) (slot : Int)
    (h : Runtime a p initial carry current) (ints : ColumnInt32 p initial slot)
    (bounds : RegionName→Nat) (backed : AccessBounds p initial bounds)
    (lo : first p initial≤slot) (hi : slot<last p initial) :
    Safety.traceSafeList a.model bounds (loopBody native p)
      (current.setReg "col_idx_idx" .int [] (fun _=>slot)) := by
  let start := visitPrefixExecution_local a p initial current carry slot h ints
  have startInv := visitPrefixInvariant_local a p initial current carry slot h ints
  have safeLoop := channel_loop_safe a native p initial start.val carry slot startInv bounds backed lo hi
  obtain ⟨mid,hm,hmi⟩ := channelLoopExecution a native p initial start.val carry
    (csrColumn p initial slot) 0 (by omega) startInv
  have loopRun : run a.model [.forLoop "score_block" p.blocks (channelBody native p)] start.val=some mid := by
    simp [run,step,hm]
  rw [loopBody_partition,Safety.traceSafeList_append,Safety.traceSafeList_append]
  refine ⟨⟨visit_prefix_safe a p initial current carry slot h ints bounds backed lo hi,?_⟩,?_⟩
  · intro next ran
    rw [start.property] at ran
    cases ran
    simpa only [AccessSafety.singleton_safe,Safety.stepSafe_unfold] using safeLoop
  · intro next ran
    rw [run_append,start.property] at ran
    simp only [Option.bind_some,loopRun] at ran
    cases ran
    exact suffix_safe a native p initial mid carry slot hmi bounds backed lo hi

theorem outer_loop_safe (a : Arithmetic α) (native : Bool) (p : Parameters)
    (initial current : State (Replicas α)) (carry : Carry α)
    (h : Runtime a p initial carry current) (ints : ReadInt32 p initial)
    (bounds : RegionName→Nat) (backed : AccessBounds p initial bounds) :
    Safety.rangeIntSafe a.model bounds "col_idx_idx" (first p initial) (last p initial) 1
      (loopBody native p) current := by
  let I := fun slot s=>first p initial≤slot ∧ ∃ c,Runtime a p initial c s
  have localInts (slot : Int) (lo : first p initial≤slot) (hi : slot<last p initial) :
      ColumnInt32 p initial slot := by
    have live : (slot-first p initial).toNat<visitCount p initial := by
      unfold visitCount
      exact (Int.toNat_lt_toNat (lt_of_le_of_lt (sub_nonneg.mpr lo) (sub_lt_sub_right hi _))).mpr
        (sub_lt_sub_right hi _)
    have result := ints.columns _ live
    have equal : first p initial+((slot-first p initial).toNat : Int)=slot := by
      rw [Int.toNat_of_nonneg (sub_nonneg.mpr lo)]
      rw [← Int.add_sub_assoc,Int.add_comm (first p initial) slot,Int.add_sub_cancel]
    simpa only [equal] using result
  apply Safety.rangeIntSafe_of_invariant a.model bounds "col_idx_idx" (last p initial) 1
    (loopBody native p) I _ _ (first p initial) current ⟨le_refl _,carry,h⟩
  · intro slot hi s hs
    obtain ⟨c,hc⟩ := hs.2
    exact visit_safe a native p initial s c slot hc (localInts slot hs.1 hi) bounds backed hs.1 hi
  · intro slot hi s hs t ran
    obtain ⟨c,hc⟩ := hs.2
    obtain ⟨next,known,hnext⟩ := visitExecution_local a native p initial s c slot hc (localInts slot hs.1 hi)
    rw [known] at ran
    cases ran
    exact ⟨le_trans hs.1 (show slot≤slot+(1 : Nat) from le_add_of_nonneg_right (by decide)),_,hnext⟩

/-- All actual source loads/stores in both signed CSR and channel loops,
including the selected native expression-site lowering. -/
theorem source_safe (a : Arithmetic α) (native : Bool) (p : Parameters)
    (initial : State (Replicas α)) (ints : ReadInt32 p initial)
    (bounds : RegionName→Nat) (backed : AccessBounds p initial bounds) :
    Safety.traceSafeList a.model bounds
      (if native then selectedSource p else (source p).surfaceBody)
      (bindScale (broadcast (a.argument p.scale)) initial) := by
  let start := prefixExecution_local a.model p (bindScale (broadcast (a.argument p.scale)) initial) ints.rows
  have startInv := initialRuntime_local a p initial ints.rows
  have safeLoop := outer_loop_safe a native p initial start.val (initialCarry a) startInv ints bounds backed
  obtain ⟨mid,hm,hmi⟩ := outerLoopExecution_local a native p initial start.val 0 (by omega) startInv ints.columns
  have loopRun : run a.model [.forRangeInt "col_idx_idx" (.ref .int [] "start_l")
      (.ref .int [] "end_l") (.constNat 1) (loopBody native p)] start.val=some mid := by
    simpa [run,step,evalOp_unfold,startInv.first,startInv.last] using hm
  rw [execution_partition,Safety.traceSafeList_append,Safety.traceSafeList_append]
  refine ⟨⟨prefix_safe a p initial bounds ints.rows backed,?_⟩,?_⟩
  · intro next ran
    rw [start.property] at ran
    cases ran
    simpa [AccessSafety.singleton_safe,Safety.stepSafe_unfold,Safety.safeOp_unfold,
      evalOp_unfold,startInv.first,startInv.last] using safeLoop
  · intro next ran
    rw [run_append,start.property] at ran
    simp only [Option.bind_some,loopRun] at ran
    cases ran
    exact epilogue_safe a p initial mid _ hmi bounds backed

#axiomsClean source_safe
end
end VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly.SourceSafety
