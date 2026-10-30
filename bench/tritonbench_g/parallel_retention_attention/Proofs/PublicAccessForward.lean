import bench.tritonbench_g.parallel_retention_attention.Proofs.PublicAccessForwardParts

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel.PublicAccess
open VeriTile.Triton FP.Structural FPStages
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages
  (DotPrimitive withDot ForwardConfig queryChunk keyChunk valueChunk)
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 10000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem forward_first_range_safe (M : Algebra α) (D : DotPrimitive α)
    (bounds : RegionName → Nat) (c : ForwardConfig) (s t : State α)
    (start stop : Nat) (acc : TileIndex [c.BTL,c.BV] → α)
    (h : ForwardReady M c s t start acc) (hb : ForwardBounds bounds c s) :
    Safety.rangeSafe (withDot M D) bounds "_i" start stop c.BTS
      (firstBody c.BTL c.BTS c.BK c.BV) t := by
  apply Safety.rangeSafe_of_invariant (withDot M D) bounds "_i" stop c.BTS _
    (fun n u => ∃ a, ForwardReady M c s u n a) _ _ start t ⟨acc,h⟩
  · intro n hn u hu
    obtain ⟨a,ha⟩ := hu
    exact forward_first_body_safe M D bounds c s _ n a
      (forwardReady_set_index M c s u n a ha n) hb
  · intro n hn u hu v hv
    obtain ⟨a,ha⟩ := hu
    obtain ⟨v',run,ready⟩ := first_body_run M D c s _ n a
      (forwardReady_set_index M c s u n a ha n)
    rw [hv] at run
    cases run
    exact ⟨_,ready⟩

theorem forward_second_range_safe (M : Algebra α) (D : DotPrimitive α)
    (bounds : RegionName → Nat) (c : ForwardConfig) (s t : State α)
    (start stop : Nat) (acc : TileIndex [c.BTL,c.BV] → α)
    (h : DiagonalReady M c s t start acc) (hb : ForwardBounds bounds c s) :
    Safety.rangeSafe (withDot M D) bounds "_i" start stop c.BTS
      (secondBody c.BTL c.BTS c.BK c.BV) t := by
  apply Safety.rangeSafe_of_invariant (withDot M D) bounds "_i" stop c.BTS _
    (fun n u => ∃ a, DiagonalReady M c s u n a) _ _ start t ⟨acc,h⟩
  · intro n hn u hu
    obtain ⟨a,ha⟩ := hu
    exact forward_second_body_safe M D bounds c s _ n a
      (diagonalReady_set_index M c s u n a ha n) hb
  · intro n hn u hu v hv
    obtain ⟨a,ha⟩ := hu
    obtain ⟨v',run,ready⟩ := second_body_run M D c s _ n a
      (diagonalReady_set_index M c s u n a ha n)
    rw [hv] at run
    cases run
    exact ⟨_,ready⟩

/-- Safety follows both actual loop executions and their live pointer invariants. -/
theorem forward_safe (M : Algebra α) (D : DotPrimitive α)
    (bounds : RegionName → Nat) (c : ForwardConfig) (s : State α)
    (hb : ForwardBounds bounds c s) :
    TraceSafe (withDot M D) bounds (forwardKernel c) s := by
  obtain ⟨initial,hp,hi⟩ := first_prefix_run M D c s
  obtain ⟨first,hr,last,hf⟩ := first_range_run M D c s initial 0 (queryChunk c s) _ hi
  obtain ⟨diagonal,hm,hd⟩ := between_loops_run M D c s first last _ hf
  obtain ⟨second,hs,finish,hz⟩ := second_range_run M D c s diagonal (queryChunk c s)
    ((s.pids 1+1)*c.BTL) _ hd
  unfold TraceSafe
  rw [forward_source_split]
  simp only [List.append_assoc,List.cons_append,List.nil_append]
  rw [Safety.traceSafeList_append]
  refine ⟨forward_prefix_safe M D bounds c s hb,?_⟩
  intro u hu
  rw [hp] at hu
  cases hu
  apply Safety.traceSafeList_cons_of_step (t := first)
  · simpa [firstRangeStatement,Safety.stepSafe_unfold,Safety.safeOp_unfold,
      evalOp_unfold,hi.chunk,numeric,bop,Option.bind,queryChunk] using
      forward_first_range_safe M D bounds c s initial 0 (queryChunk c s) _ hi hb
  · simpa only [first_range_statement M D c s initial 0 _ hi] using hr
  · rw [Safety.traceSafeList_append]
    refine ⟨forward_middle_safe M D bounds c s first last _ hf,?_⟩
    intro u hu
    rw [hm] at hu
    cases hu
    apply Safety.traceSafeList_cons_of_step (t := second)
    · simpa [secondRangeStatement,Safety.stepSafe_unfold,Safety.safeOp_unfold,
        evalOp_unfold,hd.chunk,numeric,bop,Option.bind,queryChunk] using
        forward_second_range_safe M D bounds c s diagonal (queryChunk c s)
          ((s.pids 1+1)*c.BTL) _ hd hb
    · simpa only [second_range_statement M D c s diagonal (queryChunk c s) _ hd.toForwardReady] using hs
    · exact forward_suffix_safe M D bounds c s second finish _ hz hb

#axiomsClean forward_safe
end
end VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel.PublicAccess
