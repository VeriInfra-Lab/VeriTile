import bench.tritonbench_g.parallel_retention_attention.Proofs.PublicAccessQueryParts

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel.PublicQueryAccess
open VeriTile.Triton FP.Structural FPQuery
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages
  (DotPrimitive withDot BackwardConfig)
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 10000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem first_range_safe (M : Algebra α) (D : DotPrimitive α)
    (bounds : RegionName → Nat) (c : BackwardConfig) (dq : RegionName) (s t : State α)
    (start stop : Nat) (acc : TileIndex [c.BTL,c.BK] → α)
    (h : Ready M c s t start acc) (hb : Bounds bounds c dq) :
    Safety.rangeSafe (withDot M D) bounds "_i" start stop c.BTS
      (firstBody c) t := by
  apply Safety.rangeSafe_of_invariant (withDot M D) bounds "_i" stop c.BTS _
    (fun n u => ∃ a, Ready M c s u n a) _ _ start t ⟨acc,h⟩
  · intro n hn u hu
    obtain ⟨a,ha⟩ := hu
    exact first_body_safe M D bounds c dq s _ n a
      (ready_set_index M c s u n a ha n) hb
  · intro n hn u hu v hv
    obtain ⟨a,ha⟩ := hu
    obtain ⟨v',run,ready⟩ := first_body_run M D c s _ n a
      (ready_set_index M c s u n a ha n)
    rw [hv] at run
    cases run
    exact ⟨_,ready⟩

theorem second_range_safe (M : Algebra α) (D : DotPrimitive α)
    (bounds : RegionName → Nat) (c : BackwardConfig) (dq : RegionName) (s t : State α)
    (start stop : Nat) (acc : TileIndex [c.BTL,c.BK] → α)
    (h : DiagonalReady M c s t start acc) (hb : Bounds bounds c dq) :
    Safety.rangeSafe (withDot M D) bounds "_i" start stop c.BTS
      (secondBody c) t := by
  apply Safety.rangeSafe_of_invariant (withDot M D) bounds "_i" stop c.BTS _
    (fun n u => ∃ a, DiagonalReady M c s u n a) _ _ start t ⟨acc,h⟩
  · intro n hn u hu
    obtain ⟨a,ha⟩ := hu
    exact second_body_safe M D bounds c dq s _ n a
      (diagonal_ready_set_index M c s u n a ha n) hb
  · intro n hn u hu v hv
    obtain ⟨a,ha⟩ := hu
    obtain ⟨v',run,ready⟩ := second_body_run M D c s _ n a
      (diagonal_ready_set_index M c s u n a ha n)
    rw [hv] at run
    cases run
    exact ⟨_,ready⟩

/-- Safety follows both actual loop executions and their live pointer invariants. -/
theorem query_safe (M : Algebra α) (D : DotPrimitive α)
    (bounds : RegionName → Nat) (c : BackwardConfig) (dq : RegionName) (s : State α)
    (hb : Bounds bounds c dq) :
    TraceSafe (withDot M D) bounds (source c dq) s := by
  obtain ⟨initial,hp,hi⟩ := prefix_run M D c dq s
  obtain ⟨first,hr,last,hf⟩ := first_range_run M D c s initial 0 (c.i_c*c.BTL) _ hi
  obtain ⟨diagonal,hm,hd⟩ := middle_run M D c dq s first last _ hf
  obtain ⟨second,hs,finish,hz⟩ := second_range_run M D c s diagonal (c.i_c*c.BTL)
    ((c.i_c+1)*c.BTL) _ hd
  unfold TraceSafe
  rw [source_split]
  simp only [List.append_assoc,List.cons_append,List.nil_append]
  rw [Safety.traceSafeList_append]
  refine ⟨prefix_safe M D bounds c dq s hb,?_⟩
  intro u hu
  rw [hp] at hu
  cases hu
  apply Safety.traceSafeList_cons_of_step (t := first)
  · simpa [firstRange,Safety.stepSafe_unfold,Safety.safeOp_unfold,
      evalOp_unfold,numeric,bop,Option.bind] using
      first_range_safe M D bounds c dq s initial 0 (c.i_c*c.BTL) _ hi hb
  · simpa only [first_range_statement M D c initial] using hr
  · rw [Safety.traceSafeList_append]
    refine ⟨middle_safe M D bounds c dq s first last _ hf,?_⟩
    intro u hu
    rw [hm] at hu
    cases hu
    apply Safety.traceSafeList_cons_of_step (t := second)
    · simpa [secondRange,Safety.stepSafe_unfold,Safety.safeOp_unfold,
        evalOp_unfold,numeric,bop,Option.bind] using
        second_range_safe M D bounds c dq s diagonal (c.i_c*c.BTL)
          ((c.i_c+1)*c.BTL) _ hd hb
    · simpa only [second_range_statement M D c diagonal] using hs
    · exact suffix_safe M D bounds c dq s second finish _ hz hb

#axiomsClean query_safe
end
end VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel.PublicQueryAccess
