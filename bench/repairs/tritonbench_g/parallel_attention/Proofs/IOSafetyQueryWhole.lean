import bench.repairs.tritonbench_g.parallel_attention.Proofs.IOSafetyQuery

namespace VeriTile.Bench.Repairs.ParallelAttentionKernel.IOSafety
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 800000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem query_first_range (M : Algebra α) (D : DotPrimitive α) (c : QueryConfig)
    (bounds : RegionName → Nat) (s t : State α) (start stop : Nat)
    (acc : TileIndex [c.BTL,c.BK] → α) (h : QueryReady M c s t start acc)
    (hb : QueryBounds c bounds) :
    Safety.rangeSafe (withDot M D) bounds "_i" start stop c.BTS (queryFirstBody c) t := by
  apply Safety.rangeSafe_of_invariant (withDot M D) bounds "_i" stop c.BTS _
    (fun q u=>∃ a,QueryReady M c s u q a)
  · rintro q _ u ⟨a,hu⟩
    exact query_first_body M D c bounds s _ q a (queryReady_index M c s u q a hu q) hb
  · rintro q _ u ⟨a,hu⟩ v ran
    obtain ⟨w,hw,hready⟩:=query_first_body_run M D c s _ q a (queryReady_index M c s u q a hu q)
    rw [hw] at ran
    cases Option.some.inj ran
    exact ⟨_,hready⟩
  · exact ⟨acc,h⟩

theorem query_second_range (M : Algebra α) (D : DotPrimitive α) (c : QueryConfig)
    (bounds : RegionName → Nat) (s t : State α) (start stop : Nat)
    (acc : TileIndex [c.BTL,c.BK] → α) (h : QueryDiagonalReady M c s t start acc)
    (hb : QueryBounds c bounds) :
    Safety.rangeSafe (withDot M D) bounds "_i" start stop c.BTS (querySecondBody c) t := by
  apply Safety.rangeSafe_of_invariant (withDot M D) bounds "_i" stop c.BTS _
    (fun q u=>∃ a,QueryDiagonalReady M c s u q a)
  · rintro q _ u ⟨a,hu⟩
    exact query_second_body M D c bounds s _ q a (queryDiagonalReady_index M c s u q a hu q) hb
  · rintro q _ u ⟨a,hu⟩ v ran
    obtain ⟨w,hw,hready⟩:=query_second_body_run M D c s _ q a (queryDiagonalReady_index M c s u q a hu q)
    rw [hw] at ran
    cases Option.some.inj ran
    exact ⟨_,hready⟩
  · exact ⟨acc,h⟩

theorem query_source (M : Algebra α) (D : DotPrimitive α) (c : QueryConfig)
    (bounds : RegionName → Nat) (s : State α) (hb : QueryBounds c bounds) :
    TraceSafe (withDot M D) bounds (queryKernel c) s := by
  obtain ⟨initial,hp,hi⟩:=query_prefix_run M D c s
  obtain ⟨first,hr,last,hf⟩:=query_first_range_run M D c s initial 0 (c.i_c*c.BTL) _ hi
  obtain ⟨diagonal,hm,hd⟩:=query_middle_run M D c s first last _ hf
  obtain ⟨second,hs,finish,hz⟩:=query_second_range_run M D c s diagonal
    (c.i_c*c.BTL) ((c.i_c+1)*c.BTL) _ hd
  change Safety.traceSafeList _ _ _ _
  rw [query_source_split]
  simp only [List.append_assoc]
  rw [Safety.traceSafeList_append]
  refine ⟨query_prefix M D c bounds s hb,?_⟩
  intro t ht
  rw [hp] at ht
  cases Option.some.inj ht
  simp only [List.singleton_append]
  apply Safety.traceSafeList_cons_of_step
  · have safe:=query_first_range M D c bounds s initial 0 (c.i_c*c.BTL) _ hi hb
    simpa [queryFirstRange,queryFirstBody,queryKernel,querySource,ComputeKernel.surfaceBody,
      Safety.stepSafe_unfold,Safety.safeOp_unfold,evalOp_unfold,numeric,bop,Option.bind] using safe
  · rw [query_first_range_statement]
    exact hr
  · rw [Safety.traceSafeList_append]
    refine ⟨query_middle M D c bounds first,?_⟩
    intro t ht
    rw [hm] at ht
    cases Option.some.inj ht
    apply Safety.traceSafeList_cons_of_step
    · have safe:=query_second_range M D c bounds s diagonal
        (c.i_c*c.BTL) ((c.i_c+1)*c.BTL) _ hd hb
      simpa [querySecondRange,querySecondBody,queryKernel,querySource,ComputeKernel.surfaceBody,
        Safety.stepSafe_unfold,Safety.safeOp_unfold,evalOp_unfold,numeric,bop,Option.bind] using safe
    · rw [query_second_range_statement]
      exact hs
    · exact query_suffix M D c bounds s second finish _ hz hb

#axiomsClean query_source
end
end VeriTile.Bench.Repairs.ParallelAttentionKernel.IOSafety
