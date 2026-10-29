import bench.repairs.tritonbench_g.mixed_sparse_attention.Proofs.IOSourceSafety

namespace VeriTile.Bench.Repairs.MixedSparseAttention.IOSourceSafety
open VeriTile Triton FP.Structural
open Whole002 Source
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 35000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem range_safe (M : Algebra α) (D : Dot α) (c : Config) (s t : State α)
    (ph : Bool) (first last stride : Nat) (a : Accumulator α c.BM c.BD)
    (h : Ready M c s t a) (hbn : 0<c.BN) (capacity : RegionName → Nat)
    (hb : Bounds c s capacity) :
    Safety.rangeSafe (backend M D) capacity (indexName ph) first last stride (loopBody c ph) t := by
  apply Safety.rangeSafe_of_invariant (backend M D) capacity (indexName ph) last stride
    (loopBody c ph) (fun _ u=>∃ b,Ready M c s u b)
  · intro n _ u hu
    obtain ⟨b,hu⟩:=hu
    exact body_safe M D c s u ph n b hu hbn capacity hb
  · intro n _ u hu v ran
    obtain ⟨b,hu⟩:=hu
    obtain ⟨next,he,hn⟩:=loop_body_run M D c s u ph n b hu hbn
    have eq:=Option.some.inj (ran.symm.trans he)
    subst v
    exact ⟨_,hn⟩
  · exact ⟨a,h⟩

theorem block_safe (M : Algebra α) (D : Dot α) (c : Config) (s t : State α)
    (a : Accumulator α c.BM c.BD) (h : Ready M c s t a) (hbn : 0<c.BN)
    (capacity : RegionName → Nat) (hb : Bounds c s capacity) :
    Safety.stepSafe (backend M D) capacity (blockStatement c) t := by
  simpa [blockStatement,Safety.stepSafe_unfold,Safety.safeOp_unfold,evalOp_unfold,
    h.maxblocks,Option.bind,loopBody,indexName] using
    range_safe M D c s t Bool.false 0 (blockCount c s) 1 a h hbn capacity hb

theorem column_safe (M : Algebra α) (D : Dot α) (c : Config) (s t : State α)
    (a : Accumulator α c.BM c.BD) (h : Ready M c s t a)
    (hc : t.regs .nat [] "max_num_cols"=some (fun _=>columnCount c s)) (hbn : 0<c.BN)
    (capacity : RegionName → Nat) (hb : Bounds c s capacity) :
    Safety.stepSafe (backend M D) capacity (columnStatement c) t := by
  simpa [columnStatement,Safety.stepSafe_unfold,Safety.safeOp_unfold,evalOp_unfold,
    hc,Option.bind,loopBody,indexName] using
    range_safe M D c s t Bool.true 0 (columnCount c s) c.BN a h hbn capacity hb

theorem active_body_safe (M : Algebra α) (D : Dot α) (c : Config) (s : State α)
    (hbn : 0<c.BN) (capacity : RegionName → Nat) (hb : Bounds c s capacity) :
    Safety.traceSafeList (backend M D) capacity
      ((parsedKernel c).surfaceBody.take 3 ++ activeBody c) s := by
  obtain ⟨a,ha,har⟩:=prefix_run M D c s
  obtain ⟨b,hbRun,hbr⟩:=block_statement_run M D c s a _ har hbn
  obtain ⟨d,hd,hdr,hdc⟩:=between_run M D c s b _ hbr
  obtain ⟨e,he,her⟩:=column_statement_run M D c s d _ hdr hdc hbn
  rw [source_split]
  simp only [List.append_assoc,Safety.traceSafeList_append]
  refine ⟨prelude_safe M D c s capacity hb,?_⟩
  intro u hu
  have eu:=Option.some.inj (hu.symm.trans ha)
  subst u
  refine ⟨Safety.traceSafeList_cons_intro (block_safe M D c s a _ har hbn capacity hb)
    (fun _ _=>Safety.traceSafeList_nil _ _ _),?_⟩
  intro u hu
  have eu : b=u := by simpa only [run,hbRun,bind,pure,Option.bind,Option.some.injEq] using hu
  subst u
  refine ⟨between_safe M D c b capacity,?_⟩
  intro u hu
  have eu:=Option.some.inj (hu.symm.trans hd)
  subst u
  refine ⟨Safety.traceSafeList_cons_intro (column_safe M D c s d _ hdr hdc hbn capacity hb)
    (fun _ _=>Safety.traceSafeList_nil _ _ _),?_⟩
  intro u hu
  have eu : e=u := by simpa only [run,he,bind,pure,Option.bind,Option.some.injEq] using hu
  subst u
  exact suffix_safe M D c s e _ her capacity hb

#axiomsClean active_body_safe
end
end VeriTile.Bench.Repairs.MixedSparseAttention.IOSourceSafety
