import bench.repairs.tritonbench_g.softmax_optimize.Proofs.FPExecution

namespace VeriTile.Bench.Repairs.SoftmaxOptimize.FPExecution
open VeriTile Triton FP.Structural
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem signed_range (A : Algebra α) (body : List ComputeStmt)
    (P : State α → Prop) (B : Nat) (hB : 0<B)
    (body_step : ∀ index s,P s → ∃ t,
      run A body (s.setReg "start_n" .int [] (fun _ => index))=some t ∧ P t)
    (cur stop : Int) (s : State α) (hs : P s) :
    ∃ t,rangeInt A "start_n" cur stop B body s=some t ∧ P t := by
  by_cases hc : cur<stop
  · obtain ⟨mid,he,hi⟩ := body_step cur s hs
    obtain ⟨t,ht,hI⟩ := signed_range A body P B hB body_step (cur+B) stop mid hi
    refine ⟨t,?_,hI⟩
    rw [rangeInt,if_neg (Nat.ne_of_gt hB),if_pos hc,he]
    exact ht
  · exact ⟨s,by rw [rangeInt,if_neg (Nat.ne_of_gt hB),if_neg hc],hs⟩
termination_by (stop-cur).toNat
decreasing_by omega

def loopStatements (Y X : RegionName) (N B : Nat) (input : Bool) : List ComputeStmt :=
  [.forRangeInt "start_n" (.constInt 0) (.ref .int [] "prev_multiple") (.constNat B)
      (if input then TypedSource.fullInput X N B else TypedSource.fullOutput Y X N B),
   .forRangeInt "start_n" (.ref .int [] "prev_multiple") (.ref .int [] "cols") (.constNat B)
      (if input then TypedSource.tailInput X N B else TypedSource.tailOutput Y X N B)]

theorem inputs (A : Algebra α) (eq : α → α → Bool)
    (Y X : RegionName) (N B pid : Nat) (hB : 0<B)
    (s : State α) (ready : Ready [B] N B pid s) :
    ∃ t,run (VeriTile.Triton.FP.ProfiledRewrite.engine (withEq A eq)) (loopStatements Y X N B Bool.true) s=some t ∧
      Ready [B] N B pid t ∧ t.mem=s.mem := by
  have body (tail : Bool) (index : Int) (t : State α)
      (ht : Ready [B] N B pid t ∧ t.mem=s.mem) :
      ∃ u,run (VeriTile.Triton.FP.ProfiledRewrite.engine (withEq A eq))
        (if tail then TypedSource.tailInput X N B else TypedSource.fullInput X N B)
        (t.setReg "start_n" .int [] (fun _ => index))=some u ∧
        Ready [B] N B pid u ∧ u.mem=s.mem := by
    obtain ⟨u,he,hr,hm⟩ := input_body A eq X N B pid tail t ht.1 index
    exact ⟨u,he,hr,hm.trans ht.2⟩
  obtain ⟨mid,hfull,hmid⟩ := signed_range (VeriTile.Triton.FP.ProfiledRewrite.engine (withEq A eq))
    (TypedSource.fullInput X N B) (fun t => Ready [B] N B pid t ∧ t.mem=s.mem)
    B hB (body Bool.false) 0 (previous N B) s ⟨ready,rfl⟩
  obtain ⟨t,htail,hr,hm⟩ := signed_range (VeriTile.Triton.FP.ProfiledRewrite.engine (withEq A eq))
    (TypedSource.tailInput X N B) (fun t => Ready [B] N B pid t ∧ t.mem=s.mem)
    B hB (body Bool.true) (previous N B) N mid hmid
  refine ⟨t,?_,hr,hm⟩
  simp only [loopStatements, Bool.false_eq_true,↓reduceIte,run,step,evalOp_unfold,bind,Option.bind,
    ready.2.2.1,hmid.1.2.2.1,hmid.1.2.1,Option.bind_some,hfull,htail]

theorem outputs (A : Algebra α) (eq : α → α → Bool)
    (Y X : RegionName) (N B pid : Nat) (hB : 0<B)
    (s : State α) (ready : Ready [] N B pid s) :
    ∃ t,run (VeriTile.Triton.FP.ProfiledRewrite.engine (withEq A eq)) (loopStatements Y X N B Bool.false) s=some t ∧
      Ready [] N B pid t ∧ Frame Y s t := by
  have body (tail : Bool) (index : Int) (t : State α)
      (ht : Ready [] N B pid t ∧ Frame Y s t) :
      ∃ u,run (VeriTile.Triton.FP.ProfiledRewrite.engine (withEq A eq))
        (if tail then TypedSource.tailOutput Y X N B else TypedSource.fullOutput Y X N B)
        (t.setReg "start_n" .int [] (fun _ => index))=some u ∧
        Ready [] N B pid u ∧ Frame Y s u := by
    obtain ⟨u,he,hr,hm⟩ := output_body A eq Y X N B pid tail t ht.1 index
    exact ⟨u,he,hr,fun r i h => (hm r i h).trans (ht.2 r i h)⟩
  obtain ⟨mid,hfull,hmid⟩ := signed_range (VeriTile.Triton.FP.ProfiledRewrite.engine (withEq A eq))
    (TypedSource.fullOutput Y X N B) (fun t => Ready [] N B pid t ∧ Frame Y s t)
    B hB (body Bool.false) 0 (previous N B) s ⟨ready,fun _ _ _ => rfl⟩
  obtain ⟨t,htail,hr,hm⟩ := signed_range (VeriTile.Triton.FP.ProfiledRewrite.engine (withEq A eq))
    (TypedSource.tailOutput Y X N B) (fun t => Ready [] N B pid t ∧ Frame Y s t)
    B hB (body Bool.true) (previous N B) N mid hmid
  refine ⟨t,?_,hr,hm⟩
  simp only [loopStatements, Bool.false_eq_true,↓reduceIte,run,step,evalOp_unfold,bind,Option.bind,
    ready.2.2.1,hmid.1.2.2.1,hmid.1.2.1,Option.bind_some,hfull,htail]

/-- All four actual loops, their comparisons, loads, reductions and typed
stores execute successfully for arbitrary total primitive operations. -/
theorem complete (A : Algebra α) (eq : α → α → Bool)
    (Y X : RegionName) (M N B : Nat) (hB : 0<B) (s : State α) :
    ∃ t, FP.Structural.exec (VeriTile.Triton.FP.ProfiledRewrite.engine (withEq A eq)) (kernel Y X M N B) s=some t ∧
      Frame Y s t := by
  obtain ⟨a,ha,hra,hma⟩ := prelude A eq Y X N B hB s
  obtain ⟨b,hb,hrb,hmb⟩ := inputs A eq Y X N B (s.pids 0) hB a hra
  obtain ⟨c,hc,hrc,hmc⟩ := reduction A eq Y X N B (s.pids 0) hB b hrb
  obtain ⟨t,ht,hrt,hmt⟩ := outputs A eq Y X N B (s.pids 0) hB c hrc
  refine ⟨t,?_,fun r i h => (hmt r i h).trans ?_⟩
  · change run _ (kernel Y X M N B).surfaceBody s=some t
    rw [TypedSource.source_body]
    change run _ (((TypedSource.body Y X N B).take 5 ++ loopStatements Y X N B Bool.true) ++
      ((TypedSource.body Y X N B).drop 7).take 4 ++ loopStatements Y X N B Bool.false) s=some t
    simp only [run_append,ha,hb,hc,ht,Option.bind_some]
  · rw [hmc,hmb,hma]

#axiomsClean complete
end
end VeriTile.Bench.Repairs.SoftmaxOptimize.FPExecution
