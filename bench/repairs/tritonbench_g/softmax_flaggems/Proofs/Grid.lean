import bench.repairs.tritonbench_g.softmax_flaggems.Proofs.SourceBridge

namespace VeriTile.Bench.Repairs.SoftmaxFlaggems.Grid
open VeriTile Triton
open VeriTile.Bench.TritonBenchG.SoftmaxFlaggems
open HostGrid
set_option maxHeartbeats 4000000
set_option maxRecDepth 20000

theorem inner_flatten (P G Y : RegionName) (M N C B : Nat) (one : Bool) :
    (FullSource.softmax_backward_kernel_inner .fp32 P G Y M N C B one).eraseDType.toAlgKernel.FlattenOk := by
  cases one
  · exact RealBackwardInner.original_flattenOk .fp32 P G Y M N B C
  · exact RealBackwardInner.one_original_flattenOk .fp32 P G Y M N B C

theorem noninner_flatten (P G Y : RegionName) (M N K B C : Nat) (one : Bool) :
    (FullSource.softmax_backward_kernel_non_inner .fp32 P G Y M N K B C one).eraseDType.toAlgKernel.FlattenOk := by
  cases one
  · exact RealBackwardNonInner.original_flattenOk .fp32 P G Y M N K B C
  · exact RealBackwardNonInner.one_original_flattenOk .fp32 P G Y M N K B C

theorem run_undef (kernel : Kernel) (hok : kernel.FlattenOk)
    (gx gy : Nat) (ps : List Program) (s t : BlockState)
    (h : run kernel gx gy ps s = some t) : t.undef = s.undef := by
  induction ps generalizing s with
  | nil =>
    have he : s = t := Option.some.inj h
    simpa [he]
  | cons p ps ih =>
    cases he : exec kernel (setup gx gy p s) with
    | none => simp [run, he] at h
    | some mid =>
      have hu := stepStmts_undef kernel.body (setup gx gy p s) mid hok he
      have ht : run kernel gx gy ps mid = some t := by simpa [run, he] using h
      exact (ih mid ht).trans hu

theorem run_equiv (old repaired : Kernel) (hok : old.FlattenOk)
    (hb : ZeroPadding.Equivalent old.body repaired.body)
    (gx gy : Nat) (ps : List Program) (s : BlockState) (hs : ZeroPadding.Zero s) :
    run old gx gy ps s = run repaired gx gy ps s := by
  induction ps generalizing s with
  | nil => rfl
  | cons p ps ih =>
    have hb' : exec old (setup gx gy p s) = exec repaired (setup gx gy p s) :=
      hb (setup gx gy p s) hs
    simp only [run, ← hb']
    cases he : exec old (setup gx gy p s) with
    | none => rfl
    | some mid =>
      exact ih mid ((stepStmts_undef old.body (setup gx gy p s) mid hok he).trans hs)

/-- Execute the actual repaired quotation at every source-scheduled CTA. -/
noncomputable def backwardRun (P G Y : RegionName)
    (d : Dimensions) (c : LaunchConfig) (s : BlockState) : Option BlockState :=
  if 1 < d.K then
    run (Source.softmax_backward_kernel_non_inner_fp32 P G Y d.M d.N d.K c.B c.C c.one).eraseDType.toAlgKernel
      d.M (RealAddress.rounds d.K c.C) (grid d.M (RealAddress.rounds d.K c.C)) s
  else
    run (Source.softmax_backward_kernel_inner_fp32 P G Y d.M d.N c.C c.B c.one).eraseDType.toAlgKernel
      (RealAddress.rounds d.M c.C) 1 (grid (RealAddress.rounds d.M c.C) 1) s

theorem backward_eq (P G Y : RegionName) (d : Dimensions) (c : LaunchConfig)
    (s : BlockState) (hs : ZeroPadding.Zero s) :
    HostGrid.backwardRun .fp32 P G Y d c s = backwardRun P G Y d c s := by
  unfold HostGrid.backwardRun backwardRun
  split
  · exact run_equiv _ _ (noninner_flatten P G Y d.M d.N d.K c.B c.C c.one)
      (SourceBridge.noninner_body P G Y d.M d.N d.K c.B c.C c.one) _ _ _ s hs
  · exact run_equiv _ _ (inner_flatten P G Y d.M d.N c.C c.B c.one)
      (SourceBridge.inner_body P G Y d.M d.N c.C c.B c.one) _ _ _ s hs

theorem old_backward_undef (P G Y : RegionName) (d : Dimensions) (c : LaunchConfig)
    (s t : BlockState) (he : HostGrid.backwardRun .fp32 P G Y d c s = some t) :
    t.undef = s.undef := by
  unfold HostGrid.backwardRun at he
  split at he
  · exact run_undef _ (noninner_flatten P G Y d.M d.N d.K c.B c.C c.one) _ _ _ s t he
  · exact run_undef _ (inner_flatten P G Y d.M d.N c.C c.B c.one) _ _ _ s t he

noncomputable def trialRun (P G Y : RegionName) (d : Dimensions) :
    List (Fin 6) → BlockState → Option BlockState
  | [], s => some s
  | choice :: rest, s => do
    let mid ← backwardRun P G Y d (backwardConfig d choice) s
    trialRun P G Y d rest mid

theorem old_trial_undef (P G Y : RegionName) (d : Dimensions)
    (trials : List (Fin 6)) (s t : BlockState)
    (he : HostGrid.trialRun .fp32 P G Y d trials s = some t) :
    t.undef = s.undef := by
  induction trials generalizing s with
  | nil =>
    have h : s = t := Option.some.inj he
    simpa [h]
  | cons c cs ih =>
    cases hb : HostGrid.backwardRun .fp32 P G Y d (backwardConfig d c) s with
    | none => simp [HostGrid.trialRun, hb] at he
    | some mid =>
      have ht : HostGrid.trialRun .fp32 P G Y d cs mid = some t := by
        simpa [HostGrid.trialRun, hb] using he
      exact (ih mid ht).trans (old_backward_undef P G Y d _ s mid hb)

theorem trial_eq (P G Y : RegionName) (d : Dimensions)
    (trials : List (Fin 6)) (s : BlockState) (hs : ZeroPadding.Zero s) :
    HostGrid.trialRun .fp32 P G Y d trials s = trialRun P G Y d trials s := by
  induction trials generalizing s with
  | nil => rfl
  | cons c cs ih =>
    simp only [HostGrid.trialRun, trialRun, ← backward_eq P G Y d _ s hs]
    cases hb : HostGrid.backwardRun .fp32 P G Y d (backwardConfig d c) s with
    | none => rfl
    | some mid => exact ih mid ((old_backward_undef P G Y d _ s mid hb).trans hs)

noncomputable def tunedBackward (P G Y : RegionName) (d : Dimensions)
    (trials : List (Fin 6)) (choice : Fin 6) (s : BlockState) : Option BlockState := do
  let warmed ← trialRun P G Y d trials s
  backwardRun P G Y d (backwardConfig d choice) warmed

theorem tuned_eq (P G Y : RegionName) (d : Dimensions)
    (trials : List (Fin 6)) (choice : Fin 6) (s : BlockState) (hs : ZeroPadding.Zero s) :
    HostGrid.tunedBackward .fp32 P G Y d trials choice s =
      tunedBackward P G Y d trials choice s := by
  simp only [HostGrid.tunedBackward, tunedBackward, ← trial_eq P G Y d trials s hs]
  cases ht : HostGrid.trialRun .fp32 P G Y d trials s with
  | none => rfl
  | some mid => exact backward_eq P G Y d _ mid ((old_trial_undef P G Y d trials s mid ht).trans hs)

#axiomsClean backward_eq
#axiomsClean trial_eq
#axiomsClean tuned_eq
end VeriTile.Bench.Repairs.SoftmaxFlaggems.Grid
