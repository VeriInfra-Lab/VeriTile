import bench.repairs.tritonbench_g.context_attn_fwd.Proofs.NativeIOComplete

namespace VeriTile.Bench.Repairs.ContextAttnFwd.NativeTransport
open VeriTile.Triton FP.Structural NativeFormula NativeArrayBridge
noncomputable section
variable {α : Type} [Inhabited α]

def launch {g : Grid} (idx : GridIndex g) : Launch := ⟨idx.toPids,some g.toNumPids⟩

theorem run_grid (p : Layout) (r : Regions) (scale : ℝ) (M : Algebra α)
    (g : Grid) (jobs : List (GridIndex g)) (s : State α) :
    runSchedule ((jobs.map launch).map (invocation p r scale M)) s=KernelRawGrid.run p r scale M g jobs s := by
  induction jobs generalizing s with
  | nil => rfl
  | cons idx rest ih =>
    simp only [List.map_cons,runSchedule,KernelRawGrid.run]
    change (exec (M.withDefaultPrecision .fp32) (kernel p r .fp32 scale) (FPSuccess.atIndex s idx)).bind _ = _
    cases he : exec (M.withDefaultPrecision .fp32) (kernel p r .fp32 scale) (FPSuccess.atIndex s idx) with
    | none => rfl
    | some t => exact ih t

/-- Exact original raw-grid signature obtained through shared IO execution.
The reference/frame identification reuses the established source theorem;
`run_grid` proves equality of the actual schedules, including arbitrary aliases. -/
theorem complete (p : Layout) (r : Regions) (scale : ℝ) (M : Algebra α)
    (dots : FPSuccess.DotSupport p M) (positive : 0 < p.BN)
    (g : Grid) (jobs : List (GridIndex g)) (s : State α) :
    ∃ final,KernelRawGrid.run p r scale M g jobs s=some final ∧
      final.mem=(KernelRawGrid.reference p r scale M dots g jobs s).mem ∧
      (∀ rg off,KernelRawGrid.Outside p r scale M dots g jobs s rg off → final.mem rg off=s.mem rg off) := by
  let io := NativeIO.io p r M dots scale (jobs.map launch)
  obtain ⟨_,final,ran,_,_⟩ := NativeIOComplete.complete p r M dots scale (jobs.map launch) positive
    (io.snapshot s) trivial s (io.snapshot_binding rfl s)
  have actual : KernelRawGrid.run p r scale M g jobs s=some final := by
    rw [←run_grid]
    exact ran
  obtain ⟨old,oldRan,memory,frame⟩ := KernelRawGrid.complete p r scale M dots positive g jobs s
  have same : final=old := Option.some.inj (actual.symm.trans oldRan)
  subst old
  exact ⟨final,actual,memory,frame⟩

/-- The unrestricted singleton keeps its caller's exact launch metadata.
No active-lane injectivity or capacity assumption is needed for execution. -/
theorem single (p : Layout) (r : Regions) (scale : ℝ) (M : Algebra α)
    (dots : FPSuccess.DotSupport p M) (positive : 0 < p.BN) (s : State α) :
    ∃ final,exec (M.withDefaultPrecision .fp32) (kernel p r .fp32 scale) s=some final ∧
      final.mem=(FPSuccess.scatter p r s s (FPReference.output p M dots scale (FPReference.inputs p r s))).mem := by
  let job : Launch := ⟨s.pids,none⟩
  let io := NativeIO.io p r M dots scale [job]
  obtain ⟨_,final,ran,_,_⟩ := NativeIOComplete.complete p r M dots scale [job] positive
    (io.snapshot s) trivial s (io.snapshot_binding rfl s)
  have actual : exec (M.withDefaultPrecision .fp32) (kernel p r .fp32 scale) s=some final := by
    change runSchedule [invocation p r scale M job] s=some final at ran
    rw [runSchedule_singleton] at ran
    exact ran
  obtain ⟨old,oldRan,memory⟩ := FPReference.program p r scale M dots positive s
  have same : final=old := Option.some.inj (actual.symm.trans oldRan)
  subst old
  exact ⟨final,actual,memory⟩

#axiomsClean complete
#axiomsClean single
end
end VeriTile.Bench.Repairs.ContextAttnFwd.NativeTransport
