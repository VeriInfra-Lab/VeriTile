import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.NativeMachineHistory
import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.NativeOracleReads

namespace VeriTile.Bench.Repairs.LayerNormFwd.NativeMachine
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open NativeRow Address PhysicalDomain NativeABI
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 12000

def dispatch {α : Type} [Inhabited α] (c : Config α block)
    (benchmarks : List (DispatchEvent block)) (selected : Schedule block) (physical : Memory c) : DispatchResult c :=
  let before := history c benchmarks physical
  match before.error with
  | some _ => before
  | none => ⟨executeGrid (c.withSchedule selected) before.memory,none,before.completed++[selected]⟩

theorem dispatch_failure {α : Type} [Inhabited α] (c : Config α block)
    (benchmarks : List (DispatchEvent block)) (selected : Schedule block) (physical : Memory c)
    (code : Nat) (h : (history c benchmarks physical).error=some code) :
    dispatch c benchmarks selected physical=history c benchmarks physical := by
  simp [dispatch,h]

theorem dispatch_success {α : Type} [Inhabited α] (c : Config α block)
    (benchmarks : List (DispatchEvent block)) (selected : Schedule block) (physical : Memory c)
    (h : (history c benchmarks physical).error=none) :
    dispatch c benchmarks selected physical=
      ⟨executeGrid (c.withSchedule selected) (history c benchmarks physical).memory,none,
        (history c benchmarks physical).completed++[selected]⟩ := by
  simp [dispatch,h]

theorem dispatch_frame {α : Type} [Inhabited α] (c : Config α block)
    (benchmarks : List (DispatchEvent block)) (selected : Schedule block) (physical : Memory c)
    (i : Fin (layout c.allocations).capacity) (hi : ∀ row : Fin c.M, Outside c row i) :
    (dispatch c benchmarks selected physical).memory i=physical i := by
  cases he : (history c benchmarks physical).error with
  | some code => rw [dispatch_failure c benchmarks selected physical code he]; exact history_frame c benchmarks physical i hi
  | none =>
    rw [dispatch_success c benchmarks selected physical he]
    exact (grid_frame (c.withSchedule selected) (history c benchmarks physical).memory i hi).trans
      (history_frame c benchmarks physical i hi)

theorem dispatch_reads {α : Type} [Inhabited α] (c : Config α block)
    (hsep : Domain.ReadWriteSeparated c.flags c.regions)
    (hgeo : NativeGeometry.Geometry c.flags c.regions c.strides c.M c.N)
    (benchmarks : List (DispatchEvent block)) (selected : Schedule block) (physical : Memory c) (template : State α) :
    TypedSource.ReadsEqual c.flags c.regions (view c (dispatch c benchmarks selected physical).memory template)
      (view c physical template) := by
  cases he : (history c benchmarks physical).error with
  | some code =>
    rw [dispatch_failure c benchmarks selected physical code he]
    exact history_reads c hsep hgeo benchmarks physical template
  | none =>
    rw [dispatch_success c benchmarks selected physical he]
    intro region hr offset
    exact (grid_reads (c.withSchedule selected) hsep hgeo (history c benchmarks physical).memory template region hr offset).trans
      (history_reads c hsep hgeo benchmarks physical template region hr offset)

/-- Final selected output is evaluated against the original pre-benchmark
input snapshot. Every intervening launch runs on the actual predecessor state;
input preservation and target dependence are proved, not API premises. -/
theorem dispatch_return {α : Type} [Inhabited α] (c : Config α block)
    (hsep : Domain.ReadWriteSeparated c.flags c.regions)
    (hgeo : NativeGeometry.Geometry c.flags c.regions c.strides c.M c.N)
    (benchmarks : List (DispatchEvent block)) (selected : Schedule block) (physical : Memory c) (template : State α)
    (h : (history c benchmarks physical).error=none) :
    TypedSource.observe c.flags c.regions c.strides c.M c.N
      (view c (dispatch c benchmarks selected physical).memory template)=
      NativeSchedule.oracle c.ops selected.reductions selected.plan selected.plan1 c.flags c.regions c.strides
        c.M c.N c.eps c.p (view c physical template) := by
  rw [dispatch_success c benchmarks selected physical h]
  have hreturn := (grid_return (c.withSchedule selected) hsep hgeo (history c benchmarks physical).memory template).1
  change TypedSource.observe c.flags c.regions c.strides c.M c.N
      (view c (executeGrid (c.withSchedule selected) (history c benchmarks physical).memory) template)=_
  exact hreturn.trans (NativeSchedule.oracle_reads c.ops selected.reductions selected.plan selected.plan1
    c.flags c.regions c.strides c.M c.N c.eps c.p _ _ (history_reads c hsep hgeo benchmarks physical template))

#axiomsClean dispatch_failure
#axiomsClean dispatch_success
#axiomsClean dispatch_frame
#axiomsClean dispatch_reads
#axiomsClean dispatch_return
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.NativeMachine
