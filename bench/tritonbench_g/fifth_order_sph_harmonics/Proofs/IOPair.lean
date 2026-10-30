import bench.tritonbench_g.fifth_order_sph_harmonics.PairIO
import bench.tritonbench_g.fifth_order_sph_harmonics.Proofs.SourcePair
import bench.tritonbench_g.fifth_order_sph_harmonics.Proofs.IOCorrect

namespace VeriTile.Bench.Repairs.FifthOrderSphHarmonics.IOPair
open VeriTile Triton FP.Structural HarmonicIO KernelScalar
noncomputable section

private theorem map_pair {ι : Type} (left right : ι → Invocation Packing.Word)
    (same : ∀ i s, (left i).run s = (right i).run s)
    (safe : ∀ i s bounds, (left i).Safe bounds s ↔ (right i).Safe bounds s)
    (ids : List ι) (s : State Packing.Word) :
    runSchedule (ids.map left) s = runSchedule (ids.map right) s ∧
    ∀ bounds, ScheduleSafe bounds (ids.map left) s ↔ ScheduleSafe bounds (ids.map right) s := by
  induction ids generalizing s with
  | nil => exact ⟨rfl,fun _ => Iff.rfl⟩
  | cons i ids ih =>
    constructor
    · simp only [List.map_cons,runSchedule,same]
      congr 1
      funext t
      exact (ih t).1
    · intro bounds
      simp only [List.map_cons,ScheduleSafe,safe,same]
      exact and_congr Iff.rfl (forall_congr' fun t => imp_congr_right fun _ => (ih t).2 bounds)

theorem schedules (d : Grid.Direction) (p : Parameters) (values : (io d p).Inputs)
    (s : State Packing.Word) :
    runSchedule ((io d p).schedule values) s = runSchedule (schedule d p) s ∧
    ∀ bounds, ScheduleSafe bounds ((io d p).schedule values) s ↔ ScheduleSafe bounds (schedule d p) s := by
  have pair := map_pair (job (backend d) d p) (candidateJob (backend d) d p)
    (fun pid s => SourcePair.source_exec (backend d) d p _)
    (fun pid s bounds => SourcePair.source_safe (backend d) d p _ bounds)
    (List.range (Grid.blocks p)) s
  cases d <;> exact pair

theorem implemented (d : Grid.Direction) (p : Parameters) (fit : p.columnOffset + 11 ≤ p.outputWidth) :
    (io d p).Implements (target d p) := by
  cases d
  · exact IOCorrect.native_forward p fit
  · exact IOCorrect.native_backward p fit

/-- Both complete grids succeed on the same final tagged memory. The original
IO proof provides the full active output and exterior frame certificate. -/
theorem equivalent (d : Grid.Direction) (p : Parameters) (fit : p.columnOffset + 11 ≤ p.outputWidth) :
    (io d p).Equivalent (variant d p) := by
  intro values _ bounds hb s hin
  obtain ⟨safe,final,ran,_,frame⟩ := implemented d p fit values trivial bounds hb s hin
  have same := schedules d p values s
  refine ⟨safe,(same.2 bounds).mp safe,final,final,ran,?_,?_,frame,frame⟩
  · exact same.1.symm.trans ran
  · intro o j active
    rfl
end
end VeriTile.Bench.Repairs.FifthOrderSphHarmonics.IOPair
