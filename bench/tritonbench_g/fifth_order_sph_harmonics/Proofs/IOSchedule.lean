import bench.tritonbench_g.fifth_order_sph_harmonics.IO

namespace VeriTile.Bench.Repairs.FifthOrderSphHarmonics.IOSchedule
open VeriTile.Triton FP.Structural HarmonicIO
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 12000

theorem order_run {α : Type} [Inhabited α] (M : Algebra α) (d : Grid.Direction)
    (p : Parameters) (order : List Nat) (s : State α) :
    runSchedule (order.map fun pid => {
      kernel := match d with | .forward => forwardKernel p | .backward => backwardKernel p
      backend := M
      pids := fun _ => pid
      setPids := fun ax => ax == 0 }) s = Grid.execute M d p order s := by
  induction order generalizing s with
  | nil => rfl
  | cons pid rest ih =>
    simp only [List.map_cons, runSchedule, Grid.execute, Invocation.run]
    cases d <;>
      simp only [Invocation.initial, bindRegisters, List.foldl_nil, Bool.false_eq_true,
        ↓reduceIte, Option.getD_none, beq_iff_eq, Grid.source, Grid.atProgram] <;>
      congr 1 <;> funext t <;> exact ih t

theorem launch_run {α : Type} [Inhabited α] (M : Algebra α) (d : Grid.Direction)
    (p : Parameters) (s : State α) :
    runSchedule (launches M d p) s = Grid.launch M d p s :=
  order_run M d p (List.range (Grid.blocks p)) s

#axiomsClean launch_run
end
end VeriTile.Bench.Repairs.FifthOrderSphHarmonics.IOSchedule
