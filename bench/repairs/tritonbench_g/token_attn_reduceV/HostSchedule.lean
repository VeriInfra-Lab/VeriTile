import bench.repairs.tritonbench_g.token_attn_reduceV.Host
import bench.repairs.tritonbench_g.token_attn_reduceV.Schedule
namespace VeriTile.Bench.Repairs.TokenAttnReduceV.Host
open VeriTile.Triton FP.Structural Grid
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 25000
set_option linter.unusedSimpArgs false

def dimensions (c : Call) (a : Nat) := if a=0 then batches c else if a=1 then heads c else 1

/-- The source has no num_programs operation. Its actual successful execution
therefore has the same memory for the supplied physical grid dimensions. -/
theorem launch_dimensions {α : Type} [Inhabited α] (M : Algebra α) (R : SumPrimitive α)
    (p : Parameters) (m : Memory α) (pid : PID) (dims : Nat→Nat) :
    (FP.Structural.exec (withSum M R) (kernel p) {state m pid with numPids:=dims}).map State.mem=
      launch M R p pid m := by
  obtain ⟨t,run,mem⟩ := complete_run M R p {state m pid with numPids:=dims}
  rw [run,Option.map_some,mem,launch_complete]
  congr 1
  simp only [reference,fold_write_memory]
  rfl

def runActual {α : Type} [Inhabited α] (M : Algebra α) (R : SumPrimitive α)
    (c : Call) : List PID→Memory α→Option (Memory α)
  | [],m => some m
  | pid::rest,m =>
    ((FP.Structural.exec (withSum M R) (kernel (params c))
      {state m pid with numPids:=dimensions c}).map State.mem).bind (runActual M R c rest)

theorem runActual_eq {α : Type} [Inhabited α] (M : Algebra α) (R : SumPrimitive α)
    (c : Call) (work : List PID) (m : Memory α) :
    runActual M R c work m=Grid.run M R (params c) work m := by
  induction work generalizing m with
  | nil => rfl
  | cons pid rest ih => simp only [runActual,launch_dimensions,Grid.run]; congr 1; funext; apply ih

def invokeActual {α : Type} [Inhabited α] (M : Algebra α) (R : SumPrimitive α)
    (c : Call) (work : List PID) (m : Memory α) : Option (Result α) := do
  let final ← runActual M R c work m
  pure ⟨none,c.output,final⟩

/-- Arbitrary scheduling of the complete wrapper grid, with its actual
num_programs dimensions and fresh per-CTA registers. -/
theorem invokeActual_eq {α : Type} [Inhabited α] (M : Algebra α) (R : SumPrimitive α)
    (c : Call) (work : List PID) (m : Memory α) (layout : Layout c)
    (legal : (fullGrid (batches c) (heads c)).Perm work) :
    invokeActual M R c work m=invoke M R c m := by
  simp only [invokeActual,invoke,runActual_eq,Grid.complete]
  rw [← Grid.schedules_equal M R (params c) _ _ m legal layout.separated (unique c layout)]

#axiomsClean launch_dimensions
#axiomsClean invokeActual_eq
end
end VeriTile.Bench.Repairs.TokenAttnReduceV.Host
