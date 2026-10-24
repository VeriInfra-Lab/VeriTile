import bench.tritonbench_g.fifth_order_sph_harmonics.Proofs.TypedReference
namespace VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.TypedFP
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 20000000
set_option maxRecDepth 48000
set_option linter.unusedSimpArgs false

private theorem fold_regs {α ι : Type} (xs : List ι)
    (active : ι → Prop) [DecidablePred active] (r : RegionName)
    (off : ι → Nat) (v : ι → Cell α) (s : State α) :
    (xs.foldl (fun t i => if active i then t.write r (off i) (v i) else t) s).regs = s.regs := by
  induction xs generalizing s with
  | nil => rfl
  | cons i xs ih => simp only [List.foldl_cons, ih]; split <;> rfl

private theorem fold_pids {α ι : Type} (xs : List ι)
    (active : ι → Prop) [DecidablePred active] (r : RegionName)
    (off : ι → Nat) (v : ι → Cell α) (s : State α) :
    (xs.foldl (fun t i => if active i then t.write r (off i) (v i) else t) s).pids = s.pids := by
  induction xs generalizing s with
  | nil => rfl
  | cons i xs ih => simp only [List.foldl_cons, ih]; split <;> rfl

/-- The actual full source terminates, including every one of its eleven
masked channel stores. Values are not supplied as final-state assumptions. -/
def execution {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) :
    {t : State α // FP.Structural.exec M (kernel p) s = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp (config := {maxSteps := 4000000}) [kernel, source, FP.Structural.exec,
    ComputeKernel.surfaceBody, run, step, evalExpr, evalComputeOp, evalOp_unfold,
    State.setReg, store, ofFloat, toFloat, Region.cast, ComputeDType.eraseDType,
    numeric, numericLt, natLt, bop, Option.bind, fold_regs, fold_pids]
  try rfl

#axiomsClean execution
end
end VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.TypedFP
