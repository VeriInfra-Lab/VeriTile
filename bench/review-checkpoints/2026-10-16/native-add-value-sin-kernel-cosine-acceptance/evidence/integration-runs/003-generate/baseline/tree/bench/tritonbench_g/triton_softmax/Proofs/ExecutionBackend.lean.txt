import bench.tritonbench_g.triton_softmax.Proofs.Precision
import VeriTile.Triton.Float.ProfiledSoundness
import VeriTile.Triton.Semantics.TileOps

namespace VeriTile.Bench.TritonBenchG.TritonSoftmax.NumericExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
open scoped Classical
set_option maxHeartbeats 8000000
set_option maxRecDepth 16000

/-- Exact-real operations supply an inhabited abstract typed backend. This is
an execution witness, not a model of IEEE rounding or native target admission. -/
def model : Algebra (WithBot ℝ) where
  literal := fun _ _ r => some r
  negInf := none
  binary := fun _ _ op a b => match op with
    | .add => WithBot.realAdd a b
    | .sub => WithBot.realSub a b
    | .mul => WithBot.realMul a b
    | .div => WithBot.realDiv a b
    | .max => max a b
    | .pow => some 0
  unary := fun _ op a => match op with
    | .exp => WithBot.realExp a
    | _ => some 0
  cast := fun _ _ _ x => x
  fromNat := fun _ n => some n
  fromInt := fun _ n => some n
  fp32Bits := fun _ => some 0
  fp32Load := id
  reduceMax := fun _ {_} ax keep v =>
    ((Tile.reduceMax ax keep ⟨v⟩).getD ⟨fun _ => none⟩).data
  reduceSum := fun _ {_} ax keep v => (Tile.reduceSum ax keep ⟨v⟩).data

/-- Program one reads two zeros in a padded four-lane row. Output cells start
at seven; unequal input/output strides make the actual memory wiring visible. -/
def initial (input output : Storage) : State (WithBot ℝ) where
  mem := fun r _ => if r = "X" then
    .mk (Precision.channel input).toTileDType (ofFloat (Precision.channel input) (some 0))
    else .mk (Precision.channel output).toTileDType
      (ofFloat (Precision.channel output) (some 7))
  regs := fun _ _ _ => none
  pids := fun ax => if ax = 0 then 1 else 0
  numPids := fun _ => 2
  undef := fun d _ _ => defaultValue d

/-- All 36 floating carrier pairs and both complete variants execute from
empty registers with nonzero program id, independent strides and partial tails. -/
def source_run (input output : Storage) (v : Bool) :
    {t : State (WithBot ℝ) // run (engine model)
      (typedKernel input output v "Y" "X" 5 7 2 4).surfaceBody
        (initial input output) = some t} := by
  cases input <;> cases output <;> cases v
  all_goals
    refine ⟨?_, ?_⟩
    swap
    simp (maxSteps := 500000) [typedKernel, ComputeKernel.surfaceBody,
      run, step, FP.Structural.evalExpr, evalComputeOp, evalOp_unfold,
      Option.bind, engine, Algebra.withDefaultPrecision, resolvePrecision, model,
      ComputeDType.eraseDType, numeric, bop, natLt, initial, Precision.channel, Cell.read,
      ofFloat, toFloat, TileShape.axisDim, Region.cast, State.setReg]
    try rfl


end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.TritonSoftmax.NumericExecution
