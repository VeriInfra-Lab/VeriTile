import bench.tritonbench_g.pow_scalar_tensor.Kernels
import VeriTile.Triton.Float.ProfiledSoundness
import VeriTile.Triton.Semantics.TileOps
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.PowScalarTensor.NumericExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
open scoped Classical
set_option maxHeartbeats 8000000
set_option maxRecDepth 12000
set_option linter.unusedSimpArgs false

/-- Real mathematical power, not a constant-result backend. Identity floating
casts make this a numerical nonvacuity witness, not native IEEE evidence. -/
def model : Algebra (WithBot ℝ) where
  literal := fun _ _ r => some r
  negInf := none
  binary := fun _ _ op a b => match op with
    | .add => WithBot.realAdd a b
    | .sub => WithBot.realSub a b
    | .mul => WithBot.realMul a b
    | .div => WithBot.realDiv a b
    | .max => max a b
    | .pow => WithBot.realPow a b
  libdevicePowi := fun _ => some WithBot.realPowi
  unary := fun _ _ _ => none
  cast := fun _ _ _ x => x
  fromNat := fun _ n => some n
  fromInt := fun _ n => some n
  fp32Bits := fun _ => none
  fp32Load := id
  reduceMax := fun _ {_} ax keep v =>
    ((Tile.reduceMax ax keep ⟨v⟩).getD ⟨fun _ => none⟩).data
  reduceSum := fun _ {_} ax keep v => (Tile.reduceSum ax keep ⟨v⟩).data

/-- A legal nonzero program id, grid width two, independent input/output
strides and tagged sentinels. Exponents at the visited cells are 2, 3 and 1. -/
def initial : State (WithBot ℝ) where
  mem := fun r o => if r = "X" then
      .mk .real (some (if o = 4 then 2 else if o = 6 then 3 else 1))
    else .mk .int 37
  regs := fun _ _ _ => none
  pids := fun ax => if ax = 0 then 1 else 0
  numPids := fun ax => if ax = 0 then 2 else 1
  undef := fun d _ _ => defaultValue d

/-- One-tile mode reaches a partial tile. Grid-stride mode runs two actual
iterations, skips another program's tile and reaches a partial final tile.
The arbitrary unused task count is intentionally different from the shape. -/
def source (output : TypedSource.FloatingOutput) (oneTile candidate : Bool) : ComputeKernel :=
  TypedSource.kernel output oneTile candidate 2 "X" "Y" 2 3
    (if oneTile then 3 else 7) 918 2 2

private theorem range_two {α : Type} [Inhabited α] (M : Algebra α)
    (body : List ComputeStmt) (s : State α) :
    FP.Structural.range M "j" 0 2 1 body s =
      (run M body (s.setReg "j" .nat [] (fun _ => 0))).bind (fun t =>
        run M body (t.setReg "j" .nat [] (fun _ => 1))) := by
  simp [FP.Structural.range]

/-- Successful execution of every complete floating destination/source variant
under the numerical backend, including both actual grid-stride iterations. -/
def numeric_run (output : TypedSource.FloatingOutput) (oneTile candidate : Bool) :
    {t : State (WithBot ℝ) // run (engine model)
      (source output oneTile candidate).surfaceBody initial = some t} := by
  cases output <;> cases oneTile <;> cases candidate
  all_goals
    refine ⟨?_, ?_⟩
    swap
    simp (maxSteps := 500000) [source, TypedSource.kernel, TypedSource.fp32Kernel,
      TypedSource.fp64Kernel, TypedSource.fp16Kernel, TypedSource.bf16Kernel,
      TypedSource.f8e4Kernel, TypedSource.f8e5Kernel,
      ComputeKernel.surfaceBody, ComputeKernel.inputs, ComputeKernel.outputs,
      run, step, range_two, FP.Structural.evalExpr, evalComputeOp, evalOp_unfold,
      evalOffsets, evalDeltas, Option.bind, engine, Algebra.withDefaultPrecision,
      resolvePrecision, model, ComputeDType.eraseDType, numeric, numericLt,
      numericEq, bop, natLt, initial, Cell.read, ofFloat, toFloat,
      TileShape.axisDim, Region.cast, FP.Structural.store, Tile.select,
      Tile.ptrAdd, TileShape.allIndices, List.finRange_succ]
    try rfl

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.PowScalarTensor.NumericExecution
