import bench.tritonbench_g.pow_scalar_tensor.Proofs.IntegerExecutionValues

namespace VeriTile.Bench.TritonBenchG.PowScalarTensor.IntegerExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
open scoped Classical
set_option maxHeartbeats 8000000
set_option maxRecDepth 12000
set_option linter.unusedSimpArgs false

/-- Bad source input remains an integer cell; no wrapping or float conversion
is used to manufacture an in-range exponent. -/
def invalidInitial (n : Int) : State (WithBot ℝ) where
  mem := fun r _ => if r = "X" then .mk .int n else .mk .int 37
  regs := fun _ _ _ => none
  pids := fun ax => if ax = 0 then 1 else 0
  numPids := fun ax => if ax = 0 then 2 else 1
  undef := fun d _ _ => defaultValue d

private theorem range_two {α : Type} [Inhabited α] (M : Algebra α)
    (body : List ComputeStmt) (s : State α) :
    FP.Structural.range M "j" 0 2 1 body s =
      (run M body (s.setReg "j" .nat [] (fun _ => 0))).bind (fun t =>
        run M body (t.setReg "j" .nat [] (fun _ => 1))) := by
  simp [FP.Structural.range]

/-- Either complete source branch fails on its actual first integer-power
operation when an active input lies outside the signed32 domain. -/
theorem range_failure (output : TypedSource.FloatingOutput) (oneTile candidate : Bool)
    (n : Int) (hn : Int32Bits.inRange n = Bool.false) :
    run (engine model) (source output oneTile candidate).surfaceBody (invalidInitial n) = none := by
  cases output <;> cases oneTile <;> cases candidate
  all_goals
    simp (maxSteps := 500000) [source, TypedIntegerSource.kernel, TypedIntegerSource.fp32Kernel,
      TypedIntegerSource.fp64Kernel, TypedIntegerSource.fp16Kernel, TypedIntegerSource.bf16Kernel,
      TypedIntegerSource.f8e4Kernel, TypedIntegerSource.f8e5Kernel,
      ComputeKernel.surfaceBody, ComputeKernel.inputs, ComputeKernel.outputs,
      run, step, range_two, FP.Structural.evalExpr, evalComputeOp, evalOp_unfold,
      evalOffsets, evalDeltas, Option.bind, engine, Algebra.withDefaultPrecision,
      resolvePrecision, model, NumericExecution.model, Int32Bits.valuesValid, hn, ComputeDType.eraseDType, numeric, numericLt,
      numericEq, bop, natLt, invalidInitial, State.setReg, Cell.read, defaultValue, ofFloat, toFloat,
      TileShape.axisDim, Region.cast, FP.Structural.store, Tile.select,
      Tile.ptrAdd, TileShape.allIndices, List.finRange_succ,
      TileShape.indexToList, BlockPtr.inBounds_1d, BlockPtr.address_1d, show Int32Bits.inRange 0 = Bool.true by decide]

/-- Both immediately out-of-range endpoints fail rather than wrap. -/
theorem range_endpoints (output : TypedSource.FloatingOutput) (oneTile candidate : Bool) :
    run (engine model) (source output oneTile candidate).surfaceBody
      (invalidInitial 2147483648) = none ∧
    run (engine model) (source output oneTile candidate).surfaceBody
      (invalidInitial (-2147483649)) = none :=
  ⟨range_failure output oneTile candidate 2147483648 rfl,
    range_failure output oneTile candidate (-2147483649) rfl⟩

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.PowScalarTensor.IntegerExecution
