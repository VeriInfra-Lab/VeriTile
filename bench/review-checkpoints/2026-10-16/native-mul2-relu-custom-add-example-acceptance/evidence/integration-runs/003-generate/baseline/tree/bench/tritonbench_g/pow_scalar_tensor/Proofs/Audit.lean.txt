import bench.tritonbench_g.pow_scalar_tensor.Correct
import bench.tritonbench_g.pow_scalar_tensor.FPEquiv

#axiomsClean VeriTile.Bench.TritonBenchG.PowScalarTensor.pow_scalar_tensor_one_tile_io_correctness
#axiomsClean VeriTile.Bench.TritonBenchG.PowScalarTensor.floating_one_tile_correct
#axiomsClean VeriTile.Bench.TritonBenchG.PowScalarTensor.floating_grid_correct
#axiomsClean VeriTile.Bench.TritonBenchG.PowScalarTensor.integer_one_tile_correct
#axiomsClean VeriTile.Bench.TritonBenchG.PowScalarTensor.integer_grid_correct
#axiomsClean VeriTile.Bench.TritonBenchG.PowScalarTensor.FPEquiv.floating_source_equivalence
#axiomsClean VeriTile.Bench.TritonBenchG.PowScalarTensor.FPEquiv.integer_source_equivalence
#axiomsClean VeriTile.Bench.TritonBenchG.PowScalarTensor.IntegerSteps.perTile_steps
#axiomsClean VeriTile.Bench.TritonBenchG.PowScalarTensor.IntegerSteps.exponents_valid
#axiomsClean VeriTile.Bench.TritonBenchG.PowScalarTensor.IntegerSteps.load_int
#axiomsClean VeriTile.Bench.TritonBenchG.PowScalarTensor.IntegerGrid.range_run
#axiomsClean VeriTile.Bench.TritonBenchG.PowScalarTensor.IntegerFiniteWitness.one_tile
#axiomsClean VeriTile.Bench.TritonBenchG.PowScalarTensor.IntegerFiniteWitness.grid
#axiomsClean VeriTile.Bench.TritonBenchG.PowScalarTensor.IntegerFiniteWitness.grid_values
#axiomsClean VeriTile.Bench.TritonBenchG.PowScalarTensor.IntegerExecution.numeric_cells
#axiomsClean VeriTile.Bench.TritonBenchG.PowScalarTensor.IntegerExecution.range_failure
#axiomsClean VeriTile.Bench.TritonBenchG.PowScalarTensor.IntegerExecution.range_endpoints

#specNonCircular VeriTile.Bench.TritonBenchG.PowScalarTensor.IntegerSteps.powSpec avoiding [VeriTile.Bench.TritonBenchG.PowScalarTensor.TypedIntegerSource.kernel, VeriTile.Bench.TritonBenchG.PowScalarTensor.IntegerSource.kernel, VeriTile.Triton.exec, VeriTile.Triton.execR]
#specNonCircular VeriTile.Bench.TritonBenchG.PowScalarTensor.powSpec avoiding [VeriTile.Bench.TritonBenchG.PowScalarTensor.TypedSource.kernel, VeriTile.Bench.TritonBenchG.PowScalarTensor.TypedIntegerSource.kernel, VeriTile.Triton.exec, VeriTile.Triton.execR]
#specNonCircular VeriTile.Bench.TritonBenchG.PowScalarTensor.FiniteWitness.exponent avoiding [VeriTile.Bench.TritonBenchG.PowScalarTensor.TypedSource.kernel, VeriTile.Bench.TritonBenchG.PowScalarTensor.TypedIntegerSource.kernel, VeriTile.Triton.exec, VeriTile.Triton.execR]
#specNonCircular VeriTile.Bench.TritonBenchG.PowScalarTensor.IntegerFiniteWitness.exponent avoiding [VeriTile.Bench.TritonBenchG.PowScalarTensor.TypedSource.kernel, VeriTile.Bench.TritonBenchG.PowScalarTensor.TypedIntegerSource.kernel, VeriTile.Triton.exec, VeriTile.Triton.execR]
#print_fp_assumptions VeriTile.Bench.TritonBenchG.PowScalarTensor.FPEquiv.floating_source_equivalence
#print_fp_assumptions VeriTile.Bench.TritonBenchG.PowScalarTensor.FPEquiv.integer_source_equivalence

#guard_msgs (drop info) in
#auditModuleAxioms

/- Audit every theorem in the port, including private historical helpers. -/
run_cmd do
  let env ← Lean.getEnv
  let mut count := 0
  for (name, info) in env.constants.toList do
    let userName := ((Lean.privateToUserName? name).getD name).eraseMacroScopes
    if (`VeriTile.Bench.TritonBenchG.PowScalarTensor).isPrefixOf userName then
      match info with
      | .thmInfo _ =>
        VeriTile.Meta.auditAxioms name
        count := count + 1
      | _ => pure ()
  Lean.logInfo m!"Complete port theorem audit: {count} declarations"
