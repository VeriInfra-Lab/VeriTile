import bench.tritonbench_g.rmsnorm_implementation.Correct
import bench.tritonbench_g.rmsnorm_implementation.FPEquiv

#print axioms VeriTile.Bench.TritonBenchG.RmsnormImplementation.rmsnorm_implementation_io_correctness
#print axioms VeriTile.Bench.TritonBenchG.RmsnormImplementation.typed_real_correct
#print axioms VeriTile.Bench.TritonBenchG.RmsnormImplementation.full_real_correct
#print axioms VeriTile.Bench.TritonBenchG.RmsnormImplementation.FPEquiv.source_equivalence
#print axioms VeriTile.Bench.TritonBenchG.RmsnormImplementation.typed_full_exact_io
#print axioms VeriTile.Bench.TritonBenchG.RmsnormImplementation.typed_trace_safe
#print axioms VeriTile.Bench.TritonBenchG.RmsnormImplementation.originalTyped
#print axioms VeriTile.Bench.TritonBenchG.RmsnormImplementation.typedKernel
#print axioms VeriTile.Bench.TritonBenchG.RmsnormImplementation.Precision.source_shape
#print axioms VeriTile.Bench.TritonBenchG.RmsnormImplementation.Precision.source_loads
#print axioms VeriTile.Bench.TritonBenchG.RmsnormImplementation.Precision.source_arithmetic
#print axioms VeriTile.Bench.TritonBenchG.RmsnormImplementation.Precision.variance_precision
#print axioms VeriTile.Bench.TritonBenchG.RmsnormImplementation.Precision.io_roles
#print axioms VeriTile.Bench.TritonBenchG.RmsnormImplementation.FiniteWitness.contract
#print axioms VeriTile.Bench.TritonBenchG.RmsnormImplementation.FiniteWitness.cells
#print axioms VeriTile.Bench.TritonBenchG.RmsnormImplementation.NumericExecution.numeric_run
#print axioms VeriTile.Bench.TritonBenchG.RmsnormImplementation.NumericExecution.numeric_cells
#print axioms VeriTile.Bench.TritonBenchG.RmsnormImplementation.NumericExecution.numeric_frame
#print axioms VeriTile.Bench.TritonBenchG.RmsnormImplementation.NumericExecution.numerical_execution
#specNonCircular VeriTile.Bench.TritonBenchG.RmsnormImplementation.rmsImplStreamSpec avoiding [VeriTile.Bench.TritonBenchG.RmsnormImplementation.originalTyped, VeriTile.Bench.TritonBenchG.RmsnormImplementation.typedKernel, VeriTile.Triton.exec, VeriTile.Triton.execR]
#specNonCircular VeriTile.Bench.TritonBenchG.RmsnormImplementation.rmsnormWeightedYFullNSpec avoiding [VeriTile.Bench.TritonBenchG.RmsnormImplementation.rmsnorm_implementation, VeriTile.Triton.exec, VeriTile.Triton.execR]
#print_fp_assumptions VeriTile.Bench.TritonBenchG.RmsnormImplementation.FPEquiv.source_equivalence

#guard_msgs (drop info) in
#auditModuleAxioms

/- Audit every theorem in the port, including private historical helpers. -/
run_cmd do
  let env ← Lean.getEnv
  let mut count := 0
  for (name, info) in env.constants.toList do
    let userName := ((Lean.privateToUserName? name).getD name).eraseMacroScopes
    if (`VeriTile.Bench.TritonBenchG.RmsnormImplementation).isPrefixOf userName then
      match info with
      | .thmInfo _ =>
        VeriTile.Meta.auditAxioms name
        count := count + 1
      | _ => pure ()
  Lean.logInfo m!"Complete port theorem audit: {count} declarations"
