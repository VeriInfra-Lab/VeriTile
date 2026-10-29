import VeriTile.Meta.StatementAudit
import bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Compose
import bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Contract
import bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Coordinates
import bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Correct
import bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Execution
import bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.FPEquiv
import bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Geometry
import bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.IO
import bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Inner
import bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Kernels
import bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Loop
import bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Misc.Audit
import bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Misc.CandidateCorrect
import bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Misc.NativeCorrect
import bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Misc.TensorBridge
import bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Model
import bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Native
import bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Outer
import bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.PairIO
import bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Physical
import bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Pointer
import bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.PrefixExecution
import bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Prelude
import bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Proofs.IOBounds
import bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Proofs.InputConnection
import bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Proofs.KernelNativeResult
import bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Proofs.KernelRealResult
import bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Proofs.LogicalIO
import bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Proofs.Pair
import bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Proofs.PairIO
import bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Proofs.PublicData
import bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Proofs.PublicResult
import bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Proofs.SafetyAccess
import bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Proofs.SafetyLoops
import bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Proofs.SafetySteps
import bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Public
import bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.RawKernel
import bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Registers
import bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Setup
import bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Store
import bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Suffix
import bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Targets
import bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Tensor
import bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Tile
import bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Values
import bench.tests.BoolArithmeticBody
import bench.tritonbench_g.matmul_persistent_triton.Proofs.Source
import bench.tritonbench_g.matmul_persistent_triton.Types

open Lean Elab Command
run_cmd do
  let env ← getEnv
  let modules : Array Name := #[`VeriTile.Meta.StatementAudit,`bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Compose,`bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Contract,`bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Coordinates,`bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Correct,`bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Execution,`bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.FPEquiv,`bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Geometry,`bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.IO,`bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Inner,`bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Kernels,`bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Loop,`bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Misc.Audit,`bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Misc.CandidateCorrect,`bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Misc.NativeCorrect,`bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Misc.TensorBridge,`bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Model,`bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Native,`bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Outer,`bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.PairIO,`bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Physical,`bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Pointer,`bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.PrefixExecution,`bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Prelude,`bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Proofs.IOBounds,`bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Proofs.InputConnection,`bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Proofs.KernelNativeResult,`bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Proofs.KernelRealResult,`bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Proofs.LogicalIO,`bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Proofs.Pair,`bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Proofs.PairIO,`bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Proofs.PublicData,`bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Proofs.PublicResult,`bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Proofs.SafetyAccess,`bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Proofs.SafetyLoops,`bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Proofs.SafetySteps,`bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Public,`bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.RawKernel,`bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Registers,`bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Setup,`bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Store,`bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Suffix,`bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Targets,`bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Tensor,`bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Tile,`bench.repairs.tritonbench_g.matmul_persistent_triton_scalar_cast.Values,`bench.tests.BoolArithmeticBody,`bench.tritonbench_g.matmul_persistent_triton.Proofs.Source,`bench.tritonbench_g.matmul_persistent_triton.Types]
  for m in modules do
    unless env.header.modules.any (·.module == m) do
      throwError "Audit root not imported: {m}"
  let mut names : Array Name := #[]
  for (name, info) in env.constants.toList do
    let .thmInfo _ := info | continue
    let some idx := env.getModuleIdxFor? name | continue
    if modules.contains env.header.modules[idx.toNat]!.module then
      names := names.push name
  names := names.qsort (·.toString < ·.toString)
  -- One immutable checked environment, with shared visited-declaration state.
  -- The accumulated dependency union is checked after every root. Therefore
  -- each individual root's axiom footprint is contained in the standard base.
  let allowed : Array Name := #[`propext, `Classical.choice, `Quot.sound]
  let mut footprint : CollectAxioms.State := {}
  for name in names do
    unless (env.checked.get.find? name).isSome do
      throwError "Audit root missing from checked environment: {name}"
    let (_, next) := ((CollectAxioms.collect name).run env).run footprint
    footprint := next
    let bad := footprint.axioms.filter (fun a => ! allowed.contains a)
    unless bad.isEmpty do
      throwError "DISALLOWED axiom dependency from audit prefix ending at {name}: {bad}"
    logInfo m!"{name.toString}: axiom footprint ⊆ standard base ✓ (shared traversal)"
  logInfo m!"Checked dependency union: {footprint.axioms}"
  logInfo m!"Task theorem audit count: {names.size}"
#print_fp_assumptions VeriTile.Bench.Repairs.MatmulPersistentTritonScalarCast.FPEquiv.source_pair_equiv
