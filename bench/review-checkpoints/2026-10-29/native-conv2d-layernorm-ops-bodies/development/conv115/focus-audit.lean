import VeriTile.Meta.StatementAudit
import bench.repairs.tritonbench_g.triton_conv2d_fwd.CandidateKernels
import bench.repairs.tritonbench_g.triton_conv2d_fwd.Contract
import bench.repairs.tritonbench_g.triton_conv2d_fwd.Correct
import bench.repairs.tritonbench_g.triton_conv2d_fwd.FPEquiv
import bench.repairs.tritonbench_g.triton_conv2d_fwd.IO
import bench.repairs.tritonbench_g.triton_conv2d_fwd.Kernels
import bench.repairs.tritonbench_g.triton_conv2d_fwd.Misc.AuxiliaryFPEquiv
import bench.repairs.tritonbench_g.triton_conv2d_fwd.Misc.NativeCorrect
import bench.repairs.tritonbench_g.triton_conv2d_fwd.Misc.Reader
import bench.repairs.tritonbench_g.triton_conv2d_fwd.Misc.Review
import bench.repairs.tritonbench_g.triton_conv2d_fwd.Misc.TensorBridge
import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.Arithmetic
import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.Contract
import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.Coordinates
import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.Formed
import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.Grid
import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.Host
import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.ISA
import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.KernelBinding
import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.KernelDefaultComplete
import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.KernelDefaultExecution
import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.KernelDefaultGraph
import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.KernelDefaultLoops
import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.KernelDefaultNative
import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.KernelDefaultSafety
import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.KernelDefaultSource
import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.KernelEntries
import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.KernelExecution
import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.KernelGrid
import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.KernelHalfComplete
import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.KernelHalfExecution
import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.KernelHalfGraph
import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.KernelHalfLoops
import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.KernelHalfSafety
import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.KernelHalfSource
import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.KernelInputs
import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.KernelLaunch
import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.KernelNative
import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.KernelNativeResult
import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.KernelNumeric
import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.KernelNumericProof
import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.KernelOperands
import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.KernelReal
import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.KernelRealResult
import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.KernelReference
import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.KernelStorage
import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.KernelStores
import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.KernelTensor
import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.LogicalIO
import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.LogicalInputs
import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.LogicalReference
import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.Lowering
import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.Memory
import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.Native
import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.Ownership
import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.Pair
import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.Physical
import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.PublicResult
import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.PublicSafety
import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.Real
import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.Safety
import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.Scatter
import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.Software
import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.SourceIndex
import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.Witness
import bench.repairs.tritonbench_g.triton_conv2d_fwd.Spec
import bench.tests.TypedDotLoopBody
import bench.tritonbench_g.triton_conv2d_fwd.BranchesCorrect
import bench.tritonbench_g.triton_conv2d_fwd.FPCorrect
import bench.tritonbench_g.triton_conv2d_fwd.Kernels
import bench.tritonbench_g.triton_conv2d_fwd.Proofs.FPCleanUndef
import bench.tritonbench_g.triton_conv2d_fwd.Proofs.FPComplete
import bench.tritonbench_g.triton_conv2d_fwd.Proofs.FPDefaultComplete
import bench.tritonbench_g.triton_conv2d_fwd.Proofs.FPDefaultExecution
import bench.tritonbench_g.triton_conv2d_fwd.Proofs.FPDefaultLoops
import bench.tritonbench_g.triton_conv2d_fwd.Proofs.FPDefaultReference
import bench.tritonbench_g.triton_conv2d_fwd.Proofs.FPDefaultSoftware
import bench.tritonbench_g.triton_conv2d_fwd.Proofs.FPDefaultTypedSource
import bench.tritonbench_g.triton_conv2d_fwd.Proofs.FPDefaultWitness
import bench.tritonbench_g.triton_conv2d_fwd.Proofs.FPExecution
import bench.tritonbench_g.triton_conv2d_fwd.Proofs.FPLoops
import bench.tritonbench_g.triton_conv2d_fwd.Proofs.FPReference
import bench.tritonbench_g.triton_conv2d_fwd.Proofs.FPShape
import bench.tritonbench_g.triton_conv2d_fwd.Proofs.FPSoftware
import bench.tritonbench_g.triton_conv2d_fwd.Proofs.FPTypedSource
import bench.tritonbench_g.triton_conv2d_fwd.Proofs.FPWitness
import bench.tritonbench_g.triton_conv2d_fwd.Proofs.MaskedPointerSimulation
import bench.tritonbench_g.triton_conv2d_fwd.Proofs.PositivePointerCorrect
import bench.tritonbench_g.triton_conv2d_fwd.Proofs.PositivePointerModel
import bench.tritonbench_g.triton_conv2d_fwd.Proofs.SignedBodySimulation
import bench.tritonbench_g.triton_conv2d_fwd.Proofs.SignedFlatSimulation
import bench.tritonbench_g.triton_conv2d_fwd.Proofs.SignedLoopSimulation
import bench.tritonbench_g.triton_conv2d_fwd.Proofs.SignedPointerModel
import bench.tritonbench_g.triton_conv2d_fwd.Proofs.SignedPostSimulation
import bench.tritonbench_g.triton_conv2d_fwd.TritonConv2dFwd

open Lean Elab Command
run_cmd do
  let env ← getEnv
  let modules : Array Name := #[`VeriTile.Meta.StatementAudit,`bench.repairs.tritonbench_g.triton_conv2d_fwd.CandidateKernels,`bench.repairs.tritonbench_g.triton_conv2d_fwd.Contract,`bench.repairs.tritonbench_g.triton_conv2d_fwd.Correct,`bench.repairs.tritonbench_g.triton_conv2d_fwd.FPEquiv,`bench.repairs.tritonbench_g.triton_conv2d_fwd.IO,`bench.repairs.tritonbench_g.triton_conv2d_fwd.Kernels,`bench.repairs.tritonbench_g.triton_conv2d_fwd.Misc.AuxiliaryFPEquiv,`bench.repairs.tritonbench_g.triton_conv2d_fwd.Misc.NativeCorrect,`bench.repairs.tritonbench_g.triton_conv2d_fwd.Misc.Reader,`bench.repairs.tritonbench_g.triton_conv2d_fwd.Misc.Review,`bench.repairs.tritonbench_g.triton_conv2d_fwd.Misc.TensorBridge,`bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.Arithmetic,`bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.Contract,`bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.Coordinates,`bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.Formed,`bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.Grid,`bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.Host,`bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.ISA,`bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.KernelBinding,`bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.KernelDefaultComplete,`bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.KernelDefaultExecution,`bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.KernelDefaultGraph,`bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.KernelDefaultLoops,`bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.KernelDefaultNative,`bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.KernelDefaultSafety,`bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.KernelDefaultSource,`bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.KernelEntries,`bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.KernelExecution,`bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.KernelGrid,`bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.KernelHalfComplete,`bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.KernelHalfExecution,`bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.KernelHalfGraph,`bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.KernelHalfLoops,`bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.KernelHalfSafety,`bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.KernelHalfSource,`bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.KernelInputs,`bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.KernelLaunch,`bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.KernelNative,`bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.KernelNativeResult,`bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.KernelNumeric,`bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.KernelNumericProof,`bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.KernelOperands,`bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.KernelReal,`bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.KernelRealResult,`bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.KernelReference,`bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.KernelStorage,`bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.KernelStores,`bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.KernelTensor,`bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.LogicalIO,`bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.LogicalInputs,`bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.LogicalReference,`bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.Lowering,`bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.Memory,`bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.Native,`bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.Ownership,`bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.Pair,`bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.Physical,`bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.PublicResult,`bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.PublicSafety,`bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.Real,`bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.Safety,`bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.Scatter,`bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.Software,`bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.SourceIndex,`bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.Witness,`bench.repairs.tritonbench_g.triton_conv2d_fwd.Spec,`bench.tests.TypedDotLoopBody,`bench.tritonbench_g.triton_conv2d_fwd.BranchesCorrect,`bench.tritonbench_g.triton_conv2d_fwd.FPCorrect,`bench.tritonbench_g.triton_conv2d_fwd.Kernels,`bench.tritonbench_g.triton_conv2d_fwd.Proofs.FPCleanUndef,`bench.tritonbench_g.triton_conv2d_fwd.Proofs.FPComplete,`bench.tritonbench_g.triton_conv2d_fwd.Proofs.FPDefaultComplete,`bench.tritonbench_g.triton_conv2d_fwd.Proofs.FPDefaultExecution,`bench.tritonbench_g.triton_conv2d_fwd.Proofs.FPDefaultLoops,`bench.tritonbench_g.triton_conv2d_fwd.Proofs.FPDefaultReference,`bench.tritonbench_g.triton_conv2d_fwd.Proofs.FPDefaultSoftware,`bench.tritonbench_g.triton_conv2d_fwd.Proofs.FPDefaultTypedSource,`bench.tritonbench_g.triton_conv2d_fwd.Proofs.FPDefaultWitness,`bench.tritonbench_g.triton_conv2d_fwd.Proofs.FPExecution,`bench.tritonbench_g.triton_conv2d_fwd.Proofs.FPLoops,`bench.tritonbench_g.triton_conv2d_fwd.Proofs.FPReference,`bench.tritonbench_g.triton_conv2d_fwd.Proofs.FPShape,`bench.tritonbench_g.triton_conv2d_fwd.Proofs.FPSoftware,`bench.tritonbench_g.triton_conv2d_fwd.Proofs.FPTypedSource,`bench.tritonbench_g.triton_conv2d_fwd.Proofs.FPWitness,`bench.tritonbench_g.triton_conv2d_fwd.Proofs.MaskedPointerSimulation,`bench.tritonbench_g.triton_conv2d_fwd.Proofs.PositivePointerCorrect,`bench.tritonbench_g.triton_conv2d_fwd.Proofs.PositivePointerModel,`bench.tritonbench_g.triton_conv2d_fwd.Proofs.SignedBodySimulation,`bench.tritonbench_g.triton_conv2d_fwd.Proofs.SignedFlatSimulation,`bench.tritonbench_g.triton_conv2d_fwd.Proofs.SignedLoopSimulation,`bench.tritonbench_g.triton_conv2d_fwd.Proofs.SignedPointerModel,`bench.tritonbench_g.triton_conv2d_fwd.Proofs.SignedPostSimulation,`bench.tritonbench_g.triton_conv2d_fwd.TritonConv2dFwd]
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
#print_fp_assumptions VeriTile.Bench.Repairs.Conv2d.FPEquiv.grid_equiv
