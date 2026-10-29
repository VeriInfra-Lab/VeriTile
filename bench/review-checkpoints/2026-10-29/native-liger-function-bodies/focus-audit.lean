import VeriTile.Meta.StatementAudit
import bench.tests.TypedIntegerOperandBody
import bench.tests.KnownIntegerRangeStepBody
import bench.repairs.tritonbench_g.layer_norm_liger.Contract
import bench.repairs.tritonbench_g.layer_norm_liger.Correct
import bench.repairs.tritonbench_g.layer_norm_liger.FPEquiv
import bench.repairs.tritonbench_g.layer_norm_liger.IO
import bench.repairs.tritonbench_g.layer_norm_liger.Kernels
import bench.repairs.tritonbench_g.layer_norm_liger.Misc.HostCorrect
import bench.repairs.tritonbench_g.layer_norm_liger.Misc.HostFPEquiv
import bench.repairs.tritonbench_g.layer_norm_liger.Misc.KernelAudit
import bench.repairs.tritonbench_g.layer_norm_liger.Misc.KernelProofAudit
import bench.repairs.tritonbench_g.layer_norm_liger.Misc.NativeCorrect
import bench.repairs.tritonbench_g.layer_norm_liger.Misc.TensorBridge
import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.BackwardContract
import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.BackwardInterface
import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.BackwardKernels
import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.BackwardModel
import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.BackwardResult
import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.BackwardRun
import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.BackwardSoftware
import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.BackwardWhole
import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.CandidateBackwardRun
import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.CandidateBackwardSource
import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.CandidateBackwardWhole
import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.CandidateForward
import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.CandidateSafety
import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.CandidateSafetyForward
import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.ClosedFP
import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.Contract
import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.FoldFacts
import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.ForwardContract
import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.ForwardInterface
import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.ForwardModel
import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.ForwardResult
import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.ForwardRun
import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.ForwardSaved
import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.ForwardSoftware
import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.Frame
import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.Grid
import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.GridCells
import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.GridHost
import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.HostContract
import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.HostResults
import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.IOBindings
import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.KernelBoundary
import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.KernelFP
import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.KernelIndex
import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.KernelReal
import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.KernelStrides
import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.LaunchCoverage
import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.Logical
import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.LogicalBackward
import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.LogicalForward
import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.LogicalIO
import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.LogicalInputs
import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.NativeIndex
import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.NativeInterface
import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.NativeLaws
import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.NativePlan
import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.OrderedMemory
import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.OrderedReference
import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.Preparation
import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.ProgramPair
import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.PublicCheck
import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.PublicReturn
import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.RealInterface
import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.ReductionPlan
import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.Returned
import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.Settings
import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.SourceOffsets
import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.SourceSafety
import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.SourceShape
import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.WholeCheck
import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.Wrapper
import bench.tritonbench_g.layer_norm_liger.FPCorrect
import bench.tritonbench_g.layer_norm_liger.LayerNormLiger
import bench.tritonbench_g.layer_norm_liger.Proofs.FPComplete
import bench.tritonbench_g.layer_norm_liger.Proofs.FPContract
import bench.tritonbench_g.layer_norm_liger.Proofs.FPModel
import bench.tritonbench_g.layer_norm_liger.Proofs.FPShape
import bench.tritonbench_g.layer_norm_liger.Proofs.FPSoftware
import bench.tritonbench_g.layer_norm_liger.Proofs.FPSupplied
import bench.tritonbench_g.layer_norm_liger.Proofs.FPTypedSource
import bench.tritonbench_g.layer_norm_liger.Proofs.FPWitness

open Lean Elab Command
run_cmd do
  let env ← getEnv
  let modules : Array Name := #[`VeriTile.Meta.StatementAudit,`bench.repairs.tritonbench_g.layer_norm_liger.Contract,`bench.repairs.tritonbench_g.layer_norm_liger.Correct,`bench.repairs.tritonbench_g.layer_norm_liger.FPEquiv,`bench.repairs.tritonbench_g.layer_norm_liger.IO,`bench.repairs.tritonbench_g.layer_norm_liger.Kernels,`bench.repairs.tritonbench_g.layer_norm_liger.Misc.HostCorrect,`bench.repairs.tritonbench_g.layer_norm_liger.Misc.HostFPEquiv,`bench.repairs.tritonbench_g.layer_norm_liger.Misc.KernelAudit,`bench.repairs.tritonbench_g.layer_norm_liger.Misc.KernelProofAudit,`bench.repairs.tritonbench_g.layer_norm_liger.Misc.NativeCorrect,`bench.repairs.tritonbench_g.layer_norm_liger.Misc.TensorBridge,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.BackwardContract,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.BackwardInterface,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.BackwardKernels,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.BackwardModel,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.BackwardResult,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.BackwardRun,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.BackwardSoftware,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.BackwardWhole,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.CandidateBackwardRun,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.CandidateBackwardSource,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.CandidateBackwardWhole,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.CandidateForward,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.CandidateSafety,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.CandidateSafetyForward,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.ClosedFP,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.Contract,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.FoldFacts,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.ForwardContract,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.ForwardInterface,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.ForwardModel,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.ForwardResult,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.ForwardRun,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.ForwardSaved,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.ForwardSoftware,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.Frame,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.Grid,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.GridCells,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.GridHost,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.HostContract,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.HostResults,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.IOBindings,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.KernelBoundary,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.KernelFP,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.KernelIndex,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.KernelReal,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.KernelStrides,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.LaunchCoverage,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.Logical,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.LogicalBackward,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.LogicalForward,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.LogicalIO,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.LogicalInputs,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.NativeIndex,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.NativeInterface,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.NativeLaws,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.NativePlan,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.OrderedMemory,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.OrderedReference,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.Preparation,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.ProgramPair,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.PublicCheck,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.PublicReturn,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.RealInterface,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.ReductionPlan,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.Returned,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.Settings,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.SourceOffsets,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.SourceSafety,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.SourceShape,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.WholeCheck,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.Wrapper,`bench.tests.KnownIntegerRangeStepBody,`bench.tests.TypedIntegerOperandBody,`bench.tritonbench_g.layer_norm_liger.FPCorrect,`bench.tritonbench_g.layer_norm_liger.LayerNormLiger,`bench.tritonbench_g.layer_norm_liger.Proofs.FPComplete,`bench.tritonbench_g.layer_norm_liger.Proofs.FPContract,`bench.tritonbench_g.layer_norm_liger.Proofs.FPModel,`bench.tritonbench_g.layer_norm_liger.Proofs.FPShape,`bench.tritonbench_g.layer_norm_liger.Proofs.FPSoftware,`bench.tritonbench_g.layer_norm_liger.Proofs.FPSupplied,`bench.tritonbench_g.layer_norm_liger.Proofs.FPTypedSource,`bench.tritonbench_g.layer_norm_liger.Proofs.FPWitness]
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
#print_fp_assumptions VeriTile.Bench.Repairs.LayerNormLiger.Kernel.FPEquiv.forward_pair
#print_fp_assumptions VeriTile.Bench.Repairs.LayerNormLiger.Kernel.FPEquiv.backward_pair
