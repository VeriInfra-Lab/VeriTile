import bench.repairs.tritonbench_g.ksoftmax_triton_grid_y_checked.Proofs.PrecisionProfile
import bench.repairs.tritonbench_g.ksoftmax_triton.Contract
import bench.repairs.tritonbench_g.ksoftmax_triton.Correct
import bench.repairs.tritonbench_g.ksoftmax_triton.FPEquiv
import bench.repairs.tritonbench_g.ksoftmax_triton.Kernels
import bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.Binding
import bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.Congruence
import bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.Contract
import bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.Counterexamples
import bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.Extended
import bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.FP
import bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.Graph
import bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.Grid
import bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.GridValue
import bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.Host
import bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.HostCorrect
import bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.HostFPEquiv
import bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.KernelAccess
import bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.KernelCompiler
import bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.KernelCorrect
import bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.KernelExecution
import bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.KernelExtended
import bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.KernelIdeal
import bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.KernelInputs
import bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.KernelLogBackward
import bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.KernelNative
import bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.KernelSchedule
import bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.KernelTensor
import bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.KernelWordCorrect
import bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.Math
import bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.Native
import bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.Observed
import bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.Pointers
import bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.PrecisionProfile
import bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.Real
import bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.RowValue
import bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.Source
import bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.ValueMath
import bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.Words
import bench.repairs.tritonbench_g.ksoftmax_triton.SourceParameters
import bench.repairs.tritonbench_g.ksoftmax_triton_grid_y_checked.Contract
import bench.repairs.tritonbench_g.ksoftmax_triton_grid_y_checked.Correct
import bench.repairs.tritonbench_g.ksoftmax_triton_grid_y_checked.FPEquiv
import bench.repairs.tritonbench_g.ksoftmax_triton_grid_y_checked.IO
import bench.repairs.tritonbench_g.ksoftmax_triton_grid_y_checked.Kernels
import bench.repairs.tritonbench_g.ksoftmax_triton_grid_y_checked.Misc.InterfaceChecks
import bench.repairs.tritonbench_g.ksoftmax_triton_grid_y_checked.Misc.Native
import bench.repairs.tritonbench_g.ksoftmax_triton_grid_y_checked.Misc.TensorBridge
import bench.repairs.tritonbench_g.ksoftmax_triton_grid_y_checked.Proofs.AccessSafety
import bench.repairs.tritonbench_g.ksoftmax_triton_grid_y_checked.Proofs.Interface
import bench.repairs.tritonbench_g.ksoftmax_triton_grid_y_checked.Proofs.LogicalReader
import bench.repairs.tritonbench_g.ksoftmax_triton_grid_y_checked.Proofs.LogicalWidth
import bench.repairs.tritonbench_g.ksoftmax_triton_grid_y_checked.Proofs.NativeIO
import bench.repairs.tritonbench_g.ksoftmax_triton_grid_y_checked.Proofs.RawKernel
import bench.repairs.tritonbench_g.ksoftmax_triton_grid_y_checked.Proofs.RawNative
import bench.repairs.tritonbench_g.ksoftmax_triton_grid_y_checked.Proofs.RawReal
import bench.repairs.tritonbench_g.ksoftmax_triton_grid_y_checked.Proofs.RealIO
import bench.repairs.tritonbench_g.ksoftmax_triton_grid_y_checked.Proofs.ScheduleEquivalence
import bench.tritonbench_g.ksoftmax_triton.Correct
import bench.tritonbench_g.ksoftmax_triton.FPEquiv
import bench.tritonbench_g.ksoftmax_triton.Kernels
import bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardBoolControl
import bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardCandidate
import bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardCandidatePrefix
import bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardCandidateStructure
import bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardDTypeProbe
import bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardErasure
import bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardErasureProbe
import bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardExecution
import bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardFP
import bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardFullErasure
import bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardFullFP
import bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardFullTrace
import bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardFullTypedIO
import bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardIO
import bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardMath
import bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardMathSource
import bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardOrdinaryProbe
import bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardOrdinaryValueProbe
import bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardProfiles
import bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardPublic
import bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardReal
import bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardRun
import bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardSource
import bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardStaticControl
import bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardSymbolicErasureProbe
import bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardTraceControl
import bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardTraceProbe
import bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardTypedExecution
import bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardTypedIO
import bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardTypedModel
import bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardTypedProbe
import bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardTypedValueProbe
import bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardTypedValues
import bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardTypedVariant
import bench.tritonbench_g.ksoftmax_triton.Proofs.ForwardCandidateReal
import bench.tritonbench_g.ksoftmax_triton.Proofs.ForwardConstructors
import bench.tritonbench_g.ksoftmax_triton.Proofs.ForwardFP
import bench.tritonbench_g.ksoftmax_triton.Proofs.ForwardMath
import bench.tritonbench_g.ksoftmax_triton.Proofs.ForwardPublic
import bench.tritonbench_g.ksoftmax_triton.Proofs.ForwardRealFrame
import bench.tritonbench_g.ksoftmax_triton.Proofs.ForwardRealIO
import bench.tritonbench_g.ksoftmax_triton.Proofs.ForwardRealReadback
import bench.tritonbench_g.ksoftmax_triton.Proofs.ForwardRealRun
import bench.tritonbench_g.ksoftmax_triton.Proofs.ForwardSource
import bench.tritonbench_g.ksoftmax_triton.Proofs.ForwardSourceScore
import bench.tritonbench_g.ksoftmax_triton.Proofs.ForwardTileBridge
import bench.tritonbench_g.ksoftmax_triton.Proofs.ForwardTypedBKProbe
import bench.tritonbench_g.ksoftmax_triton.Proofs.ForwardTypedBKValues
import bench.tritonbench_g.ksoftmax_triton.Proofs.ForwardTypedCandidate
import bench.tritonbench_g.ksoftmax_triton.Proofs.ForwardTypedModel
import bench.tritonbench_g.ksoftmax_triton.Proofs.ForwardTypedProbe
import bench.tritonbench_g.ksoftmax_triton.Proofs.ForwardTypedProfilesTotal
import bench.tritonbench_g.ksoftmax_triton.Proofs.ForwardTypedTotal
import bench.tritonbench_g.ksoftmax_triton.Proofs.ForwardTypedValues
import bench.tritonbench_g.ksoftmax_triton.Proofs.ForwardVariantIO
import VeriTile.Meta.StatementAudit

open Lean Elab Command
run_cmd do
  let env ← getEnv
  let modules : Array Name := #[`bench.repairs.tritonbench_g.ksoftmax_triton_grid_y_checked.Proofs.PrecisionProfile,`bench.repairs.tritonbench_g.ksoftmax_triton.Contract,`bench.repairs.tritonbench_g.ksoftmax_triton.Correct,`bench.repairs.tritonbench_g.ksoftmax_triton.FPEquiv,`bench.repairs.tritonbench_g.ksoftmax_triton.Kernels,`bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.Binding,`bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.Congruence,`bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.Contract,`bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.Counterexamples,`bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.Extended,`bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.FP,`bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.Graph,`bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.Grid,`bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.GridValue,`bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.Host,`bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.HostCorrect,`bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.HostFPEquiv,`bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.KernelAccess,`bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.KernelCompiler,`bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.KernelCorrect,`bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.KernelExecution,`bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.KernelExtended,`bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.KernelIdeal,`bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.KernelInputs,`bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.KernelLogBackward,`bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.KernelNative,`bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.KernelSchedule,`bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.KernelTensor,`bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.KernelWordCorrect,`bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.Math,`bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.Native,`bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.Observed,`bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.Pointers,`bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.PrecisionProfile,`bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.Real,`bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.RowValue,`bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.Source,`bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.ValueMath,`bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.Words,`bench.repairs.tritonbench_g.ksoftmax_triton.SourceParameters,`bench.repairs.tritonbench_g.ksoftmax_triton_grid_y_checked.Contract,`bench.repairs.tritonbench_g.ksoftmax_triton_grid_y_checked.Correct,`bench.repairs.tritonbench_g.ksoftmax_triton_grid_y_checked.FPEquiv,`bench.repairs.tritonbench_g.ksoftmax_triton_grid_y_checked.IO,`bench.repairs.tritonbench_g.ksoftmax_triton_grid_y_checked.Kernels,`bench.repairs.tritonbench_g.ksoftmax_triton_grid_y_checked.Misc.InterfaceChecks,`bench.repairs.tritonbench_g.ksoftmax_triton_grid_y_checked.Misc.Native,`bench.repairs.tritonbench_g.ksoftmax_triton_grid_y_checked.Misc.TensorBridge,`bench.repairs.tritonbench_g.ksoftmax_triton_grid_y_checked.Proofs.AccessSafety,`bench.repairs.tritonbench_g.ksoftmax_triton_grid_y_checked.Proofs.Interface,`bench.repairs.tritonbench_g.ksoftmax_triton_grid_y_checked.Proofs.LogicalReader,`bench.repairs.tritonbench_g.ksoftmax_triton_grid_y_checked.Proofs.LogicalWidth,`bench.repairs.tritonbench_g.ksoftmax_triton_grid_y_checked.Proofs.NativeIO,`bench.repairs.tritonbench_g.ksoftmax_triton_grid_y_checked.Proofs.RawKernel,`bench.repairs.tritonbench_g.ksoftmax_triton_grid_y_checked.Proofs.RawNative,`bench.repairs.tritonbench_g.ksoftmax_triton_grid_y_checked.Proofs.RawReal,`bench.repairs.tritonbench_g.ksoftmax_triton_grid_y_checked.Proofs.RealIO,`bench.repairs.tritonbench_g.ksoftmax_triton_grid_y_checked.Proofs.ScheduleEquivalence,`bench.tritonbench_g.ksoftmax_triton.Correct,`bench.tritonbench_g.ksoftmax_triton.FPEquiv,`bench.tritonbench_g.ksoftmax_triton.Kernels,`bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardBoolControl,`bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardCandidate,`bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardCandidatePrefix,`bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardCandidateStructure,`bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardDTypeProbe,`bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardErasure,`bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardErasureProbe,`bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardExecution,`bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardFP,`bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardFullErasure,`bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardFullFP,`bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardFullTrace,`bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardFullTypedIO,`bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardIO,`bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardMath,`bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardMathSource,`bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardOrdinaryProbe,`bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardOrdinaryValueProbe,`bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardProfiles,`bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardPublic,`bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardReal,`bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardRun,`bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardSource,`bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardStaticControl,`bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardSymbolicErasureProbe,`bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardTraceControl,`bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardTraceProbe,`bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardTypedExecution,`bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardTypedIO,`bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardTypedModel,`bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardTypedProbe,`bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardTypedValueProbe,`bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardTypedValues,`bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardTypedVariant,`bench.tritonbench_g.ksoftmax_triton.Proofs.ForwardCandidateReal,`bench.tritonbench_g.ksoftmax_triton.Proofs.ForwardConstructors,`bench.tritonbench_g.ksoftmax_triton.Proofs.ForwardFP,`bench.tritonbench_g.ksoftmax_triton.Proofs.ForwardMath,`bench.tritonbench_g.ksoftmax_triton.Proofs.ForwardPublic,`bench.tritonbench_g.ksoftmax_triton.Proofs.ForwardRealFrame,`bench.tritonbench_g.ksoftmax_triton.Proofs.ForwardRealIO,`bench.tritonbench_g.ksoftmax_triton.Proofs.ForwardRealReadback,`bench.tritonbench_g.ksoftmax_triton.Proofs.ForwardRealRun,`bench.tritonbench_g.ksoftmax_triton.Proofs.ForwardSource,`bench.tritonbench_g.ksoftmax_triton.Proofs.ForwardSourceScore,`bench.tritonbench_g.ksoftmax_triton.Proofs.ForwardTileBridge,`bench.tritonbench_g.ksoftmax_triton.Proofs.ForwardTypedBKProbe,`bench.tritonbench_g.ksoftmax_triton.Proofs.ForwardTypedBKValues,`bench.tritonbench_g.ksoftmax_triton.Proofs.ForwardTypedCandidate,`bench.tritonbench_g.ksoftmax_triton.Proofs.ForwardTypedModel,`bench.tritonbench_g.ksoftmax_triton.Proofs.ForwardTypedProbe,`bench.tritonbench_g.ksoftmax_triton.Proofs.ForwardTypedProfilesTotal,`bench.tritonbench_g.ksoftmax_triton.Proofs.ForwardTypedTotal,`bench.tritonbench_g.ksoftmax_triton.Proofs.ForwardTypedValues,`bench.tritonbench_g.ksoftmax_triton.Proofs.ForwardVariantIO]
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
  for name in names do VeriTile.Meta.auditAxioms name
  logInfo m!"Task theorem audit count: {names.size}"
#print_fp_assumptions VeriTile.Bench.Repairs.KsoftmaxGridYChecked.FPEquiv.reversed_schedule_equivalent
