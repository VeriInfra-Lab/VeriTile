import VeriTile.Meta.StatementAudit
import bench.tests.OptionalPointerExpressionBody
import bench.tests.OptionalPointerBody
import bench.tests.ConstexprMaskBody
import bench.repairs.tritonbench_g.layer_norm_ops.Contract
import bench.repairs.tritonbench_g.layer_norm_ops.Correct
import bench.repairs.tritonbench_g.layer_norm_ops.FPEquiv
import bench.repairs.tritonbench_g.layer_norm_ops.IO
import bench.repairs.tritonbench_g.layer_norm_ops.Kernels
import bench.repairs.tritonbench_g.layer_norm_ops.Misc.NativeCorrect
import bench.repairs.tritonbench_g.layer_norm_ops.Misc.TensorBridge
import bench.repairs.tritonbench_g.layer_norm_ops.PairIO
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.Allocation
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.BackwardBody000
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.BackwardBody001
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.BackwardBody010
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.BackwardBody011
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.BackwardBody100
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.BackwardBody101
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.BackwardBody110
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.BackwardBody111
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.BackwardIO
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.BackwardInvariant
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.BackwardModel
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.BackwardPrelude00
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.BackwardPrelude01
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.BackwardPrelude10
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.BackwardPrelude11
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.BackwardRun
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.BackwardValues
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.BackwardWhole
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.CheckedHost
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.Contract
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.FiniteReference
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.ForwardIO
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.ForwardModel
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.ForwardRun
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.ForwardRun00
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.ForwardRun01
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.ForwardRun10
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.ForwardRun11
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.ForwardSafety
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.ForwardSafety00
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.ForwardSafety01
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.ForwardSafety10
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.ForwardSafety11
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.Frame
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.Host
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.HostCorrect
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.HostFPEquiv
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.HostResults
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.IONative
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.IOReal
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.IOResults
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.IOSafety
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.KernelBound
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.KernelCompiler
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.KernelCoverage
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.KernelDispatch
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.KernelExecution
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.KernelFrame
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.KernelIndex
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.KernelInputs
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.KernelInvariant
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.KernelNative
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.KernelReadback
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.KernelReal
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.KernelReference
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.Logical
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.LogicalBackward
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.LogicalHost
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.LogicalIO
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.LogicalLoop
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.LogicalResult
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.MathBackwardLoop
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.MathForward
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.MathGrid
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.MathModel
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.MathPartials
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.MathRecomputed
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.MathReturned
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.MathSaved
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.MathValues
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.Mathematical
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.MemoryHelpers
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.Native
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.NativeLaws
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.PairEquivalence
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.PublicFrame
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.Readback
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.ScalarPlan
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.Scope
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.SourcePair
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.SourceSafety
import bench.tritonbench_g.layer_norm_ops.FPCorrect
import bench.tritonbench_g.layer_norm_ops.LayerNormOps
import bench.tritonbench_g.layer_norm_ops.Proofs.FPClosed
import bench.tritonbench_g.layer_norm_ops.Proofs.FPClosedWitness
import bench.tritonbench_g.layer_norm_ops.Proofs.FPComplete
import bench.tritonbench_g.layer_norm_ops.Proofs.FPContract
import bench.tritonbench_g.layer_norm_ops.Proofs.FPModel
import bench.tritonbench_g.layer_norm_ops.Proofs.FPShape
import bench.tritonbench_g.layer_norm_ops.Proofs.FPSoftware
import bench.tritonbench_g.layer_norm_ops.Proofs.FPSupplied
import bench.tritonbench_g.layer_norm_ops.Proofs.FPTypedSource
import bench.tritonbench_g.layer_norm_ops.Proofs.FPWitness
import bench.tritonbench_g.layer_norm_ops.SoftwareCorrect

open Lean Elab Command
run_cmd do
  let env ← getEnv
  let modules : Array Name := #[`VeriTile.Meta.StatementAudit,`bench.repairs.tritonbench_g.layer_norm_ops.Contract,`bench.repairs.tritonbench_g.layer_norm_ops.Correct,`bench.repairs.tritonbench_g.layer_norm_ops.FPEquiv,`bench.repairs.tritonbench_g.layer_norm_ops.IO,`bench.repairs.tritonbench_g.layer_norm_ops.Kernels,`bench.repairs.tritonbench_g.layer_norm_ops.Misc.NativeCorrect,`bench.repairs.tritonbench_g.layer_norm_ops.Misc.TensorBridge,`bench.repairs.tritonbench_g.layer_norm_ops.PairIO,`bench.repairs.tritonbench_g.layer_norm_ops.Proofs.Allocation,`bench.repairs.tritonbench_g.layer_norm_ops.Proofs.BackwardBody000,`bench.repairs.tritonbench_g.layer_norm_ops.Proofs.BackwardBody001,`bench.repairs.tritonbench_g.layer_norm_ops.Proofs.BackwardBody010,`bench.repairs.tritonbench_g.layer_norm_ops.Proofs.BackwardBody011,`bench.repairs.tritonbench_g.layer_norm_ops.Proofs.BackwardBody100,`bench.repairs.tritonbench_g.layer_norm_ops.Proofs.BackwardBody101,`bench.repairs.tritonbench_g.layer_norm_ops.Proofs.BackwardBody110,`bench.repairs.tritonbench_g.layer_norm_ops.Proofs.BackwardBody111,`bench.repairs.tritonbench_g.layer_norm_ops.Proofs.BackwardIO,`bench.repairs.tritonbench_g.layer_norm_ops.Proofs.BackwardInvariant,`bench.repairs.tritonbench_g.layer_norm_ops.Proofs.BackwardModel,`bench.repairs.tritonbench_g.layer_norm_ops.Proofs.BackwardPrelude00,`bench.repairs.tritonbench_g.layer_norm_ops.Proofs.BackwardPrelude01,`bench.repairs.tritonbench_g.layer_norm_ops.Proofs.BackwardPrelude10,`bench.repairs.tritonbench_g.layer_norm_ops.Proofs.BackwardPrelude11,`bench.repairs.tritonbench_g.layer_norm_ops.Proofs.BackwardRun,`bench.repairs.tritonbench_g.layer_norm_ops.Proofs.BackwardValues,`bench.repairs.tritonbench_g.layer_norm_ops.Proofs.BackwardWhole,`bench.repairs.tritonbench_g.layer_norm_ops.Proofs.CheckedHost,`bench.repairs.tritonbench_g.layer_norm_ops.Proofs.Contract,`bench.repairs.tritonbench_g.layer_norm_ops.Proofs.FiniteReference,`bench.repairs.tritonbench_g.layer_norm_ops.Proofs.ForwardIO,`bench.repairs.tritonbench_g.layer_norm_ops.Proofs.ForwardModel,`bench.repairs.tritonbench_g.layer_norm_ops.Proofs.ForwardRun,`bench.repairs.tritonbench_g.layer_norm_ops.Proofs.ForwardRun00,`bench.repairs.tritonbench_g.layer_norm_ops.Proofs.ForwardRun01,`bench.repairs.tritonbench_g.layer_norm_ops.Proofs.ForwardRun10,`bench.repairs.tritonbench_g.layer_norm_ops.Proofs.ForwardRun11,`bench.repairs.tritonbench_g.layer_norm_ops.Proofs.ForwardSafety,`bench.repairs.tritonbench_g.layer_norm_ops.Proofs.ForwardSafety00,`bench.repairs.tritonbench_g.layer_norm_ops.Proofs.ForwardSafety01,`bench.repairs.tritonbench_g.layer_norm_ops.Proofs.ForwardSafety10,`bench.repairs.tritonbench_g.layer_norm_ops.Proofs.ForwardSafety11,`bench.repairs.tritonbench_g.layer_norm_ops.Proofs.Frame,`bench.repairs.tritonbench_g.layer_norm_ops.Proofs.Host,`bench.repairs.tritonbench_g.layer_norm_ops.Proofs.HostCorrect,`bench.repairs.tritonbench_g.layer_norm_ops.Proofs.HostFPEquiv,`bench.repairs.tritonbench_g.layer_norm_ops.Proofs.HostResults,`bench.repairs.tritonbench_g.layer_norm_ops.Proofs.IONative,`bench.repairs.tritonbench_g.layer_norm_ops.Proofs.IOReal,`bench.repairs.tritonbench_g.layer_norm_ops.Proofs.IOResults,`bench.repairs.tritonbench_g.layer_norm_ops.Proofs.IOSafety,`bench.repairs.tritonbench_g.layer_norm_ops.Proofs.KernelBound,`bench.repairs.tritonbench_g.layer_norm_ops.Proofs.KernelCompiler,`bench.repairs.tritonbench_g.layer_norm_ops.Proofs.KernelCoverage,`bench.repairs.tritonbench_g.layer_norm_ops.Proofs.KernelDispatch,`bench.repairs.tritonbench_g.layer_norm_ops.Proofs.KernelExecution,`bench.repairs.tritonbench_g.layer_norm_ops.Proofs.KernelFrame,`bench.repairs.tritonbench_g.layer_norm_ops.Proofs.KernelIndex,`bench.repairs.tritonbench_g.layer_norm_ops.Proofs.KernelInputs,`bench.repairs.tritonbench_g.layer_norm_ops.Proofs.KernelInvariant,`bench.repairs.tritonbench_g.layer_norm_ops.Proofs.KernelNative,`bench.repairs.tritonbench_g.layer_norm_ops.Proofs.KernelReadback,`bench.repairs.tritonbench_g.layer_norm_ops.Proofs.KernelReal,`bench.repairs.tritonbench_g.layer_norm_ops.Proofs.KernelReference,`bench.repairs.tritonbench_g.layer_norm_ops.Proofs.Logical,`bench.repairs.tritonbench_g.layer_norm_ops.Proofs.LogicalBackward,`bench.repairs.tritonbench_g.layer_norm_ops.Proofs.LogicalHost,`bench.repairs.tritonbench_g.layer_norm_ops.Proofs.LogicalIO,`bench.repairs.tritonbench_g.layer_norm_ops.Proofs.LogicalLoop,`bench.repairs.tritonbench_g.layer_norm_ops.Proofs.LogicalResult,`bench.repairs.tritonbench_g.layer_norm_ops.Proofs.MathBackwardLoop,`bench.repairs.tritonbench_g.layer_norm_ops.Proofs.MathForward,`bench.repairs.tritonbench_g.layer_norm_ops.Proofs.MathGrid,`bench.repairs.tritonbench_g.layer_norm_ops.Proofs.MathModel,`bench.repairs.tritonbench_g.layer_norm_ops.Proofs.MathPartials,`bench.repairs.tritonbench_g.layer_norm_ops.Proofs.MathRecomputed,`bench.repairs.tritonbench_g.layer_norm_ops.Proofs.MathReturned,`bench.repairs.tritonbench_g.layer_norm_ops.Proofs.MathSaved,`bench.repairs.tritonbench_g.layer_norm_ops.Proofs.MathValues,`bench.repairs.tritonbench_g.layer_norm_ops.Proofs.Mathematical,`bench.repairs.tritonbench_g.layer_norm_ops.Proofs.MemoryHelpers,`bench.repairs.tritonbench_g.layer_norm_ops.Proofs.Native,`bench.repairs.tritonbench_g.layer_norm_ops.Proofs.NativeLaws,`bench.repairs.tritonbench_g.layer_norm_ops.Proofs.PairEquivalence,`bench.repairs.tritonbench_g.layer_norm_ops.Proofs.PublicFrame,`bench.repairs.tritonbench_g.layer_norm_ops.Proofs.Readback,`bench.repairs.tritonbench_g.layer_norm_ops.Proofs.ScalarPlan,`bench.repairs.tritonbench_g.layer_norm_ops.Proofs.Scope,`bench.repairs.tritonbench_g.layer_norm_ops.Proofs.SourcePair,`bench.repairs.tritonbench_g.layer_norm_ops.Proofs.SourceSafety,`bench.tests.ConstexprMaskBody,`bench.tests.OptionalPointerBody,`bench.tests.OptionalPointerExpressionBody,`bench.tritonbench_g.layer_norm_ops.FPCorrect,`bench.tritonbench_g.layer_norm_ops.LayerNormOps,`bench.tritonbench_g.layer_norm_ops.Proofs.FPClosed,`bench.tritonbench_g.layer_norm_ops.Proofs.FPClosedWitness,`bench.tritonbench_g.layer_norm_ops.Proofs.FPComplete,`bench.tritonbench_g.layer_norm_ops.Proofs.FPContract,`bench.tritonbench_g.layer_norm_ops.Proofs.FPModel,`bench.tritonbench_g.layer_norm_ops.Proofs.FPShape,`bench.tritonbench_g.layer_norm_ops.Proofs.FPSoftware,`bench.tritonbench_g.layer_norm_ops.Proofs.FPSupplied,`bench.tritonbench_g.layer_norm_ops.Proofs.FPTypedSource,`bench.tritonbench_g.layer_norm_ops.Proofs.FPWitness,`bench.tritonbench_g.layer_norm_ops.SoftwareCorrect]
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
#print_fp_assumptions VeriTile.Bench.Repairs.LayerNormOps.FPEquiv.forward_mask_reused_equiv
