import VeriTile.Triton.Core.Ast
import VeriTile.Triton.DSL.CallSignature
import VeriTile.Triton.DSL.ConstexprMask
import VeriTile.Triton.DSL.Expansion.Compute
import VeriTile.Triton.DSL.Expansion.Main
import VeriTile.Triton.DSL.Metadata
import VeriTile.Triton.DSL.OrderedRowSum
import VeriTile.Triton.DSL.Syntax
import VeriTile.Triton.DSL.TypedStores
import VeriTile.Triton.Float.Erasure
import VeriTile.Triton.Float.EvalOpR
import VeriTile.Triton.Float.Structural
import VeriTile.Triton.Float.StructuralSafety
import VeriTile.Triton.Memory.Bounds
import VeriTile.Triton.Memory.Flatten
import VeriTile.Triton.Memory.FlattenR
import VeriTile.Triton.Memory.Locality
import VeriTile.Triton.Memory.Typing.Checker
import VeriTile.Triton.Memory.Typing.Contracts
import VeriTile.Triton.Semantics.EvalOp
import bench.repairs.tritonbench_g.fused_layernorm_triton.Proofs.ReadableSafety
import bench.repairs.tritonbench_g.layer_norm_welfold.Contract
import bench.repairs.tritonbench_g.layer_norm_welfold.Correct
import bench.repairs.tritonbench_g.layer_norm_welfold.FPEquiv
import bench.repairs.tritonbench_g.layer_norm_welfold.IO
import bench.repairs.tritonbench_g.layer_norm_welfold.Kernels
import bench.repairs.tritonbench_g.layer_norm_welfold.Misc.Audit
import bench.repairs.tritonbench_g.layer_norm_welfold.Misc.NativeCorrect
import bench.repairs.tritonbench_g.layer_norm_welfold.Misc.PublicAudit
import bench.repairs.tritonbench_g.layer_norm_welfold.Misc.TensorBridge
import bench.repairs.tritonbench_g.layer_norm_welfold.Misc.Wrapper
import bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.Accumulate
import bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.Bounds
import bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.Contract
import bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.GraphExecution
import bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.Grid
import bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.Host
import bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.IOImplements
import bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.IOResults
import bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.InputBinding
import bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.KernelBoundary
import bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.KernelGraph
import bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.KernelGraphResult
import bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.KernelNativeResult
import bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.KernelRealResult
import bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.LegacySource
import bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.LogicalIO
import bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.LoweringPlan
import bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.Mathematics
import bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.NativeABI
import bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.NativeHost
import bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.NativeInputs
import bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.NativeKernel
import bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.NativeTree
import bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.PairEquivalence
import bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.Physical
import bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.RawAccumulate
import bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.RawGrid
import bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.RawNormalize
import bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.RawRow
import bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.RawSafety
import bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.RawStatistics
import bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.RawSyntax
import bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.ReadableAccumulate
import bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.ReadableBoundary
import bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.ReadableGrid
import bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.ReadableNormalize
import bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.ReadableResults
import bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.ReadableRow
import bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.ReadableSafety
import bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.ReadableStatistics
import bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.ReadableSyntax
import bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.Real
import bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.Row
import bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.ScalarTransport
import bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.Source
import bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.SourceBinding
import bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.SourceGraph
import bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.SourceSafety
import bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.Statistics
import bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.Trials
import bench.tests.AtomicStoreDTypeBody
import bench.tests.BarrierLoopBody
import bench.tests.BranchStoreDType
import bench.tests.CallSignature
import bench.tests.ConstexprMaskBody
import bench.tests.DTypeResolution
import bench.tests.FP32CallValue
import bench.tests.LibdeviceRsqrtBody
import bench.tests.LoopBody
import bench.tests.LoopTupleLoadBody
import bench.tests.NativeMatrixLoop
import bench.tests.NativeMaxLoop
import bench.tests.OrderedRowSumBody
import bench.tests.OrderedRowSumRejected
import bench.tests.PositionalReductionLoop
import bench.tests.SignedRangeCallBody
import bench.tests.SourceDTypeBody
import bench.tests.TypedFP32CallArithmetic
import bench.tests.TypedStores
import bench.tests.ValueDTypeBody
import bench.tritonbench_g.fused_layernorm_triton.Proofs.WholeReal

open Lean Elab Command
run_cmd do
  let env ← getEnv
  let modules : Array Name := #[`VeriTile.Meta.StatementAudit,`VeriTile.Triton.Core.Ast,`VeriTile.Triton.DSL.CallSignature,`VeriTile.Triton.DSL.ConstexprMask,`VeriTile.Triton.DSL.Expansion.Compute,`VeriTile.Triton.DSL.Expansion.Main,`VeriTile.Triton.DSL.Metadata,`VeriTile.Triton.DSL.OrderedRowSum,`VeriTile.Triton.DSL.Syntax,`VeriTile.Triton.DSL.TypedStores,`VeriTile.Triton.Float.Erasure,`VeriTile.Triton.Float.EvalOpR,`VeriTile.Triton.Float.Structural,`VeriTile.Triton.Float.StructuralSafety,`VeriTile.Triton.Memory.Bounds,`VeriTile.Triton.Memory.Flatten,`VeriTile.Triton.Memory.FlattenR,`VeriTile.Triton.Memory.Locality,`VeriTile.Triton.Memory.Typing.Checker,`VeriTile.Triton.Memory.Typing.Contracts,`VeriTile.Triton.Semantics.EvalOp,`bench.repairs.tritonbench_g.fused_layernorm_triton.Proofs.ReadableSafety,`bench.repairs.tritonbench_g.layer_norm_welfold.Contract,`bench.repairs.tritonbench_g.layer_norm_welfold.Correct,`bench.repairs.tritonbench_g.layer_norm_welfold.FPEquiv,`bench.repairs.tritonbench_g.layer_norm_welfold.IO,`bench.repairs.tritonbench_g.layer_norm_welfold.Kernels,`bench.repairs.tritonbench_g.layer_norm_welfold.Misc.Audit,`bench.repairs.tritonbench_g.layer_norm_welfold.Misc.NativeCorrect,`bench.repairs.tritonbench_g.layer_norm_welfold.Misc.PublicAudit,`bench.repairs.tritonbench_g.layer_norm_welfold.Misc.TensorBridge,`bench.repairs.tritonbench_g.layer_norm_welfold.Misc.Wrapper,`bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.Accumulate,`bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.Bounds,`bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.Contract,`bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.GraphExecution,`bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.Grid,`bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.Host,`bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.IOImplements,`bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.IOResults,`bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.InputBinding,`bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.KernelBoundary,`bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.KernelGraph,`bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.KernelGraphResult,`bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.KernelNativeResult,`bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.KernelRealResult,`bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.LegacySource,`bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.LogicalIO,`bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.LoweringPlan,`bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.Mathematics,`bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.NativeABI,`bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.NativeHost,`bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.NativeInputs,`bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.NativeKernel,`bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.NativeTree,`bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.PairEquivalence,`bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.Physical,`bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.RawAccumulate,`bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.RawGrid,`bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.RawNormalize,`bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.RawRow,`bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.RawSafety,`bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.RawStatistics,`bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.RawSyntax,`bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.ReadableAccumulate,`bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.ReadableBoundary,`bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.ReadableGrid,`bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.ReadableNormalize,`bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.ReadableResults,`bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.ReadableRow,`bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.ReadableSafety,`bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.ReadableStatistics,`bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.ReadableSyntax,`bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.Real,`bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.Row,`bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.ScalarTransport,`bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.Source,`bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.SourceBinding,`bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.SourceGraph,`bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.SourceSafety,`bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.Statistics,`bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.Trials,`bench.tests.AtomicStoreDTypeBody,`bench.tests.BarrierLoopBody,`bench.tests.BranchStoreDType,`bench.tests.CallSignature,`bench.tests.ConstexprMaskBody,`bench.tests.DTypeResolution,`bench.tests.FP32CallValue,`bench.tests.LibdeviceRsqrtBody,`bench.tests.LoopBody,`bench.tests.LoopTupleLoadBody,`bench.tests.NativeMatrixLoop,`bench.tests.NativeMaxLoop,`bench.tests.OrderedRowSumBody,`bench.tests.OrderedRowSumRejected,`bench.tests.PositionalReductionLoop,`bench.tests.SignedRangeCallBody,`bench.tests.SourceDTypeBody,`bench.tests.TypedFP32CallArithmetic,`bench.tests.TypedStores,`bench.tests.ValueDTypeBody,`bench.tritonbench_g.fused_layernorm_triton.Proofs.WholeReal]
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
#print_fp_assumptions VeriTile.Bench.Repairs.LayerNormWelfold.FPEquiv.row_offset_reused_equiv
