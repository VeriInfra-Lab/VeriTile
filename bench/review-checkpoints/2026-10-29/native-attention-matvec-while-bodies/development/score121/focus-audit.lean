import VeriTile.Meta.StatementAudit
import VeriTile.Triton.DSL.Expansion.Common
import VeriTile.Triton.DSL.TypedStores
import VeriTile.Triton.Float.IntegerErasure
import VeriTile.Triton.Memory.AccessHelpers
import VeriTile.Triton.Memory.RegisterSafety
import VeriTile.Triton.Memory.SignedRangeTrace
import bench.repairs.tritonbench_g.attention_score_tile_checked.Contract
import bench.repairs.tritonbench_g.attention_score_tile_checked.Correct
import bench.repairs.tritonbench_g.attention_score_tile_checked.FPEquiv
import bench.repairs.tritonbench_g.attention_score_tile_checked.IO
import bench.repairs.tritonbench_g.attention_score_tile_checked.Kernels
import bench.repairs.tritonbench_g.attention_score_tile_checked.Misc.NativeBinding
import bench.repairs.tritonbench_g.attention_score_tile_checked.Misc.NativeCorrect
import bench.repairs.tritonbench_g.attention_score_tile_checked.Misc.NativeIO
import bench.repairs.tritonbench_g.attention_score_tile_checked.Misc.NativeIOProof
import bench.repairs.tritonbench_g.attention_score_tile_checked.Misc.NativeIOSafety
import bench.repairs.tritonbench_g.attention_score_tile_checked.Misc.NativeIOValues
import bench.repairs.tritonbench_g.attention_score_tile_checked.Misc.NativeReference
import bench.repairs.tritonbench_g.attention_score_tile_checked.Misc.PhysicalIOBinding
import bench.repairs.tritonbench_g.attention_score_tile_checked.Misc.RawCorrect
import bench.repairs.tritonbench_g.attention_score_tile_checked.Misc.RawNative
import bench.repairs.tritonbench_g.attention_score_tile_checked.Misc.TensorBridge
import bench.repairs.tritonbench_g.attention_score_tile_checked.PairIO
import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.IndexModel
import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.LogicalReader
import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Memory
import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.NativeSafetyLoop
import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.NativeSafetySteps
import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.NativeSourceSafety
import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.RealIOProof
import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.Contract
import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.CurrentExecution
import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.CurrentMachine
import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.CurrentRepresentation
import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.DenseSafety
import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.DenseSource
import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.FPBody
import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.FPContext
import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.FPEngine
import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.FPEntry
import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.FPGrid
import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.FPLoop
import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.FPOutput
import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.FPReturned
import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.FPStorage
import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.FPValues
import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.Geometry
import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.IndexProjection
import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.Kernels
import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.LogicalFormula
import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.NativeColumn
import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.NativeFormula
import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.NativeIdeal
import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.NativeInstructions
import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.NativeLoop
import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.NativeMath
import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.NativePhysical
import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.NativeReduction
import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.NativeReturnedMath
import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.NativeStaging
import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.Physical
import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.RealArithmetic
import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.RealArithmeticOutput
import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.RealBody
import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.RealExecution
import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.RealExpressions
import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.RealFormula
import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.RealFrame
import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.RealIteration
import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.RealLoads
import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.RealLoop
import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.RealOutput
import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.RealPhysical
import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.RealPrelude
import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.RealProgram
import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.RealReturned
import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.RealSafety
import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.RealSafetyLoop
import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.RealSafetyOutput
import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.RealSafetyParts
import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.RealSafetyPrelude
import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.RealSource
import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.RealTiles
import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.SourceDecomposition
import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.WholeGrid
import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.Widened
import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.SourceBridge
import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.SourcePair
import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.SourceTiles
import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Tensor
import bench.repairs.tritonbench_g.attention_score_tile_checked.RealIO
import bench.repairs.tritonbench_g.attention_score_tile_checked.SourceDefinitions
import bench.repairs.tritonbench_g.attention_score_tile_checked.SourceParameters
import bench.tests.CallRandomABI
import bench.tests.CallSignature
import bench.tests.FP32ArgumentCopyBody
import bench.tests.TypedDefaultDotLoopBody
import bench.tests.TypedFP32CallArithmetic

open Lean Elab Command
run_cmd do
  let env ← getEnv
  let modules : Array Name := #[`VeriTile.Meta.StatementAudit,`VeriTile.Triton.DSL.Expansion.Common,`VeriTile.Triton.DSL.TypedStores,`VeriTile.Triton.Float.IntegerErasure,`VeriTile.Triton.Memory.AccessHelpers,`VeriTile.Triton.Memory.RegisterSafety,`VeriTile.Triton.Memory.SignedRangeTrace,`bench.repairs.tritonbench_g.attention_score_tile_checked.Contract,`bench.repairs.tritonbench_g.attention_score_tile_checked.Correct,`bench.repairs.tritonbench_g.attention_score_tile_checked.FPEquiv,`bench.repairs.tritonbench_g.attention_score_tile_checked.IO,`bench.repairs.tritonbench_g.attention_score_tile_checked.Kernels,`bench.repairs.tritonbench_g.attention_score_tile_checked.Misc.NativeBinding,`bench.repairs.tritonbench_g.attention_score_tile_checked.Misc.NativeCorrect,`bench.repairs.tritonbench_g.attention_score_tile_checked.Misc.NativeIO,`bench.repairs.tritonbench_g.attention_score_tile_checked.Misc.NativeIOProof,`bench.repairs.tritonbench_g.attention_score_tile_checked.Misc.NativeIOSafety,`bench.repairs.tritonbench_g.attention_score_tile_checked.Misc.NativeIOValues,`bench.repairs.tritonbench_g.attention_score_tile_checked.Misc.NativeReference,`bench.repairs.tritonbench_g.attention_score_tile_checked.Misc.PhysicalIOBinding,`bench.repairs.tritonbench_g.attention_score_tile_checked.Misc.RawCorrect,`bench.repairs.tritonbench_g.attention_score_tile_checked.Misc.RawNative,`bench.repairs.tritonbench_g.attention_score_tile_checked.Misc.TensorBridge,`bench.repairs.tritonbench_g.attention_score_tile_checked.PairIO,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.IndexModel,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.LogicalReader,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Memory,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.NativeSafetyLoop,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.NativeSafetySteps,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.NativeSourceSafety,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.RealIOProof,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.Contract,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.CurrentExecution,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.CurrentMachine,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.CurrentRepresentation,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.DenseSafety,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.DenseSource,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.FPBody,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.FPContext,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.FPEngine,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.FPEntry,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.FPGrid,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.FPLoop,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.FPOutput,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.FPReturned,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.FPStorage,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.FPValues,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.Geometry,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.IndexProjection,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.Kernels,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.LogicalFormula,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.NativeColumn,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.NativeFormula,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.NativeIdeal,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.NativeInstructions,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.NativeLoop,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.NativeMath,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.NativePhysical,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.NativeReduction,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.NativeReturnedMath,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.NativeStaging,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.Physical,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.RealArithmetic,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.RealArithmeticOutput,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.RealBody,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.RealExecution,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.RealExpressions,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.RealFormula,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.RealFrame,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.RealIteration,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.RealLoads,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.RealLoop,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.RealOutput,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.RealPhysical,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.RealPrelude,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.RealProgram,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.RealReturned,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.RealSafety,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.RealSafetyLoop,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.RealSafetyOutput,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.RealSafetyParts,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.RealSafetyPrelude,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.RealSource,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.RealTiles,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.SourceDecomposition,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.WholeGrid,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.Widened,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.SourceBridge,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.SourcePair,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.SourceTiles,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Tensor,`bench.repairs.tritonbench_g.attention_score_tile_checked.RealIO,`bench.repairs.tritonbench_g.attention_score_tile_checked.SourceDefinitions,`bench.repairs.tritonbench_g.attention_score_tile_checked.SourceParameters,`bench.tests.CallRandomABI,`bench.tests.CallSignature,`bench.tests.FP32ArgumentCopyBody,`bench.tests.TypedDefaultDotLoopBody,`bench.tests.TypedFP32CallArithmetic]
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
#print_fp_assumptions VeriTile.Bench.Repairs.AttentionScoreTileChecked.FPEquiv.score_scaling_equiv
