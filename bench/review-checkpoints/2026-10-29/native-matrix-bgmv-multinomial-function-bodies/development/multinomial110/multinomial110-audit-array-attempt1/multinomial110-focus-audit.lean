import VeriTile.Meta.StatementAudit
import bench.repairs.tritonbench_g.multinomial_sampling.Contract
import bench.repairs.tritonbench_g.multinomial_sampling.Correct
import bench.repairs.tritonbench_g.multinomial_sampling.FPEquiv
import bench.repairs.tritonbench_g.multinomial_sampling.IO
import bench.repairs.tritonbench_g.multinomial_sampling.Kernels
import bench.repairs.tritonbench_g.multinomial_sampling.Misc.CorrectPublic
import bench.repairs.tritonbench_g.multinomial_sampling.Misc.FPEquivPublic
import bench.repairs.tritonbench_g.multinomial_sampling.Misc.HostCorrect
import bench.repairs.tritonbench_g.multinomial_sampling.Misc.HostFPEquiv
import bench.repairs.tritonbench_g.multinomial_sampling.Misc.NativeCorrect
import bench.repairs.tritonbench_g.multinomial_sampling.Misc.TensorBridge
import bench.repairs.tritonbench_g.multinomial_sampling.PairIO
import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.Bits
import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.Boundary
import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.CDFBridge
import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.Contract
import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.Core.FPBackend
import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.Core.FPBody
import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.Core.FPComplete
import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.Core.FPHost
import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.Core.FPInvariant
import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.Core.FPPrologue
import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.Core.FPReference
import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.Core.FPSoftware
import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.Core.FPSource
import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.Core.FPStorage
import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.Core.FPWrapper
import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.FPCompatibility
import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.Guarded
import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.Host
import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.IOComplete
import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.IOFacts
import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.IORealResult
import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.IOResult
import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.IOSafety
import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.IOUndef
import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.KernelCompatibility
import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.KernelMemory
import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.KernelSource
import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.LogicalReader
import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.LogicalTransport
import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.Model
import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.Native
import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.NumericTrees
import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.Numerics
import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.Physical
import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.ReaderBefore
import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.Real
import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.SourceSafety
import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.Spec
import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.TensorInputs
import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.TensorReference
import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.Unrolled
import bench.repairs.tritonbench_g.multinomial_sampling.Proofs.UnrolledPair

open Lean Elab Command
run_cmd do
  let env ← getEnv
  let modules : Array Name := #[
    `VeriTile.Meta.StatementAudit,
    `VeriTile.Triton.DSL.CallSignature,
    `VeriTile.Triton.DSL.Expansion.Main,
    `VeriTile.Triton.DSL.Syntax,
    `VeriTile.Triton.DSL.TypedStores,
    `bench.repairs.tritonbench_g.triton_matmul.Contract,
    `bench.repairs.tritonbench_g.triton_matmul.Correct,
    `bench.repairs.tritonbench_g.triton_matmul.FPEquiv,
    `bench.repairs.tritonbench_g.triton_matmul.KernelContract,
    `bench.repairs.tritonbench_g.triton_matmul.KernelOnly.CandidateKernels,
    `bench.repairs.tritonbench_g.triton_matmul.KernelOnly.Contract,
    `bench.repairs.tritonbench_g.triton_matmul.KernelOnly.Correct,
    `bench.repairs.tritonbench_g.triton_matmul.KernelOnly.FPEquiv,
    `bench.repairs.tritonbench_g.triton_matmul.KernelOnly.IO,
    `bench.repairs.tritonbench_g.triton_matmul.KernelOnly.Kernels,
    `bench.repairs.tritonbench_g.triton_matmul.KernelOnly.Misc.NativeCorrect,
    `bench.repairs.tritonbench_g.triton_matmul.KernelOnly.Misc.Operational,
    `bench.repairs.tritonbench_g.triton_matmul.KernelOnly.Misc.TensorBridge,
    `bench.repairs.tritonbench_g.triton_matmul.KernelOnly.PairIO,
    `bench.repairs.tritonbench_g.triton_matmul.KernelOnly.Proofs.LogicalIO,
    `bench.repairs.tritonbench_g.triton_matmul.KernelOnly.Proofs.Pair,
    `bench.repairs.tritonbench_g.triton_matmul.KernelOnly.Proofs.PublicIO,
    `bench.repairs.tritonbench_g.triton_matmul.KernelOnly.Proofs.SourcePair,
    `bench.repairs.tritonbench_g.triton_matmul.KernelOnly.Proofs.SourceSafety,
    `bench.repairs.tritonbench_g.triton_matmul.KernelOnly.Raw,
    `bench.repairs.tritonbench_g.triton_matmul.Kernels,
    `bench.repairs.tritonbench_g.triton_matmul.Misc.KernelCorrect,
    `bench.repairs.tritonbench_g.triton_matmul.Misc.KernelFPEquiv,
    `bench.repairs.tritonbench_g.triton_matmul.Misc.KernelWhole,
    `bench.repairs.tritonbench_g.triton_matmul.Proofs.Addresses,
    `bench.repairs.tritonbench_g.triton_matmul.Proofs.ByteExecution,
    `bench.repairs.tritonbench_g.triton_matmul.Proofs.ByteHost,
    `bench.repairs.tritonbench_g.triton_matmul.Proofs.CheckedHost,
    `bench.repairs.tritonbench_g.triton_matmul.Proofs.CheckedMath,
    `bench.repairs.tritonbench_g.triton_matmul.Proofs.Contract,
    `bench.repairs.tritonbench_g.triton_matmul.Proofs.Execution,
    `bench.repairs.tritonbench_g.triton_matmul.Proofs.FP,
    `bench.repairs.tritonbench_g.triton_matmul.Proofs.FiniteExecution,
    `bench.repairs.tritonbench_g.triton_matmul.Proofs.FiniteGrid,
    `bench.repairs.tritonbench_g.triton_matmul.Proofs.FiniteHost,
    `bench.repairs.tritonbench_g.triton_matmul.Proofs.Geometry,
    `bench.repairs.tritonbench_g.triton_matmul.Proofs.HistoricalCorrect,
    `bench.repairs.tritonbench_g.triton_matmul.Proofs.HistoricalFP,
    `bench.repairs.tritonbench_g.triton_matmul.Proofs.Host,
    `bench.repairs.tritonbench_g.triton_matmul.Proofs.IdealHost,
    `bench.repairs.tritonbench_g.triton_matmul.Proofs.IdealTensor,
    `bench.repairs.tritonbench_g.triton_matmul.Proofs.Index,
    `bench.repairs.tritonbench_g.triton_matmul.Proofs.InputAdapter,
    `bench.repairs.tritonbench_g.triton_matmul.Proofs.KernelExecution,
    `bench.repairs.tritonbench_g.triton_matmul.Proofs.KernelNative,
    `bench.repairs.tritonbench_g.triton_matmul.Proofs.KernelRealInputs,
    `bench.repairs.tritonbench_g.triton_matmul.Proofs.KernelRealModel,
    `bench.repairs.tritonbench_g.triton_matmul.Proofs.KernelSource,
    `bench.repairs.tritonbench_g.triton_matmul.Proofs.KernelTensor,
    `bench.repairs.tritonbench_g.triton_matmul.Proofs.Model,
    `bench.repairs.tritonbench_g.triton_matmul.Proofs.Native,
    `bench.repairs.tritonbench_g.triton_matmul.Proofs.OutputBytes,
    `bench.repairs.tritonbench_g.triton_matmul.Proofs.RawAccess,
    `bench.repairs.tritonbench_g.triton_matmul.Proofs.RawArithmetic,
    `bench.repairs.tritonbench_g.triton_matmul.Proofs.RawBody,
    `bench.repairs.tritonbench_g.triton_matmul.Proofs.RawBodyContext,
    `bench.repairs.tritonbench_g.triton_matmul.Proofs.RawControl,
    `bench.repairs.tritonbench_g.triton_matmul.Proofs.RawCoordinates,
    `bench.repairs.tritonbench_g.triton_matmul.Proofs.RawDecomposition,
    `bench.repairs.tritonbench_g.triton_matmul.Proofs.RawEpilogue,
    `bench.repairs.tritonbench_g.triton_matmul.Proofs.RawInactive,
    `bench.repairs.tritonbench_g.triton_matmul.Proofs.RawLaunch,
    `bench.repairs.tritonbench_g.triton_matmul.Proofs.RawLoop,
    `bench.repairs.tritonbench_g.triton_matmul.Proofs.RawLoopModel,
    `bench.repairs.tritonbench_g.triton_matmul.Proofs.RawMemory,
    `bench.repairs.tritonbench_g.triton_matmul.Proofs.RawModel,
    `bench.repairs.tritonbench_g.triton_matmul.Proofs.RawPrelude,
    `bench.repairs.tritonbench_g.triton_matmul.Proofs.RawPreludeContext,
    `bench.repairs.tritonbench_g.triton_matmul.Proofs.RawSafety,
    `bench.repairs.tritonbench_g.triton_matmul.Proofs.RawSafetyLoop,
    `bench.repairs.tritonbench_g.triton_matmul.Proofs.RawWritten,
    `bench.repairs.tritonbench_g.triton_matmul.Proofs.Real,
    `bench.repairs.tritonbench_g.triton_matmul.Proofs.SourceArithmetic,
    `bench.repairs.tritonbench_g.triton_matmul.Proofs.SourceDecomposition,
    `bench.repairs.tritonbench_g.triton_matmul.Proofs.TensorBinding,
    `bench.repairs.tritonbench_g.triton_matmul.Proofs.Whole,
    `bench.repairs.tritonbench_g.triton_matmul.Proofs.Wrapper,
    `bench.tests.AlgorithmProgramIO,
    `bench.tests.AlgorithmScheduledIO,
    `bench.tests.AliasedNestedLoadBody,
    `bench.tests.AttributeCastChain,
    `bench.tests.BackendMemory,
    `bench.tests.BinaryImportBody,
    `bench.tests.BlockPointerDType,
    `bench.tests.BoolInt64Body,
    `bench.tests.BoolMaskMultiply,
    `bench.tests.BranchDtypeRegression,
    `bench.tests.BranchStoreDType,
    `bench.tests.CallRandomABI,
    `bench.tests.CallSignature,
    `bench.tests.CheckedArangeCallBody,
    `bench.tests.ConcurrentEventIO,
    `bench.tests.ConditionalCallRebindingBody,
    `bench.tests.ConditionalCast,
    `bench.tests.ConstexprBody,
    `bench.tests.ConstexprPower,
    `bench.tests.ConstexprTypedContinuation,
    `bench.tests.DTypePredicate,
    `bench.tests.DecayCumsumReverse,
    `bench.tests.DivRNSource,
    `bench.tests.EmbeddingTritonExecution,
    `bench.tests.EquivTactics,
    `bench.tests.FP32CallValue,
    `bench.tests.FP4Source,
    `bench.tests.FP8Bitcast,
    `bench.tests.FPAddressedInputsIO,
    `bench.tests.FPArrayProgramIO,
    `bench.tests.FPBranchRewrite,
    `bench.tests.FPComputeAddition,
    `bench.tests.FPConstantBranchContexts,
    `bench.tests.FPContextAttnFwdSource,
    `bench.tests.FPContextAttnLlamaSource,
    `bench.tests.FPControl,
    `bench.tests.FPDebugBarrier,
    `bench.tests.FPDecayCumsumSource,
    `bench.tests.FPDotRebinding,
    `bench.tests.FPDualOutput,
    `bench.tests.FPEmbeddingTritonSource,
    `bench.tests.FPExpandDimsPrecision,
    `bench.tests.FPFmaSource,
    `bench.tests.FPGuardedExamples,
    `bench.tests.FPInputBindingIO,
    `bench.tests.FPIntRangeRewrite,
    `bench.tests.FPIntegerConversionNeg,
    `bench.tests.FPIntegerDotIO,
    `bench.tests.FPIntegerOutputIO,
    `bench.tests.FPIntegerStageIO,
    `bench.tests.FPInterleaveBitcast,
    `bench.tests.FPMaterializedProgramIO,
    `bench.tests.FPMaximumPrimitive,
    `bench.tests.FPMinimumReduction,
    `bench.tests.FPMultinomialSource,
    `bench.tests.FPMultinomialUndef,
    `bench.tests.FPNarrowScope,
    `bench.tests.FPObservedInputCellsIO,
    `bench.tests.FPObservedProgramEquivalence,
    `bench.tests.FPObservedProgramIO,
    `bench.tests.FPOrdinaryProgramIO,
    `bench.tests.FPPerAddressProgramIO,
    `bench.tests.FPProgramIO,
    `bench.tests.FPRangeRewrite,
    `bench.tests.FPReductionProgramIO,
    `bench.tests.FPRelationalProgramIO,
    `bench.tests.FPSoftmaxForwardPointer,
    `bench.tests.FPSoftmaxForwardPointerReject,
    `bench.tests.FPStaticRangeRewrite,
    `bench.tests.FPStrictMeta,
    `bench.tests.FPStrictRuleBoundaries,
    `bench.tests.FPStructural,
    `bench.tests.FPStructuralSafety,
    `bench.tests.FPUndefProgramIO,
    `bench.tests.FPWhereContext,
    `bench.tests.FillProgramIO,
    `bench.tests.FloatInfinityLiteral,
    `bench.tests.FloatMinMaxPromotion,
    `bench.tests.InlineSuiteBody,
    `bench.tests.IntScaledMatmulPadding,
    `bench.tests.IntegerCallFloatBranchBody,
    `bench.tests.IntegerDotSource,
    `bench.tests.LoadCastStorage,
    `bench.tests.LoopBody,
    `bench.tests.LoopTupleLoadBody,
    `bench.tests.MultinomialRoundedBoundary,
    `bench.tests.NarrowUnsignedBody,
    `bench.tests.NarrowUnsignedMemoryBody,
    `bench.tests.NativeCountLoop,
    `bench.tests.NativeMatrixLoop,
    `bench.tests.NativeMaxLoop,
    `bench.tests.NestedLoadAssignmentsBody,
    `bench.tests.NestedSignedLoadBody,
    `bench.tests.OptimizedRealExamples,
    `bench.tests.OptionalStringBody,
    `bench.tests.ParameterizedAlgorithmProgramIO,
    `bench.tests.PointerOffsetRebindingBody,
    `bench.tests.PointerOffsets,
    `bench.tests.PositionalReductionLoop,
    `bench.tests.PreparedAlgorithmProgramIO,
    `bench.tests.PythonAugmentedConditionalBody,
    `bench.tests.PythonBody,
    `bench.tests.PythonConditionalAssignmentBody,
    `bench.tests.PythonIntegerWhere,
    `bench.tests.PythonSignedConditionalBody,
    `bench.tests.RankOneBlockPtr,
    `bench.tests.RegisterDTypeLoad,
    `bench.tests.RuntimeIntegerDivision,
    `bench.tests.RuntimeIntegerScalarBody,
    `bench.tests.ScalarAssignmentBody,
    `bench.tests.SignedBitwiseSource,
    `bench.tests.SignedInteger16Cast,
    `bench.tests.SignedInteger16Contexts,
    `bench.tests.SignedIntegerCast,
    `bench.tests.SignedIntegerCastContexts,
    `bench.tests.SignedIntegerCastRNG,
    `bench.tests.SignedLoadOther,
    `bench.tests.SignedPointerMovementBody,
    `bench.tests.SignedRangeCallBody,
    `bench.tests.SignedReverseBody,
    `bench.tests.SourceConstantBody,
    `bench.tests.SourceDType,
    `bench.tests.SourceFunction,
    `bench.tests.SpecializedArgumentBody,
    `bench.tests.StatementSourceFunction,
    `bench.tests.StaticConditions,
    `bench.tests.TritonSmoke,
    `bench.tests.TupleCallBody,
    `bench.tests.TypedFMA,
    `bench.tests.TypedFP32CallArithmetic,
    `bench.tests.TypedFP32Mul,
    `bench.tests.TypedStores,
    `bench.tests.ValueDTypeBody,
    `bench.tritonbench_g.triton_matmul.Proofs.LogicalViewDefect
  ]
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
#print_fp_assumptions VeriTile.Bench.Repairs.MultinomialSampling.FPEquiv.static_unroll_equiv
