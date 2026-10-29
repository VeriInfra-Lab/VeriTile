import VeriTile.Meta.StatementAudit
import bench.repairs.tritonbench_g.matrix_vector_multip.CandidateIO
import bench.repairs.tritonbench_g.matrix_vector_multip.Contract
import bench.repairs.tritonbench_g.matrix_vector_multip.Correct
import bench.repairs.tritonbench_g.matrix_vector_multip.FPEquiv
import bench.repairs.tritonbench_g.matrix_vector_multip.IO
import bench.repairs.tritonbench_g.matrix_vector_multip.Kernels
import bench.repairs.tritonbench_g.matrix_vector_multip.Misc.NativeCorrect
import bench.repairs.tritonbench_g.matrix_vector_multip.Misc.TensorBridge
import bench.repairs.tritonbench_g.matrix_vector_multip.Misc.TileEquiv
import bench.repairs.tritonbench_g.matrix_vector_multip.Proofs.AccessSafety
import bench.repairs.tritonbench_g.matrix_vector_multip.Proofs.Contract
import bench.repairs.tritonbench_g.matrix_vector_multip.Proofs.ExplicitFMA
import bench.repairs.tritonbench_g.matrix_vector_multip.Proofs.FP
import bench.repairs.tritonbench_g.matrix_vector_multip.Proofs.GridPair
import bench.repairs.tritonbench_g.matrix_vector_multip.Proofs.HistoricalCorrect
import bench.repairs.tritonbench_g.matrix_vector_multip.Proofs.HistoricalFPEquiv
import bench.repairs.tritonbench_g.matrix_vector_multip.Proofs.Host
import bench.repairs.tritonbench_g.matrix_vector_multip.Proofs.IO
import bench.repairs.tritonbench_g.matrix_vector_multip.Proofs.Index
import bench.repairs.tritonbench_g.matrix_vector_multip.Proofs.InputBinding
import bench.repairs.tritonbench_g.matrix_vector_multip.Proofs.KernelBinding
import bench.repairs.tritonbench_g.matrix_vector_multip.Proofs.KernelFP
import bench.repairs.tritonbench_g.matrix_vector_multip.Proofs.KernelIndex
import bench.repairs.tritonbench_g.matrix_vector_multip.Proofs.KernelSource
import bench.repairs.tritonbench_g.matrix_vector_multip.Proofs.LogicalIO
import bench.repairs.tritonbench_g.matrix_vector_multip.Proofs.NativeResult
import bench.repairs.tritonbench_g.matrix_vector_multip.Proofs.Numerical
import bench.repairs.tritonbench_g.matrix_vector_multip.Proofs.Real
import bench.repairs.tritonbench_g.matrix_vector_multip.Proofs.RealResult
import bench.repairs.tritonbench_g.matrix_vector_multip.Proofs.ReturnedTensor
import bench.repairs.tritonbench_g.matrix_vector_multip.Proofs.SourceBody
import bench.repairs.tritonbench_g.matrix_vector_multip.Proofs.SourceCode
import bench.repairs.tritonbench_g.matrix_vector_multip.Proofs.SourceLoop
import bench.repairs.tritonbench_g.matrix_vector_multip.Proofs.SourceOutput
import bench.repairs.tritonbench_g.matrix_vector_multip.Proofs.SourcePair
import bench.repairs.tritonbench_g.matrix_vector_multip.Proofs.SourcePairSafety
import bench.repairs.tritonbench_g.matrix_vector_multip.Proofs.SourcePairSafetySteps
import bench.repairs.tritonbench_g.matrix_vector_multip.Proofs.SourcePairSafetyTools
import bench.repairs.tritonbench_g.matrix_vector_multip.Proofs.SourcePrelude
import bench.repairs.tritonbench_g.matrix_vector_multip.Proofs.SourceSafety
import bench.repairs.tritonbench_g.matrix_vector_multip.Proofs.TensorBinding
import bench.repairs.tritonbench_g.matrix_vector_multip.SourceDomain
import bench.repairs.tritonbench_g.matrix_vector_multip.SourceShapes
import bench.tests.WhileBody
import bench.tests.WhileExecution
import VeriTile.Triton
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
import VeriTile.Triton.Core.Ast
import VeriTile.Triton.DSL.RandomProfile
import VeriTile.Triton.DSL.TypedStorageReduction
import VeriTile.Triton.DSL.Inference
import VeriTile.Triton.DSL.CallSignature
import VeriTile.Triton.DSL.Syntax
import VeriTile.Triton.DSL.NestedLoadAssignments
import VeriTile.Triton.DSL.KnownRangeStep
import VeriTile.Triton.Source.While
import VeriTile.Triton.Semantics.Step
import VeriTile.Triton.Float.StepR
import VeriTile.Triton.Float.Erasure
import VeriTile.Triton.Float.StructuralSafety
import VeriTile.Triton.Float.Structural
import VeriTile.Triton.Memory.Flatten
import VeriTile.Triton.Memory.FlattenR
import VeriTile.Triton.Memory.Bounds
import VeriTile.Triton.Memory.Locality
import VeriTile.Triton.Control.While
import VeriTile.Triton.Memory.Typing.Contracts
import VeriTile.Triton.Memory.Typing.Checker
import VeriTile.Triton.DSL.Expansion.Compute
import VeriTile.Triton.DSL.Expansion.Main
import bench.tests.IntegerLiteralAssignmentBody
import bench.repairs.tritonbench_g.rotary_transform_tile_checked.Proofs.SourceTiles
import bench.tritonbench_g.isfinite_kernel.Proofs.TypedFlatSyntax

open Lean Elab Command
run_cmd do
  let env ← getEnv
  let modules : Array Name := #[`VeriTile.Meta.StatementAudit,`VeriTile.Triton,`VeriTile.Triton.Control.While,`VeriTile.Triton.Core.Ast,`VeriTile.Triton.DSL.CallSignature,`VeriTile.Triton.DSL.Expansion.Common,`VeriTile.Triton.DSL.Expansion.Compute,`VeriTile.Triton.DSL.Expansion.Main,`VeriTile.Triton.DSL.Inference,`VeriTile.Triton.DSL.KnownRangeStep,`VeriTile.Triton.DSL.NestedLoadAssignments,`VeriTile.Triton.DSL.RandomProfile,`VeriTile.Triton.DSL.Syntax,`VeriTile.Triton.DSL.TypedStorageReduction,`VeriTile.Triton.DSL.TypedStores,`VeriTile.Triton.Float.Erasure,`VeriTile.Triton.Float.IntegerErasure,`VeriTile.Triton.Float.StepR,`VeriTile.Triton.Float.Structural,`VeriTile.Triton.Float.StructuralSafety,`VeriTile.Triton.Memory.AccessHelpers,`VeriTile.Triton.Memory.Bounds,`VeriTile.Triton.Memory.Flatten,`VeriTile.Triton.Memory.FlattenR,`VeriTile.Triton.Memory.Locality,`VeriTile.Triton.Memory.RegisterSafety,`VeriTile.Triton.Memory.SignedRangeTrace,`VeriTile.Triton.Memory.Typing.Checker,`VeriTile.Triton.Memory.Typing.Contracts,`VeriTile.Triton.Semantics.Step,`VeriTile.Triton.Source.While,`bench.repairs.tritonbench_g.attention_score_tile_checked.Contract,`bench.repairs.tritonbench_g.attention_score_tile_checked.Correct,`bench.repairs.tritonbench_g.attention_score_tile_checked.FPEquiv,`bench.repairs.tritonbench_g.attention_score_tile_checked.IO,`bench.repairs.tritonbench_g.attention_score_tile_checked.Kernels,`bench.repairs.tritonbench_g.attention_score_tile_checked.Misc.NativeBinding,`bench.repairs.tritonbench_g.attention_score_tile_checked.Misc.NativeCorrect,`bench.repairs.tritonbench_g.attention_score_tile_checked.Misc.NativeIO,`bench.repairs.tritonbench_g.attention_score_tile_checked.Misc.NativeIOProof,`bench.repairs.tritonbench_g.attention_score_tile_checked.Misc.NativeIOSafety,`bench.repairs.tritonbench_g.attention_score_tile_checked.Misc.NativeIOValues,`bench.repairs.tritonbench_g.attention_score_tile_checked.Misc.NativeReference,`bench.repairs.tritonbench_g.attention_score_tile_checked.Misc.PhysicalIOBinding,`bench.repairs.tritonbench_g.attention_score_tile_checked.Misc.RawCorrect,`bench.repairs.tritonbench_g.attention_score_tile_checked.Misc.RawNative,`bench.repairs.tritonbench_g.attention_score_tile_checked.Misc.TensorBridge,`bench.repairs.tritonbench_g.attention_score_tile_checked.PairIO,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.IndexModel,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.LogicalReader,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Memory,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.NativeSafetyLoop,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.NativeSafetySteps,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.NativeSourceSafety,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.RealIOProof,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.Contract,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.CurrentExecution,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.CurrentMachine,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.CurrentRepresentation,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.DenseSafety,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.DenseSource,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.FPBody,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.FPContext,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.FPEngine,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.FPEntry,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.FPGrid,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.FPLoop,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.FPOutput,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.FPReturned,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.FPStorage,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.FPValues,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.Geometry,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.IndexProjection,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.Kernels,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.LogicalFormula,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.NativeColumn,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.NativeFormula,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.NativeIdeal,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.NativeInstructions,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.NativeLoop,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.NativeMath,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.NativePhysical,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.NativeReduction,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.NativeReturnedMath,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.NativeStaging,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.Physical,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.RealArithmetic,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.RealArithmeticOutput,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.RealBody,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.RealExecution,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.RealExpressions,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.RealFormula,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.RealFrame,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.RealIteration,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.RealLoads,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.RealLoop,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.RealOutput,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.RealPhysical,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.RealPrelude,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.RealProgram,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.RealReturned,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.RealSafety,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.RealSafetyLoop,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.RealSafetyOutput,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.RealSafetyParts,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.RealSafetyPrelude,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.RealSource,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.RealTiles,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.SourceDecomposition,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.WholeGrid,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.Widened,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.SourceBridge,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.SourcePair,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.SourceTiles,`bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Tensor,`bench.repairs.tritonbench_g.attention_score_tile_checked.RealIO,`bench.repairs.tritonbench_g.attention_score_tile_checked.SourceDefinitions,`bench.repairs.tritonbench_g.attention_score_tile_checked.SourceParameters,`bench.repairs.tritonbench_g.layer_norm_liger.Contract,`bench.repairs.tritonbench_g.layer_norm_liger.Correct,`bench.repairs.tritonbench_g.layer_norm_liger.FPEquiv,`bench.repairs.tritonbench_g.layer_norm_liger.IO,`bench.repairs.tritonbench_g.layer_norm_liger.Kernels,`bench.repairs.tritonbench_g.layer_norm_liger.Misc.HostCorrect,`bench.repairs.tritonbench_g.layer_norm_liger.Misc.HostFPEquiv,`bench.repairs.tritonbench_g.layer_norm_liger.Misc.KernelAudit,`bench.repairs.tritonbench_g.layer_norm_liger.Misc.KernelProofAudit,`bench.repairs.tritonbench_g.layer_norm_liger.Misc.NativeCorrect,`bench.repairs.tritonbench_g.layer_norm_liger.Misc.TensorBridge,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.BackwardContract,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.BackwardInterface,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.BackwardKernels,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.BackwardModel,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.BackwardResult,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.BackwardRun,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.BackwardSoftware,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.BackwardWhole,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.CandidateBackwardRun,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.CandidateBackwardSource,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.CandidateBackwardWhole,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.CandidateForward,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.CandidateSafety,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.CandidateSafetyForward,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.ClosedFP,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.Contract,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.FoldFacts,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.ForwardContract,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.ForwardInterface,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.ForwardModel,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.ForwardResult,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.ForwardRun,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.ForwardSaved,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.ForwardSoftware,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.Frame,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.Grid,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.GridCells,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.GridHost,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.HostContract,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.HostResults,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.IOBindings,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.KernelBoundary,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.KernelFP,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.KernelIndex,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.KernelReal,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.KernelStrides,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.LaunchCoverage,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.Logical,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.LogicalBackward,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.LogicalForward,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.LogicalIO,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.LogicalInputs,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.NativeIndex,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.NativeInterface,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.NativeLaws,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.NativePlan,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.OrderedMemory,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.OrderedReference,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.Preparation,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.ProgramPair,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.PublicCheck,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.PublicReturn,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.RealInterface,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.ReductionPlan,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.Returned,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.Settings,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.SourceOffsets,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.SourceSafety,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.SourceShape,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.WholeCheck,`bench.repairs.tritonbench_g.layer_norm_liger.Proofs.Wrapper,`bench.repairs.tritonbench_g.matrix_vector_multip.CandidateIO,`bench.repairs.tritonbench_g.matrix_vector_multip.Contract,`bench.repairs.tritonbench_g.matrix_vector_multip.Correct,`bench.repairs.tritonbench_g.matrix_vector_multip.FPEquiv,`bench.repairs.tritonbench_g.matrix_vector_multip.IO,`bench.repairs.tritonbench_g.matrix_vector_multip.Kernels,`bench.repairs.tritonbench_g.matrix_vector_multip.Misc.NativeCorrect,`bench.repairs.tritonbench_g.matrix_vector_multip.Misc.TensorBridge,`bench.repairs.tritonbench_g.matrix_vector_multip.Misc.TileEquiv,`bench.repairs.tritonbench_g.matrix_vector_multip.Proofs.AccessSafety,`bench.repairs.tritonbench_g.matrix_vector_multip.Proofs.Contract,`bench.repairs.tritonbench_g.matrix_vector_multip.Proofs.ExplicitFMA,`bench.repairs.tritonbench_g.matrix_vector_multip.Proofs.FP,`bench.repairs.tritonbench_g.matrix_vector_multip.Proofs.GridPair,`bench.repairs.tritonbench_g.matrix_vector_multip.Proofs.HistoricalCorrect,`bench.repairs.tritonbench_g.matrix_vector_multip.Proofs.HistoricalFPEquiv,`bench.repairs.tritonbench_g.matrix_vector_multip.Proofs.Host,`bench.repairs.tritonbench_g.matrix_vector_multip.Proofs.IO,`bench.repairs.tritonbench_g.matrix_vector_multip.Proofs.Index,`bench.repairs.tritonbench_g.matrix_vector_multip.Proofs.InputBinding,`bench.repairs.tritonbench_g.matrix_vector_multip.Proofs.KernelBinding,`bench.repairs.tritonbench_g.matrix_vector_multip.Proofs.KernelFP,`bench.repairs.tritonbench_g.matrix_vector_multip.Proofs.KernelIndex,`bench.repairs.tritonbench_g.matrix_vector_multip.Proofs.KernelSource,`bench.repairs.tritonbench_g.matrix_vector_multip.Proofs.LogicalIO,`bench.repairs.tritonbench_g.matrix_vector_multip.Proofs.NativeResult,`bench.repairs.tritonbench_g.matrix_vector_multip.Proofs.Numerical,`bench.repairs.tritonbench_g.matrix_vector_multip.Proofs.Real,`bench.repairs.tritonbench_g.matrix_vector_multip.Proofs.RealResult,`bench.repairs.tritonbench_g.matrix_vector_multip.Proofs.ReturnedTensor,`bench.repairs.tritonbench_g.matrix_vector_multip.Proofs.SourceBody,`bench.repairs.tritonbench_g.matrix_vector_multip.Proofs.SourceCode,`bench.repairs.tritonbench_g.matrix_vector_multip.Proofs.SourceLoop,`bench.repairs.tritonbench_g.matrix_vector_multip.Proofs.SourceOutput,`bench.repairs.tritonbench_g.matrix_vector_multip.Proofs.SourcePair,`bench.repairs.tritonbench_g.matrix_vector_multip.Proofs.SourcePairSafety,`bench.repairs.tritonbench_g.matrix_vector_multip.Proofs.SourcePairSafetySteps,`bench.repairs.tritonbench_g.matrix_vector_multip.Proofs.SourcePairSafetyTools,`bench.repairs.tritonbench_g.matrix_vector_multip.Proofs.SourcePrelude,`bench.repairs.tritonbench_g.matrix_vector_multip.Proofs.SourceSafety,`bench.repairs.tritonbench_g.matrix_vector_multip.Proofs.TensorBinding,`bench.repairs.tritonbench_g.matrix_vector_multip.SourceDomain,`bench.repairs.tritonbench_g.matrix_vector_multip.SourceShapes,`bench.repairs.tritonbench_g.rotary_transform_tile_checked.Proofs.SourceTiles,`bench.tests.CallRandomABI,`bench.tests.CallSignature,`bench.tests.FP32ArgumentCopyBody,`bench.tests.IntegerLiteralAssignmentBody,`bench.tests.KnownIntegerRangeStepBody,`bench.tests.TypedDefaultDotLoopBody,`bench.tests.TypedFP32CallArithmetic,`bench.tests.TypedIntegerOperandBody,`bench.tests.WhileBody,`bench.tests.WhileExecution,`bench.tritonbench_g.isfinite_kernel.Proofs.TypedFlatSyntax,`bench.tritonbench_g.layer_norm_liger.FPCorrect,`bench.tritonbench_g.layer_norm_liger.LayerNormLiger,`bench.tritonbench_g.layer_norm_liger.Proofs.FPComplete,`bench.tritonbench_g.layer_norm_liger.Proofs.FPContract,`bench.tritonbench_g.layer_norm_liger.Proofs.FPModel,`bench.tritonbench_g.layer_norm_liger.Proofs.FPShape,`bench.tritonbench_g.layer_norm_liger.Proofs.FPSoftware,`bench.tritonbench_g.layer_norm_liger.Proofs.FPSupplied,`bench.tritonbench_g.layer_norm_liger.Proofs.FPTypedSource,`bench.tritonbench_g.layer_norm_liger.Proofs.FPWitness]
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
#print_fp_assumptions VeriTile.Bench.Repairs.AttentionScoreTileChecked.FPEquiv.score_scaling_equiv
#print_fp_assumptions VeriTile.Bench.Repairs.MatrixVectorMultip.FPEquiv.grid_explicit_fma_equiv
#print_fp_assumptions VeriTile.Bench.Repairs.MatrixVectorMultip.FPEquiv.explicit_fma_equiv
