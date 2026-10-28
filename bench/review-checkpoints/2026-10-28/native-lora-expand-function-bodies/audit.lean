import bench.repairs.tritonbench_g.isfinite_kernel.Correct
import bench.repairs.tritonbench_g.isfinite_kernel.FPEquiv
import bench.repairs.tritonbench_g.isfinite_kernel.Misc.LegacyCorrect
import bench.repairs.tritonbench_g.isfinite_kernel.Misc.LegacyFPEquiv
import bench.repairs.tritonbench_g.isfinite_kernel.Misc.NativeEndpoints
import bench.repairs.tritonbench_g.isfinite_kernel.Misc.RealEndpoints
import bench.repairs.tritonbench_g.isfinite_kernel.Misc.TensorBridge
import bench.repairs.tritonbench_g.isfinite_kernel.Misc.WordCorrect
import bench.repairs.tritonbench_g.isfinite_kernel.Proofs.Contract
import bench.repairs.tritonbench_g.isfinite_kernel.Proofs.EquivIO
import bench.repairs.tritonbench_g.isfinite_kernel.Proofs.Equivalence
import bench.repairs.tritonbench_g.isfinite_kernel.Proofs.Finite
import bench.repairs.tritonbench_g.isfinite_kernel.Proofs.Host
import bench.repairs.tritonbench_g.isfinite_kernel.Proofs.IO
import bench.repairs.tritonbench_g.isfinite_kernel.Proofs.Kernel
import bench.repairs.tritonbench_g.isfinite_kernel.Proofs.LogicalReader
import bench.repairs.tritonbench_g.isfinite_kernel.Proofs.Lowering
import bench.repairs.tritonbench_g.isfinite_kernel.Proofs.Memory
import bench.repairs.tritonbench_g.isfinite_kernel.Proofs.SourceBridge
import bench.repairs.tritonbench_g.isfinite_kernel.Proofs.Storage
import bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.CandidateKernels
import bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.Contract
import bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.Correct
import bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.FPEquiv
import bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.IO
import bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.Kernels
import bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.Misc.Access
import bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.Misc.Binding
import bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.Misc.Checked
import bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.Misc.Guard
import bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.Misc.Native
import bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.Misc.NativeCorrect
import bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.Misc.OutputFrame
import bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.Misc.Physical
import bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.Misc.PublicAudit
import bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.Misc.Scheduling
import bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.Misc.TensorBridge
import bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.Misc.UnitSplitBytes
import bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.PairIO
import bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.Proofs.IOBinding
import bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.Proofs.IOCorrect
import bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.Proofs.IOGrid
import bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.Proofs.IOResults
import bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.Proofs.IOSafety
import bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.Proofs.IOSafetyBody
import bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.Proofs.IOSafetyComplete
import bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.Proofs.IOSafetyControl
import bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.Proofs.IOSafetyLoop
import bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.Proofs.IOSafetySetup
import bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.Proofs.KernelNativeResults
import bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.Proofs.KernelRealResults
import bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.Proofs.LogicalReader
import bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.Proofs.Pair
import bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.Proofs.SourceConnection
import bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.SourceCompatibility
import bench.repairs.tritonbench_g.lora_expand_gemv.Contract
import bench.repairs.tritonbench_g.lora_expand_gemv.Correct
import bench.repairs.tritonbench_g.lora_expand_gemv.FPEquiv
import bench.repairs.tritonbench_g.lora_expand_gemv.Kernels
import bench.repairs.tritonbench_g.lora_expand_gemv.Proofs.Bounds
import bench.repairs.tritonbench_g.lora_expand_gemv.Proofs.BranchAssembly
import bench.repairs.tritonbench_g.lora_expand_gemv.Proofs.BranchBody
import bench.repairs.tritonbench_g.lora_expand_gemv.Proofs.BranchBodyValue
import bench.repairs.tritonbench_g.lora_expand_gemv.Proofs.BranchComplete
import bench.repairs.tritonbench_g.lora_expand_gemv.Proofs.BranchExecution
import bench.repairs.tritonbench_g.lora_expand_gemv.Proofs.BranchFrame
import bench.repairs.tritonbench_g.lora_expand_gemv.Proofs.BranchLoop
import bench.repairs.tritonbench_g.lora_expand_gemv.Proofs.BranchReference
import bench.repairs.tritonbench_g.lora_expand_gemv.Proofs.Caller
import bench.repairs.tritonbench_g.lora_expand_gemv.Proofs.Direct.Checked
import bench.repairs.tritonbench_g.lora_expand_gemv.Proofs.Direct.Compiler
import bench.repairs.tritonbench_g.lora_expand_gemv.Proofs.Direct.Native
import bench.repairs.tritonbench_g.lora_expand_gemv.Proofs.Direct.Physical
import bench.repairs.tritonbench_g.lora_expand_gemv.Proofs.Direct.Real
import bench.repairs.tritonbench_g.lora_expand_gemv.Proofs.Direct.Scheduling
import bench.repairs.tritonbench_g.lora_expand_gemv.Proofs.Geometry
import bench.repairs.tritonbench_g.lora_expand_gemv.Proofs.Grid
import bench.repairs.tritonbench_g.lora_expand_gemv.Proofs.Host
import bench.repairs.tritonbench_g.lora_expand_gemv.Proofs.LegacyCorrect
import bench.repairs.tritonbench_g.lora_expand_gemv.Proofs.LegacyFPEquiv
import bench.repairs.tritonbench_g.lora_expand_gemv.Proofs.Native
import bench.repairs.tritonbench_g.lora_expand_gemv.Proofs.Physical
import bench.repairs.tritonbench_g.lora_expand_gemv.Proofs.Reader
import bench.repairs.tritonbench_g.lora_expand_gemv.Proofs.Real
import bench.repairs.tritonbench_g.lora_expand_gemv.Proofs.Single
import bench.repairs.tritonbench_g.lora_expand_gemv.Proofs.Witness
import bench.repairs.tritonbench_g.lora_expand_gemv.SourceParameters
import bench.repairs.tritonbench_g.pow_scalar_tensor.Correct
import bench.repairs.tritonbench_g.pow_scalar_tensor.FPEquiv
import bench.repairs.tritonbench_g.pow_scalar_tensor.IO
import bench.repairs.tritonbench_g.pow_scalar_tensor.Misc.NativeObservation
import bench.repairs.tritonbench_g.pow_scalar_tensor.Misc.TensorBridge
import bench.repairs.tritonbench_g.pow_scalar_tensor.Proofs.Contract
import bench.repairs.tritonbench_g.pow_scalar_tensor.Proofs.HostFP32
import bench.repairs.tritonbench_g.pow_scalar_tensor.Proofs.HostReal
import bench.repairs.tritonbench_g.pow_scalar_tensor.Proofs.IOInterface
import bench.repairs.tritonbench_g.pow_scalar_tensor.Proofs.IOResults
import bench.repairs.tritonbench_g.pow_scalar_tensor.Proofs.Kernel
import bench.repairs.tritonbench_g.pow_scalar_tensor.Proofs.KernelAccess
import bench.repairs.tritonbench_g.pow_scalar_tensor.Proofs.KernelInterface
import bench.repairs.tritonbench_g.pow_scalar_tensor.Proofs.KernelSafety
import bench.repairs.tritonbench_g.pow_scalar_tensor.Proofs.LogicalReader
import bench.repairs.tritonbench_g.pow_scalar_tensor.Proofs.Physical
import bench.repairs.tritonbench_g.pow_scalar_tensor.Proofs.RawCorrect
import bench.repairs.tritonbench_g.pow_scalar_tensor.Proofs.RawNative
import bench.repairs.tritonbench_g.pow_scalar_tensor.Proofs.ReaderBefore
import bench.repairs.tritonbench_g.pow_scalar_tensor.Proofs.SafetySteps
import bench.repairs.tritonbench_g.pow_scalar_tensor.Proofs.Source
import bench.repairs.tritonbench_g.pow_scalar_tensor.Proofs.SourceResults
import bench.repairs.tritonbench_g.pow_scalar_tensor.Proofs.TileSafety
import bench.repairs.tritonbench_g.relu_strided_buffer.Contract
import bench.repairs.tritonbench_g.relu_strided_buffer.Correct
import bench.repairs.tritonbench_g.relu_strided_buffer.FPEquiv
import bench.repairs.tritonbench_g.relu_strided_buffer.IO
import bench.repairs.tritonbench_g.relu_strided_buffer.Misc.TensorBridge
import bench.repairs.tritonbench_g.relu_strided_buffer.Proofs.Contract
import bench.repairs.tritonbench_g.relu_strided_buffer.Proofs.Host
import bench.repairs.tritonbench_g.relu_strided_buffer.Proofs.HostFP32
import bench.repairs.tritonbench_g.relu_strided_buffer.Proofs.HostReal
import bench.repairs.tritonbench_g.relu_strided_buffer.Proofs.Kernel
import bench.repairs.tritonbench_g.relu_strided_buffer.Proofs.KernelNativeResult
import bench.repairs.tritonbench_g.relu_strided_buffer.Proofs.KernelRealResult
import bench.repairs.tritonbench_g.relu_strided_buffer.Proofs.LogicalReader
import bench.repairs.tritonbench_g.relu_strided_buffer.Proofs.Memory
import bench.repairs.tritonbench_g.relu_strided_buffer.Proofs.NativeGridIO
import bench.repairs.tritonbench_g.relu_strided_buffer.Proofs.PublicIO
import bench.repairs.tritonbench_g.relu_strided_buffer.Proofs.RawKernel
import bench.repairs.tritonbench_g.relu_strided_buffer.Proofs.ReaderBefore
import bench.repairs.tritonbench_g.relu_strided_buffer.Proofs.Source
import bench.repairs.tritonbench_g.relu_strided_buffer.Proofs.SourceEquivalence
import bench.repairs.tritonbench_g.relu_strided_buffer.Proofs.SourceSafety
import bench.repairs.tritonbench_g.relu_strided_buffer.Proofs.TileSafety
import bench.tritonbench_g.lora_expand_gemv.BranchCorrect
import bench.tritonbench_g.lora_expand_gemv.FPCorrect
import bench.tritonbench_g.lora_expand_gemv.LogicalHostCorrect
import bench.tritonbench_g.lora_expand_gemv.LoraExpandGemv
import bench.tritonbench_g.lora_expand_gemv.Proofs.BranchAssembly
import bench.tritonbench_g.lora_expand_gemv.Proofs.BranchBody
import bench.tritonbench_g.lora_expand_gemv.Proofs.BranchBodyValue
import bench.tritonbench_g.lora_expand_gemv.Proofs.BranchComplete
import bench.tritonbench_g.lora_expand_gemv.Proofs.BranchExecution
import bench.tritonbench_g.lora_expand_gemv.Proofs.BranchFrame
import bench.tritonbench_g.lora_expand_gemv.Proofs.BranchLoop
import bench.tritonbench_g.lora_expand_gemv.Proofs.BranchReference
import bench.tritonbench_g.lora_expand_gemv.Proofs.BranchShape
import bench.tritonbench_g.lora_expand_gemv.Proofs.BranchSource
import bench.tritonbench_g.lora_expand_gemv.Proofs.BranchWitness
import bench.tritonbench_g.lora_expand_gemv.Proofs.FPComplete
import bench.tritonbench_g.lora_expand_gemv.Proofs.FPExecution
import bench.tritonbench_g.lora_expand_gemv.Proofs.FPModel
import bench.tritonbench_g.lora_expand_gemv.Proofs.FPReference
import bench.tritonbench_g.lora_expand_gemv.Proofs.FPShape
import bench.tritonbench_g.lora_expand_gemv.Proofs.FPTypedSource
import bench.tritonbench_g.lora_expand_gemv.Proofs.FPWitness
import bench.tritonbench_g.lora_expand_gemv.Proofs.HostCases
import bench.tritonbench_g.lora_expand_gemv.Proofs.HostGrid
import bench.tritonbench_g.lora_expand_gemv.Proofs.HostMetadata
import bench.tritonbench_g.lora_expand_gemv.Proofs.HostProperties
import bench.tritonbench_g.lora_expand_gemv.SourceParameters
import bench.tritonbench_g.pow_scalar_tensor.Correct
import bench.tritonbench_g.pow_scalar_tensor.FPEquiv
import bench.tritonbench_g.pow_scalar_tensor.Proofs.Audit
import bench.tritonbench_g.pow_scalar_tensor.Proofs.Numerics
import bench.tritonbench_g.pow_scalar_tensor.Proofs.WholeAccess
import bench.tritonbench_g.pow_scalar_tensor.Proofs.WholeFP32
import bench.tritonbench_g.pow_scalar_tensor.Proofs.WholeHost
import bench.tritonbench_g.pow_scalar_tensor.Proofs.WholeMetadata
import bench.tritonbench_g.pow_scalar_tensor.Proofs.WholeReal
import bench.tritonbench_g.pow_scalar_tensor.Proofs.WholeScalarABI
import bench.tritonbench_g.pow_scalar_tensor.Proofs.WholeSource
import bench.tritonbench_g.pow_scalar_tensor.Proofs.WholeTile
import bench.tritonbench_g.relu_strided_buffer.Correct
import bench.tritonbench_g.relu_strided_buffer.FPEquiv
import bench.tritonbench_g.relu_strided_buffer.Kernels
import bench.tritonbench_g.relu_strided_buffer.Proofs.CandidateReal
import bench.tritonbench_g.relu_strided_buffer.Proofs.FP
import bench.tritonbench_g.relu_strided_buffer.Proofs.HostExecution
import bench.tritonbench_g.relu_strided_buffer.Proofs.HostIndex32
import bench.tritonbench_g.relu_strided_buffer.Proofs.HostPlan
import bench.tritonbench_g.relu_strided_buffer.Proofs.Real
import bench.tritonbench_g.relu_strided_buffer.Proofs.SourceHostExecution
import bench.tritonbench_g.relu_strided_buffer.Proofs.StructuralHost
import bench.tritonbench_g.relu_strided_buffer.Proofs.StructuralTile
import bench.tritonbench_g.relu_strided_buffer.Proofs.WholeAccess
import bench.tritonbench_g.relu_strided_buffer.Proofs.WholeFP32
import bench.tritonbench_g.relu_strided_buffer.Proofs.WholeMetadata
import bench.tritonbench_g.relu_strided_buffer.Proofs.WholeReal
import bench.tritonbench_g.relu_strided_buffer.Proofs.WholeSource
import bench.tritonbench_g.relu_strided_buffer.Proofs.WordHost
import bench.tritonbench_g.relu_strided_buffer.Proofs.WordTile
import VeriTile.Meta.StatementAudit

open Lean Elab Command
run_cmd do
  let env ← getEnv
  let modules : Array Name := #[`bench.repairs.tritonbench_g.isfinite_kernel.Correct,`bench.repairs.tritonbench_g.isfinite_kernel.FPEquiv,`bench.repairs.tritonbench_g.isfinite_kernel.Misc.LegacyCorrect,`bench.repairs.tritonbench_g.isfinite_kernel.Misc.LegacyFPEquiv,`bench.repairs.tritonbench_g.isfinite_kernel.Misc.NativeEndpoints,`bench.repairs.tritonbench_g.isfinite_kernel.Misc.RealEndpoints,`bench.repairs.tritonbench_g.isfinite_kernel.Misc.TensorBridge,`bench.repairs.tritonbench_g.isfinite_kernel.Misc.WordCorrect,`bench.repairs.tritonbench_g.isfinite_kernel.Proofs.Contract,`bench.repairs.tritonbench_g.isfinite_kernel.Proofs.EquivIO,`bench.repairs.tritonbench_g.isfinite_kernel.Proofs.Equivalence,`bench.repairs.tritonbench_g.isfinite_kernel.Proofs.Finite,`bench.repairs.tritonbench_g.isfinite_kernel.Proofs.Host,`bench.repairs.tritonbench_g.isfinite_kernel.Proofs.IO,`bench.repairs.tritonbench_g.isfinite_kernel.Proofs.Kernel,`bench.repairs.tritonbench_g.isfinite_kernel.Proofs.LogicalReader,`bench.repairs.tritonbench_g.isfinite_kernel.Proofs.Lowering,`bench.repairs.tritonbench_g.isfinite_kernel.Proofs.Memory,`bench.repairs.tritonbench_g.isfinite_kernel.Proofs.SourceBridge,`bench.repairs.tritonbench_g.isfinite_kernel.Proofs.Storage,`bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.CandidateKernels,`bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.Contract,`bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.Correct,`bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.FPEquiv,`bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.IO,`bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.Kernels,`bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.Misc.Access,`bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.Misc.Binding,`bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.Misc.Checked,`bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.Misc.Guard,`bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.Misc.Native,`bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.Misc.NativeCorrect,`bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.Misc.OutputFrame,`bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.Misc.Physical,`bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.Misc.PublicAudit,`bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.Misc.Scheduling,`bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.Misc.TensorBridge,`bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.Misc.UnitSplitBytes,`bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.PairIO,`bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.Proofs.IOBinding,`bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.Proofs.IOCorrect,`bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.Proofs.IOGrid,`bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.Proofs.IOResults,`bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.Proofs.IOSafety,`bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.Proofs.IOSafetyBody,`bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.Proofs.IOSafetyComplete,`bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.Proofs.IOSafetyControl,`bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.Proofs.IOSafetyLoop,`bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.Proofs.IOSafetySetup,`bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.Proofs.KernelNativeResults,`bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.Proofs.KernelRealResults,`bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.Proofs.LogicalReader,`bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.Proofs.Pair,`bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.Proofs.SourceConnection,`bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.SourceCompatibility,`bench.repairs.tritonbench_g.lora_expand_gemv.Contract,`bench.repairs.tritonbench_g.lora_expand_gemv.Correct,`bench.repairs.tritonbench_g.lora_expand_gemv.FPEquiv,`bench.repairs.tritonbench_g.lora_expand_gemv.Kernels,`bench.repairs.tritonbench_g.lora_expand_gemv.Proofs.Bounds,`bench.repairs.tritonbench_g.lora_expand_gemv.Proofs.BranchAssembly,`bench.repairs.tritonbench_g.lora_expand_gemv.Proofs.BranchBody,`bench.repairs.tritonbench_g.lora_expand_gemv.Proofs.BranchBodyValue,`bench.repairs.tritonbench_g.lora_expand_gemv.Proofs.BranchComplete,`bench.repairs.tritonbench_g.lora_expand_gemv.Proofs.BranchExecution,`bench.repairs.tritonbench_g.lora_expand_gemv.Proofs.BranchFrame,`bench.repairs.tritonbench_g.lora_expand_gemv.Proofs.BranchLoop,`bench.repairs.tritonbench_g.lora_expand_gemv.Proofs.BranchReference,`bench.repairs.tritonbench_g.lora_expand_gemv.Proofs.Caller,`bench.repairs.tritonbench_g.lora_expand_gemv.Proofs.Direct.Checked,`bench.repairs.tritonbench_g.lora_expand_gemv.Proofs.Direct.Compiler,`bench.repairs.tritonbench_g.lora_expand_gemv.Proofs.Direct.Native,`bench.repairs.tritonbench_g.lora_expand_gemv.Proofs.Direct.Physical,`bench.repairs.tritonbench_g.lora_expand_gemv.Proofs.Direct.Real,`bench.repairs.tritonbench_g.lora_expand_gemv.Proofs.Direct.Scheduling,`bench.repairs.tritonbench_g.lora_expand_gemv.Proofs.Geometry,`bench.repairs.tritonbench_g.lora_expand_gemv.Proofs.Grid,`bench.repairs.tritonbench_g.lora_expand_gemv.Proofs.Host,`bench.repairs.tritonbench_g.lora_expand_gemv.Proofs.LegacyCorrect,`bench.repairs.tritonbench_g.lora_expand_gemv.Proofs.LegacyFPEquiv,`bench.repairs.tritonbench_g.lora_expand_gemv.Proofs.Native,`bench.repairs.tritonbench_g.lora_expand_gemv.Proofs.Physical,`bench.repairs.tritonbench_g.lora_expand_gemv.Proofs.Reader,`bench.repairs.tritonbench_g.lora_expand_gemv.Proofs.Real,`bench.repairs.tritonbench_g.lora_expand_gemv.Proofs.Single,`bench.repairs.tritonbench_g.lora_expand_gemv.Proofs.Witness,`bench.repairs.tritonbench_g.lora_expand_gemv.SourceParameters,`bench.repairs.tritonbench_g.pow_scalar_tensor.Correct,`bench.repairs.tritonbench_g.pow_scalar_tensor.FPEquiv,`bench.repairs.tritonbench_g.pow_scalar_tensor.IO,`bench.repairs.tritonbench_g.pow_scalar_tensor.Misc.NativeObservation,`bench.repairs.tritonbench_g.pow_scalar_tensor.Misc.TensorBridge,`bench.repairs.tritonbench_g.pow_scalar_tensor.Proofs.Contract,`bench.repairs.tritonbench_g.pow_scalar_tensor.Proofs.HostFP32,`bench.repairs.tritonbench_g.pow_scalar_tensor.Proofs.HostReal,`bench.repairs.tritonbench_g.pow_scalar_tensor.Proofs.IOInterface,`bench.repairs.tritonbench_g.pow_scalar_tensor.Proofs.IOResults,`bench.repairs.tritonbench_g.pow_scalar_tensor.Proofs.Kernel,`bench.repairs.tritonbench_g.pow_scalar_tensor.Proofs.KernelAccess,`bench.repairs.tritonbench_g.pow_scalar_tensor.Proofs.KernelInterface,`bench.repairs.tritonbench_g.pow_scalar_tensor.Proofs.KernelSafety,`bench.repairs.tritonbench_g.pow_scalar_tensor.Proofs.LogicalReader,`bench.repairs.tritonbench_g.pow_scalar_tensor.Proofs.Physical,`bench.repairs.tritonbench_g.pow_scalar_tensor.Proofs.RawCorrect,`bench.repairs.tritonbench_g.pow_scalar_tensor.Proofs.RawNative,`bench.repairs.tritonbench_g.pow_scalar_tensor.Proofs.ReaderBefore,`bench.repairs.tritonbench_g.pow_scalar_tensor.Proofs.SafetySteps,`bench.repairs.tritonbench_g.pow_scalar_tensor.Proofs.Source,`bench.repairs.tritonbench_g.pow_scalar_tensor.Proofs.SourceResults,`bench.repairs.tritonbench_g.pow_scalar_tensor.Proofs.TileSafety,`bench.repairs.tritonbench_g.relu_strided_buffer.Contract,`bench.repairs.tritonbench_g.relu_strided_buffer.Correct,`bench.repairs.tritonbench_g.relu_strided_buffer.FPEquiv,`bench.repairs.tritonbench_g.relu_strided_buffer.IO,`bench.repairs.tritonbench_g.relu_strided_buffer.Misc.TensorBridge,`bench.repairs.tritonbench_g.relu_strided_buffer.Proofs.Contract,`bench.repairs.tritonbench_g.relu_strided_buffer.Proofs.Host,`bench.repairs.tritonbench_g.relu_strided_buffer.Proofs.HostFP32,`bench.repairs.tritonbench_g.relu_strided_buffer.Proofs.HostReal,`bench.repairs.tritonbench_g.relu_strided_buffer.Proofs.Kernel,`bench.repairs.tritonbench_g.relu_strided_buffer.Proofs.KernelNativeResult,`bench.repairs.tritonbench_g.relu_strided_buffer.Proofs.KernelRealResult,`bench.repairs.tritonbench_g.relu_strided_buffer.Proofs.LogicalReader,`bench.repairs.tritonbench_g.relu_strided_buffer.Proofs.Memory,`bench.repairs.tritonbench_g.relu_strided_buffer.Proofs.NativeGridIO,`bench.repairs.tritonbench_g.relu_strided_buffer.Proofs.PublicIO,`bench.repairs.tritonbench_g.relu_strided_buffer.Proofs.RawKernel,`bench.repairs.tritonbench_g.relu_strided_buffer.Proofs.ReaderBefore,`bench.repairs.tritonbench_g.relu_strided_buffer.Proofs.Source,`bench.repairs.tritonbench_g.relu_strided_buffer.Proofs.SourceEquivalence,`bench.repairs.tritonbench_g.relu_strided_buffer.Proofs.SourceSafety,`bench.repairs.tritonbench_g.relu_strided_buffer.Proofs.TileSafety,`bench.tritonbench_g.lora_expand_gemv.BranchCorrect,`bench.tritonbench_g.lora_expand_gemv.FPCorrect,`bench.tritonbench_g.lora_expand_gemv.LogicalHostCorrect,`bench.tritonbench_g.lora_expand_gemv.LoraExpandGemv,`bench.tritonbench_g.lora_expand_gemv.Proofs.BranchAssembly,`bench.tritonbench_g.lora_expand_gemv.Proofs.BranchBody,`bench.tritonbench_g.lora_expand_gemv.Proofs.BranchBodyValue,`bench.tritonbench_g.lora_expand_gemv.Proofs.BranchComplete,`bench.tritonbench_g.lora_expand_gemv.Proofs.BranchExecution,`bench.tritonbench_g.lora_expand_gemv.Proofs.BranchFrame,`bench.tritonbench_g.lora_expand_gemv.Proofs.BranchLoop,`bench.tritonbench_g.lora_expand_gemv.Proofs.BranchReference,`bench.tritonbench_g.lora_expand_gemv.Proofs.BranchShape,`bench.tritonbench_g.lora_expand_gemv.Proofs.BranchSource,`bench.tritonbench_g.lora_expand_gemv.Proofs.BranchWitness,`bench.tritonbench_g.lora_expand_gemv.Proofs.FPComplete,`bench.tritonbench_g.lora_expand_gemv.Proofs.FPExecution,`bench.tritonbench_g.lora_expand_gemv.Proofs.FPModel,`bench.tritonbench_g.lora_expand_gemv.Proofs.FPReference,`bench.tritonbench_g.lora_expand_gemv.Proofs.FPShape,`bench.tritonbench_g.lora_expand_gemv.Proofs.FPTypedSource,`bench.tritonbench_g.lora_expand_gemv.Proofs.FPWitness,`bench.tritonbench_g.lora_expand_gemv.Proofs.HostCases,`bench.tritonbench_g.lora_expand_gemv.Proofs.HostGrid,`bench.tritonbench_g.lora_expand_gemv.Proofs.HostMetadata,`bench.tritonbench_g.lora_expand_gemv.Proofs.HostProperties,`bench.tritonbench_g.lora_expand_gemv.SourceParameters,`bench.tritonbench_g.pow_scalar_tensor.Correct,`bench.tritonbench_g.pow_scalar_tensor.FPEquiv,`bench.tritonbench_g.pow_scalar_tensor.Proofs.Audit,`bench.tritonbench_g.pow_scalar_tensor.Proofs.Numerics,`bench.tritonbench_g.pow_scalar_tensor.Proofs.WholeAccess,`bench.tritonbench_g.pow_scalar_tensor.Proofs.WholeFP32,`bench.tritonbench_g.pow_scalar_tensor.Proofs.WholeHost,`bench.tritonbench_g.pow_scalar_tensor.Proofs.WholeMetadata,`bench.tritonbench_g.pow_scalar_tensor.Proofs.WholeReal,`bench.tritonbench_g.pow_scalar_tensor.Proofs.WholeScalarABI,`bench.tritonbench_g.pow_scalar_tensor.Proofs.WholeSource,`bench.tritonbench_g.pow_scalar_tensor.Proofs.WholeTile,`bench.tritonbench_g.relu_strided_buffer.Correct,`bench.tritonbench_g.relu_strided_buffer.FPEquiv,`bench.tritonbench_g.relu_strided_buffer.Kernels,`bench.tritonbench_g.relu_strided_buffer.Proofs.CandidateReal,`bench.tritonbench_g.relu_strided_buffer.Proofs.FP,`bench.tritonbench_g.relu_strided_buffer.Proofs.HostExecution,`bench.tritonbench_g.relu_strided_buffer.Proofs.HostIndex32,`bench.tritonbench_g.relu_strided_buffer.Proofs.HostPlan,`bench.tritonbench_g.relu_strided_buffer.Proofs.Real,`bench.tritonbench_g.relu_strided_buffer.Proofs.SourceHostExecution,`bench.tritonbench_g.relu_strided_buffer.Proofs.StructuralHost,`bench.tritonbench_g.relu_strided_buffer.Proofs.StructuralTile,`bench.tritonbench_g.relu_strided_buffer.Proofs.WholeAccess,`bench.tritonbench_g.relu_strided_buffer.Proofs.WholeFP32,`bench.tritonbench_g.relu_strided_buffer.Proofs.WholeMetadata,`bench.tritonbench_g.relu_strided_buffer.Proofs.WholeReal,`bench.tritonbench_g.relu_strided_buffer.Proofs.WholeSource,`bench.tritonbench_g.relu_strided_buffer.Proofs.WordHost,`bench.tritonbench_g.relu_strided_buffer.Proofs.WordTile]
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
#print_fp_assumptions VeriTile.Bench.Repairs.LoraExpandGemv.ByteSeparated.FPEquiv.grid_equiv
