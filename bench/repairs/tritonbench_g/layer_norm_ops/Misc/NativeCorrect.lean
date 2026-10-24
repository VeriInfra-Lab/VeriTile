/- Retained single-source native/software correctness. Original declaration
namespaces and full statements are preserved as ordinary auxiliary theorems. -/
import bench.repairs.tritonbench_g.layer_norm_ops.Kernels
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.IONative

namespace VeriTile.Bench.Repairs.LayerNormOps.FPEquiv
open VeriTile.Triton FP.Structural KernelNative.Shared
noncomputable section

/-- The original FP32 source and logical channels, with the existing symbolic
reduction model. Input terms remain arbitrary, including every FP32 payload. -/
def forwardIO (R : Reductions) (p : ForwardParameters) (pid : Nat) : ProgramIO Term Term where
  inputs := Interface.forwardInputs p
  outputs := Interface.forwardOutputs p
  scratch := []
  buffers := [p.x,p.weight,p.bias,p.residual,p.y,p.residualOut,p.mean,p.rstd]
  read := fun i _ => Interface.forwardRead p pid i
  inputMask := fun i _ => Interface.forwardInputMask p i
  write := fun o _ => Interface.forwardWrite p pid o
  outputMask := fun o _ => Interface.forwardOutputMask p o
  scratchWrite := fun k => k.elim0
  scratchMask := fun k => k.elim0
  schedule := fun _ => [Interface.invocation (algebra R) (forwardKernel p) pid]
  encode := id
  observe := fun _ _ => id

def backwardIO (R : Reductions) (p : BackwardParameters) (pid : Nat) : ProgramIO Term Term where
  inputs := Interface.backwardInputs p
  outputs := Interface.backwardOutputs p
  scratch := []
  buffers := [p.x,p.weight,p.bias,p.dy,p.dres,p.mean,p.rstd,p.dx,p.y,p.dresIn,p.dw,p.db]
  read := fun i _ => Interface.backwardRead p i
  inputMask := fun i _ => Interface.backwardInputMask p pid i
  write := fun o _ => Interface.backwardWrite p pid o
  outputMask := fun o _ => Interface.backwardOutputMask p pid o
  scratchWrite := fun k => k.elim0
  scratchMask := fun k => k.elim0
  schedule := fun _ => [Interface.invocation (algebra R) (backwardKernel p) pid]
  encode := id
  observe := fun _ _ => id

/-- Independent finite-input scalar graphs, evaluated in the selected native
order with the actual epsilon word and runtime signed-i32 count conversion. -/
def forwardFormula (R : Reductions) (I : Instructions) (S : OutputSchedule)
    (p : ForwardParameters) (pid : Nat) (epsilon : Word) (v : Interface.ForwardValues Term p) :
    ChannelValues Word (Interface.forwardOutputs p) :=
  fun o j => mapValue
    (Interface.forwardObservation (KernelNative.withCount (withEpsilon I p.epsilon epsilon) p.columns) S p pid o j)
    (Interface.forwardOutputs p)[o].dtype (Interface.forwardGraph (algebra R) p v o j)
def backwardFormula (R : Reductions) (I : Instructions) (S : OutputSchedule)
    (p : BackwardParameters) (pid : Nat) (v : Interface.BackwardValues Term p) :
    ChannelValues Word (Interface.backwardOutputs p) :=
  fun o j => mapValue (Interface.backwardObservation (KernelNative.withCount I p.columns) S p pid o j)
    (Interface.backwardOutputs p)[o].dtype (Interface.backwardGraph (algebra R) p pid v o j)

/-- Selected-object source/native observation. The existing source hash, ABI,
layout, reduction topology and output order binding remain explicit. Primitive
rounding is model data; epsilon/count hypotheses are scalar conversion laws.
There is no transformation-law premise or claim of native-to-Real equality. -/
theorem layer_norm_ops_forward_kernel_equiv (platform : KernelCompiler.Platform)
    (p : ForwardParameters) (pid : Nat) (host : UInt64) (R : Reductions) (S : OutputSchedule)
    (binding : KernelCompiler.Binding platform (.inl p) host R S)
    (conversion : KernelCompiler.Conversion platform (.inl p) host binding)
    (admissible : KernelExecution.ForwardAdmissible p) :
    (forwardIO R p pid).Observes
      (Interface.forwardObservation (platform.primitives binding.object.primitives) S p pid)
      (forwardFormula R (platform.primitives binding.object.primitives) S p pid (platform.epsilon binding.object host)) :=
  Interface.forward_native R (platform.primitives binding.object.primitives) S p pid admissible
    conversion.count (platform.epsilon binding.object host) conversion.epsilon

theorem layer_norm_ops_backward_kernel_equiv (platform : KernelCompiler.Platform)
    (p : BackwardParameters) (pid : Nat) (host : UInt64) (R : Reductions) (S : OutputSchedule)
    (binding : KernelCompiler.Binding platform (.inr p) host R S)
    (conversion : KernelCompiler.CountConversion platform (.inr p) host binding)
    (admissible : KernelReadback.Admissible p) :
    (backwardIO R p pid).Observes
      (Interface.backwardObservation (platform.primitives binding.object.primitives) S p pid)
      (backwardFormula R (platform.primitives binding.object.primitives) S p pid) :=
  Interface.backward_native R (platform.primitives binding.object.primitives) S p pid admissible conversion

#specNonCircular forwardFormula avoiding [forwardKernel,FP.Structural.exec,Forward.reference]
#specNonCircular backwardFormula avoiding [backwardKernel,FP.Structural.exec,Backward.reference,Backward.update]
#axiomsClean layer_norm_ops_forward_kernel_equiv
#axiomsClean layer_norm_ops_backward_kernel_equiv
#stmtSurfaceSubset layer_norm_ops_forward_kernel_equiv ⊆ [ProgramIO.Observes,forwardIO,forwardFormula,
  Interface.forwardObservation,ForwardParameters,KernelExecution.ForwardAdmissible,KernelCompiler.Platform,
  KernelCompiler.Binding,KernelCompiler.Conversion,KernelCompiler.Platform.primitives,KernelCompiler.Platform.epsilon,
  KernelCompiler.Binding.object,KernelCompiler.Object.primitives,Reductions,OutputSchedule,Term,Word,
  BackwardParameters,FastLayernorm.KernelNative.instInhabitedTerm]
#stmtSurfaceSubset layer_norm_ops_backward_kernel_equiv ⊆ [ProgramIO.Observes,backwardIO,backwardFormula,
  Interface.backwardObservation,BackwardParameters,KernelReadback.Admissible,KernelCompiler.Platform,
  KernelCompiler.Binding,KernelCompiler.CountConversion,KernelCompiler.Platform.primitives,
  KernelCompiler.Binding.object,KernelCompiler.Object.primitives,Reductions,OutputSchedule,Term,Word,
  ForwardParameters,FastLayernorm.KernelNative.instInhabitedTerm]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormOps.FPEquiv
