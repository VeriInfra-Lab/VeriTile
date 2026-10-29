import bench.repairs.tritonbench_g.attention_kernel_aligned.KernelOnly.Endpoints
import bench.repairs.tritonbench_g.attention_kernel_aligned.KernelOnly.BF16Endpoints
import bench.repairs.tritonbench_g.attention_kernel_aligned.Proofs.AcceptedFP
import bench.repairs.tritonbench_g.attention_kernel_aligned.Proofs.Contract
import bench.repairs.tritonbench_g.attention_kernel_aligned.Proofs.BF16AcceptedFP
import bench.repairs.tritonbench_g.attention_kernel_aligned.Proofs.BF16Contract
namespace VeriTile.Bench.Repairs.AttentionKernelAligned.FPEquiv
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.AttentionKernelAligned Source Whole Full Native
noncomputable section
specification aligned_native (I : Instructions) (p : Parameters) (observed : Contract.Observations) (admitted : Contract.InputContract p observed) (storage : TensorStorage p)
    (abi : InputABI p) (before : State Word) (inputs : RawInputs p before)
    (api : Contract.StorageAPI p storage observed) (profile : Profile) (compiler : CompilerBinding p storage abi (Contract.placement p storage observed api admitted) profile)
    (flat : Fin (Contract.placement p storage observed api admitted).arena → BitVec 8) (binding : WordBinding p storage (Contract.placement p storage observed api admitted) flat before)
    (work : List Program) (legal : (programs (grid p)).Perm work) :
    Spec.Real (∃ result,invoke (fun _=>algebraFor p) (fun _=>modelFor p) p abi (injectState before) work=some result ∧
      result.output=abi.output ∧ result.pythonNone=() ∧
      WordResult I p (Contract.guard p observed admitted).selected before (projectState I p before result.state) ∧
      ByteResult I p (Contract.guard p observed admitted).selected storage (Contract.placement p storage observed api admitted) before flat
        (flattenBytes p storage (Contract.placement p storage observed api admitted) (fun _=>encode16) flat (projectState I p before result.state))) :=
  Native.host_native I p (Contract.guard p observed admitted) storage abi before inputs (Contract.placement p storage observed api admitted) profile compiler flat binding work legal

#axiomsClean aligned_native
specification aligned_native_if_local (actual standard : Instructions) (agreement : InstructionAgreement actual standard)
    (p : Parameters) (observed : Contract.Observations) (admitted : Contract.InputContract p observed) (storage : TensorStorage p)
    (abi : InputABI p) (before : State Word) (inputs : RawInputs p before)
    (api : Contract.StorageAPI p storage observed) (profile : Profile) (compiler : CompilerBinding p storage abi (Contract.placement p storage observed api admitted) profile)
    (flat : Fin (Contract.placement p storage observed api admitted).arena → BitVec 8) (binding : WordBinding p storage (Contract.placement p storage observed api admitted) flat before)
    (work : List Program) (legal : (programs (grid p)).Perm work) :
    Spec.Real ((∀ job∈work,CurrentOperands p storage (Contract.placement p storage observed api admitted) flat (start (grid p) job before)) ∧
    ∃ result,invoke (fun _=>algebraFor p) (fun _=>modelFor p) p abi (injectState before) work=some result ∧
      result.output=abi.output ∧ result.pythonNone=() ∧
      WordResult standard p (Contract.guard p observed admitted).selected before (projectState actual p before result.state) ∧
      ByteResult standard p (Contract.guard p observed admitted).selected storage (Contract.placement p storage observed api admitted) before flat
        (flattenBytes p storage (Contract.placement p storage observed api admitted) (fun _=>encode16) flat (projectState actual p before result.state))) :=
  Native.fpequiv_current actual standard agreement p (Contract.guard p observed admitted) storage abi before inputs (Contract.placement p storage observed api admitted) profile compiler flat binding work legal

#axiomsClean aligned_native_if_local
end
end VeriTile.Bench.Repairs.AttentionKernelAligned.FPEquiv
namespace VeriTile.Bench.Repairs.AttentionKernelAligned.BF16.FPEquiv
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.AttentionKernelAligned.BF16 Source Whole Full Native
noncomputable section
specification aligned_native (I : Instructions) (p : Parameters) (observed : Contract.Observations) (admitted : Contract.InputContract p observed) (storage : TensorStorage p)
    (abi : InputABI p) (before : State Word) (inputs : RawInputs p before)
    (api : Contract.StorageAPI p storage observed) (profile : Profile) (compiler : CompilerBinding p storage abi (Contract.placement p storage observed api admitted) profile)
    (flat : Fin (Contract.placement p storage observed api admitted).arena → BitVec 8) (binding : WordBinding p storage (Contract.placement p storage observed api admitted) flat before)
    (work : List Program) (legal : (programs (grid p)).Perm work) :
    Spec.Real (∃ result,invoke (fun _=>algebraFor p) (fun _=>modelFor p) p abi (injectState before) work=some result ∧
      result.output=abi.output ∧ result.pythonNone=() ∧
      WordResult I p (Contract.guard p observed admitted).selected before (projectState I p before result.state) ∧
      ByteResult I p (Contract.guard p observed admitted).selected storage (Contract.placement p storage observed api admitted) before flat
        (flattenBytes p storage (Contract.placement p storage observed api admitted) (fun _=>encode16) flat (projectState I p before result.state))) :=
  Native.host_native I p (Contract.guard p observed admitted) storage abi before inputs (Contract.placement p storage observed api admitted) profile compiler flat binding work legal

#axiomsClean aligned_native
specification aligned_native_if_local (actual standard : Instructions) (agreement : InstructionAgreement actual standard)
    (p : Parameters) (observed : Contract.Observations) (admitted : Contract.InputContract p observed) (storage : TensorStorage p)
    (abi : InputABI p) (before : State Word) (inputs : RawInputs p before)
    (api : Contract.StorageAPI p storage observed) (profile : Profile) (compiler : CompilerBinding p storage abi (Contract.placement p storage observed api admitted) profile)
    (flat : Fin (Contract.placement p storage observed api admitted).arena → BitVec 8) (binding : WordBinding p storage (Contract.placement p storage observed api admitted) flat before)
    (work : List Program) (legal : (programs (grid p)).Perm work) :
    Spec.Real ((∀ job∈work,CurrentOperands p storage (Contract.placement p storage observed api admitted) flat (start (grid p) job before)) ∧
    ∃ result,invoke (fun _=>algebraFor p) (fun _=>modelFor p) p abi (injectState before) work=some result ∧
      result.output=abi.output ∧ result.pythonNone=() ∧
      WordResult standard p (Contract.guard p observed admitted).selected before (projectState actual p before result.state) ∧
      ByteResult standard p (Contract.guard p observed admitted).selected storage (Contract.placement p storage observed api admitted) before flat
        (flattenBytes p storage (Contract.placement p storage observed api admitted) (fun _=>encode16) flat (projectState actual p before result.state))) :=
  Native.fpequiv_current actual standard agreement p (Contract.guard p observed admitted) storage abi before inputs (Contract.placement p storage observed api admitted) profile compiler flat binding work legal

#axiomsClean aligned_native_if_local
end
end VeriTile.Bench.Repairs.AttentionKernelAligned.BF16.FPEquiv

namespace VeriTile.Bench.Repairs.AttentionKernelAligned.FPEquiv
abbrev kernel_native := KernelOnly.kernel_native
noncomputable abbrev kernel_reference := KernelOnly.nativeReference
#axiomsClean kernel_native
end VeriTile.Bench.Repairs.AttentionKernelAligned.FPEquiv

namespace VeriTile.Bench.Repairs.AttentionKernelAligned.BF16.FPEquiv
abbrev kernel_native := KernelOnly.kernel_native
noncomputable abbrev kernel_reference := KernelOnly.nativeReference
#axiomsClean kernel_native
end VeriTile.Bench.Repairs.AttentionKernelAligned.BF16.FPEquiv
