import bench.repairs.tritonbench_g.attention_forward_triton.Proofs.SourceFPEquiv
import bench.repairs.tritonbench_g.attention_forward_triton.Contract
namespace VeriTile.Bench.Repairs.AttentionForwardTriton.Checked
open VeriTile Triton FP.Structural Source Whole Native
noncomputable section
theorem checked_native (actual standard : Instructions)
    (agreement : InstructionAgreement actual standard)
    (p : Parameters) (strides : strideChecks p=Bool.true) (launch : launchChecks p=Bool.true) (storage : TensorStorage p)
    (args : Arguments p) (inputsOK : inputChecks p args=Bool.true) (before : State Word) (alloc : Allocation p (admittedABI p args inputsOK) before)
    (inputs : RawInputs p before) (ordinary : OrdinaryPlacement p storage) (pointers : pointerChecks p ordinary.base=Bool.true) (profile : Profile)
    (profileOK : profileCheck p ordinary.base profile=Bool.true)
    (flat : Fin (admittedPlacement p storage ordinary pointers).arena → BitVec 8) (binding : WordBinding p storage (admittedPlacement p storage ordinary pointers) flat alloc.state)
    (work : List Program) (legal : (programs (grid p)).Perm work) :
    ∃ result,invoke (fun _=>algebra) (fun _=>dotModel) p (admittedABI p args inputsOK) (injectState before)
        (injectAllocation p (admittedABI p args inputsOK) before alloc) work=some result ∧ result.tensor=alloc.output ∧
      WordResult standard p alloc.state (projectState actual p alloc.state result.state) ∧
      ByteResult standard p storage (admittedPlacement p storage ordinary pointers) alloc.state flat
        (flattenBytes p storage (admittedPlacement p storage ordinary pointers) (fun _=>encode16) flat (projectState actual p alloc.state result.state)) :=
  Native.fpequiv actual standard agreement p (admittedGuard p strides launch) storage
    (admittedABI p args inputsOK) before alloc inputs (admittedPlacement p storage ordinary pointers) profile
    (admittedCompiler p storage ordinary pointers profile profileOK) flat binding work legal

#axiomsClean checked_native
end
end VeriTile.Bench.Repairs.AttentionForwardTriton.Checked
