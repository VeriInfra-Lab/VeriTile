import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.CheckedHost
import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.Contract
import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.RealRevision.Result

namespace VeriTile.Bench.Repairs.LayerNormFwd.Correctness
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open Wrapper WrapperNative RealRevision
noncomputable section
variable {α Host : Type} [Inhabited α] {caller : Address → Prop}

specification same_call_real_return
    (store : TensorStore (HostState α Host) Descriptor (Cell α)) (hm : TensorMemory store)
    (hh : HostDescriptors store) (hr : LogicalReadLocal store)
    (tensor : TensorAPI (world (α:=α) (Host:=Host) caller)) (hn : Checked.NoCopy tensor) (device : DispatchAPI (world (α:=α) (Host:=Host) caller))
    (hs : StorageAPI store tensor) (hf : FreshAllocationAPI store tensor)
    (hc : CallerAllocationAPI store tensor caller) (hd : DispatchStorage store device)
    (hl : LayoutAPI tensor) (ha : DispatchAdmission device)
    (hmask : MaskSplit store device) (hmem : SuffixMemory device) (hsplit : SplitCoordinates store device)
    (streams : WrapperStream.API tensor device)
    (primitives : Compiler.FP32Primitives α) (casts : StorageCompiler.StoreCasts α)
    (selection : StorageCompiler.ProducedSelection Host)
    (ρ : HostState α Host → Address → ℝ) (laws : Transport store ρ tensor)
    (shadowStorage : StorageAPI (shadowStore store ρ) (revalueTensor store ρ tensor laws))
    (shadowDispatch : DispatchStorage (shadowStore store ρ) (revalueDispatch device))
    (tf : TensorFrames ρ tensor) (df : DispatchFrames ρ device)
    (epsReal dropoutReal : ℝ) (c : Admission.Call) (initial : HostState α Host)
    (hp : RequestPresent store (Admission.request c) initial) (result : Returned)
    (actual : Outcome (HostState α Host) Returned)
    (admission : Checked.run tensor device
      (WrapperStorage.kernel (caller:=caller) store primitives casts selection) c initial = .ok actual)
    (success : actual.result = .ok result)
    (nativeTemplate : State α) (realTemplate : BlockState) :
    actual.state = ((Wrapper.surround tensor device
      (WrapperStorage.kernel (caller:=caller) store primitives casts selection) (Admission.request c)).run initial).state ∧
    ∃ p args, SuccessRoute tensor device (WrapperStorage.kernel (caller:=caller) store primitives casts selection)
        (Admission.request c) initial result p args ∧
      WrapperStorage.ReturnPost store tensor device streams primitives casts selection (Admission.request c) initial result p args nativeTemplate ∧
      RealRevision.Correctness.RealPost store tensor device primitives casts selection ρ epsReal dropoutReal (Admission.request c) initial result p args realTemplate := by
  obtain ⟨domain, stateEq, resultEq⟩ := Checked.admitted tensor hn device
    (WrapperStorage.kernel (caller:=caller) store primitives casts selection) c initial actual admission
  have hprofile := Admission.defaultInputs c domain
  have oldSuccess := resultEq.symm.trans success
  refine ⟨stateEq, ?_⟩
  exact RealRevision.Correctness.same_call_real_return store hm hh hr tensor device hs hf hc hd hl ha
    hmask hmem hsplit streams primitives casts selection ρ laws shadowStorage shadowDispatch tf df
    epsReal dropoutReal (Admission.request c) initial hp hprofile result oldSuccess nativeTemplate realTemplate

#axiomsClean same_call_real_return
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.Correctness
