import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.CheckedHost
import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.Contract
import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.RealRevision.NativeResult

namespace VeriTile.Bench.Repairs.LayerNormFwd.Equivalence
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open Wrapper WrapperNative RealRevision
noncomputable section
variable {α Host : Type} [Inhabited α] {caller : Address → Prop}

specification layer_norm_fwd_repaired_requested_storage_returns
    {α Host : Type} [Inhabited α] {caller : WrapperNative.Address → Prop}
    (store : TensorStore (HostState α Host) Descriptor (Cell α)) (hm : TensorMemory store)
    (hh : HostDescriptors store) (hr : LogicalReadLocal store)
    (tensor : TensorAPI (world (α:=α) (Host:=Host) caller)) (hn : Checked.NoCopy tensor) (device : DispatchAPI (world caller))
    (hs : StorageAPI store tensor) (hf : FreshAllocationAPI store tensor)
    (hc : CallerAllocationAPI store tensor caller) (hd : DispatchStorage store device)
    (hl : LayoutAPI tensor) (ha : DispatchAdmission device)
    (hmask : MaskSplit store device) (hmem : SuffixMemory device)
    (streams : WrapperStream.API tensor device)
    (primitives : Compiler.FP32Primitives α) (casts : StorageCompiler.StoreCasts α)
    (selection : StorageCompiler.ProducedSelection Host)
    (c : Admission.Call) (initial : HostState α Host) (hp : RequestPresent store (Admission.request c) initial)
    (result : Returned)
    (actual : Outcome (HostState α Host) Returned)
    (admission : Checked.run tensor device
      (WrapperStorage.kernel (caller:=caller) store primitives casts selection) c initial = .ok actual)
    (success : actual.result = .ok result)
    (template : State α) :
    actual.state = ((Wrapper.surround tensor device
      (WrapperStorage.kernel (caller:=caller) store primitives casts selection) (Admission.request c)).run initial).state ∧
    ∃ p args, SuccessRoute tensor device
      (WrapperStorage.kernel (caller:=caller) store primitives casts selection) (Admission.request c) initial result p args ∧
      WrapperStorage.ReturnPost store tensor device streams primitives casts selection (Admission.request c) initial result p args template := by
  obtain ⟨domain, stateEq, resultEq⟩ := Checked.admitted tensor hn device
    (WrapperStorage.kernel (caller:=caller) store primitives casts selection) c initial actual admission
  have hprofile := Admission.defaultInputs c domain
  have oldSuccess := resultEq.symm.trans success
  refine ⟨stateEq, ?_⟩
  exact RealRevision.Equivalence.layer_norm_fwd_repaired_requested_storage_returns
    store hm hh hr tensor device hs hf hc hd hl ha hmask hmem streams primitives casts selection
    (Admission.request c) initial hp hprofile result oldSuccess template

#axiomsClean layer_norm_fwd_repaired_requested_storage_returns
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.Equivalence
