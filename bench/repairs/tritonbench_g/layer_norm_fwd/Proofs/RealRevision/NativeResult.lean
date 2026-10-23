import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.RealRevision.Frames
import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.WrapperStorageReturn

/-! The unchanged production-precision whole-wrapper statement is re-exported
through its existing helper. No arbitrary-Real interpretation or local numerical
obligation changes the native result, source pin, or compiler/store profile. -/
namespace VeriTile.Bench.Repairs.LayerNormFwd.RealRevision.Equivalence
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open Wrapper WrapperNative
noncomputable section

/-- Full source wrapper with default FP32 inputs and every explicitly requested
output dtype accepted by ordinary allocation/compilation. The return contract
includes the actual selected compilation, all benchmark predecessor states,
the whole final physical memory, all eight returned-coordinate observations,
the independent ordered NativeParameters reference with local store casts,
consumed emitted stores, actual per-role producer readiness on this launch
stream, and every protected caller byte. Store conversion callbacks refer to
actual native target instructions, without a total LLVM fptosi interpretation.
α is a unified opaque
storage carrier, not a 32-bit word type. No positive-N or finite-only premise:
actual zero-column requests prepare block0 and retain compiler/setup failure. -/
specification layer_norm_fwd_repaired_requested_storage_returns
    {α Host : Type} [Inhabited α] {caller : WrapperNative.Address → Prop}
    (store : TensorStore (HostState α Host) Descriptor (Cell α)) (hm : TensorMemory store)
    (hh : HostDescriptors store) (hr : LogicalReadLocal store)
    (tensor : TensorAPI (world (α:=α) (Host:=Host) caller)) (device : DispatchAPI (world caller))
    (hs : StorageAPI store tensor) (hf : FreshAllocationAPI store tensor)
    (hc : CallerAllocationAPI store tensor caller) (hd : DispatchStorage store device)
    (hl : LayoutAPI tensor) (ha : DispatchAdmission device)
    (hmask : MaskSplit store device) (hmem : SuffixMemory device)
    (streams : WrapperStream.API tensor device)
    (primitives : Compiler.FP32Primitives α) (casts : StorageCompiler.StoreCasts α)
    (selection : StorageCompiler.ProducedSelection Host)
    (q : Request) (initial : HostState α Host) (hp : RequestPresent store q initial)
    (hprofile : DefaultInputs q) (result : Returned)
    (h : ((surround tensor device
      (WrapperStorage.kernel (caller:=caller) store primitives casts selection) q).run initial).result=.ok result)
    (template : State α) :
    ∃ p args, SuccessRoute tensor device
      (WrapperStorage.kernel (caller:=caller) store primitives casts selection) q initial result p args ∧
      WrapperStorage.ReturnPost store tensor device streams primitives casts selection q initial result p args template :=
  WrapperStorage.surround_return store hm hh hr tensor device hs hf hc hd hl ha hmask hmem streams
    primitives casts selection q initial hp hprofile result h template


#axiomsClean layer_norm_fwd_repaired_requested_storage_returns
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormFwd.RealRevision.Equivalence
