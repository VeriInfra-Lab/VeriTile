import bench.repairs.tritonbench_g.rbe_triton_transform.Proofs.Binding
import bench.repairs.tritonbench_g.rbe_triton_transform.Proofs.FP

namespace VeriTile.Bench.Repairs.RbeTritonTransform.FPEquiv
open VeriTile.Triton FP.Structural Numerics
noncomputable section

theorem python_checks_iff_preconditions (x : VeriTile.Torch.ArgumentMetadata) (pos : Contract.Position) :
    Contract.checkArguments x pos=.ok () ↔ Contract.AdmittedArguments x pos :=
  Contract.checkArguments_iff x pos

/-- Complete default-fused local profile, actual returned Tensor, all four
stored bytes and caller frame. Native scalar/compiler conformance remains a
local external binding; this does not disable default source fusion. -/
theorem whole_bits (api : Memory.API Word Store (BitVec 8)) (p : Primitives)
    (r : Request) (input output : RegionName) (s : State Tagged)
    (call : Native.Preparation api leaf r input s)
    (_abi : Native.IndexABI api leaf r input s call)
    (codec : ∀ w k, api.encode w k=wordByte w k) :
    (Native.entry api leaf Tagged.word (operations p) r input output s call).tensor=call.output ∧
    (Native.entry api leaf Tagged.word (operations p) r input output s call).tensor.shape=[r.batches,r.rows,r.columns] ∧
    (∀ b m c, b<r.batches → m<r.rows → c<r.columns →
      Native.logicalRead api (Native.entry api leaf Tagged.word (operations p) r input output s call).memory
        (Native.entry api leaf Tagged.word (operations p) r input output s call).tensor b m c =
          result p r (fun o => ((s.mem input o).read .fp32).word) b m c) ∧
    (∀ b m c, b<r.batches → m<r.rows → c<r.columns → ∀ k : Fin 4,
      api.byte (Native.entry api leaf Tagged.word (operations p) r input output s call).memory
        (Native.logicalAddress (Native.entry api leaf Tagged.word (operations p) r input output s call).tensor b m c+k.val) =
          wordByte (result p r (fun o => ((s.mem input o).read .fp32).word) b m c) k) ∧
    (∀ b, call.caller b → api.byte (Native.entry api leaf Tagged.word (operations p) r input output s call).memory b=api.byte call.before b) := by
  obtain ⟨hd,hs,hv,hb,hf⟩ := Native.whole api leaf Tagged.word (operations p) r input output s call
  refine ⟨hd,hs,?_,?_,hf⟩
  · intro b m c hB hM hC
    rw [hv b m c hB hM hC]
    exact source_profile p r _ b m c
  · intro b m c hB hM hC k
    rw [hb b m c hB hM hC k,codec,source_profile]

/-- Conditional refinement to the independent Real rotation. Thirteen local
numeric groups constrain scalar conversions, transcendental nodes, the actual
right multiply/sign and one fused multiply-add. They contain no whole-result
oracle and are not asserted for every ordinary FP32 input. -/
theorem whole_fp (api : Memory.API Word Store (BitVec 8)) (p : Primitives)
    (r : Request) (input output : RegionName) (s : State Tagged)
    (call : Native.Preparation api leaf r input s)
    (abi : Native.IndexABI api leaf r input s call)
    (codec : ∀ w k, api.encode w k=wordByte w k)
    (nodes : TaskExact p r (fun o => ((s.mem input o).read .fp32).word)) :
    (Native.entry api leaf Tagged.word (operations p) r input output s call).tensor=call.output ∧
    (Native.entry api leaf Tagged.word (operations p) r input output s call).tensor.shape=[r.batches,r.rows,r.columns] ∧
    (∀ b m c, b<r.batches → m<r.rows → c<r.columns →
      decode (Native.logicalRead api (Native.entry api leaf Tagged.word (operations p) r input output s call).memory
        (Native.entry api leaf Tagged.word (operations p) r input output s call).tensor b m c) =
          Spec.result r (fun o => decode (((s.mem input o).read .fp32).word)) b m c) ∧
    (∀ b m c, b<r.batches → m<r.rows → c<r.columns → ∀ k : Fin 4,
      api.byte (Native.entry api leaf Tagged.word (operations p) r input output s call).memory
        (Native.logicalAddress (Native.entry api leaf Tagged.word (operations p) r input output s call).tensor b m c+k.val) =
          wordByte (result p r (fun o => ((s.mem input o).read .fp32).word) b m c) k) ∧
    (∀ b, call.caller b → api.byte (Native.entry api leaf Tagged.word (operations p) r input output s call).memory b=api.byte call.before b) := by
  obtain ⟨hd,hs,hv,hb,hf⟩ := whole_bits api p r input output s call abi codec
  refine ⟨hd,hs,?_,hb,hf⟩
  intro b m c hB hM hC
  rw [hv b m c hB hM hC]
  exact result_decode p r _ nodes b m c hB hM hC

#axiomsClean whole_bits
#axiomsClean whole_fp

/-- Ordered FP32 words and bytes from the actual original input allocation.
No TaskExact or FP-to-Real condition occurs in this source-equivalence result. -/
theorem rbe_triton_transform_repaired_fp32_returns
    (api : Memory.API Word Store (BitVec 8)) (p : Primitives) (r : Request)
    (input output : RegionName) (s : State Tagged) (call : Native.Preparation api leaf r input s)
    (binding : Native.Binding call) (codec : ∀ w k,api.encode w k=wordByte w k) :
    (Native.entry api leaf Tagged.word (operations p) r input output s call).tensor=call.output ∧
    call.output.shape=call.input.shape ∧ call.output.device=call.input.device ∧
    call.output.dtype="float32" ∧ call.output.elementBytes=4 ∧
    (∀ b m c,b<r.batches → m<r.rows → c<r.columns →
      Native.logicalRead api (Native.entry api leaf Tagged.word (operations p) r input output s call).memory
        call.output b m c=result p r (fun o => api.read call.before (call.input.base+4*o)) b m c) ∧
    (∀ b m c,b<r.batches → m<r.rows → c<r.columns → ∀ k : Fin 4,
      api.byte (Native.entry api leaf Tagged.word (operations p) r input output s call).memory
        (Native.logicalAddress call.output b m c+k.val)=
        wordByte (result p r (fun o => api.read call.before (call.input.base+4*o)) b m c) k) ∧
    (∀ offset,offset<call.input.storageBytes →
      api.byte (Native.entry api leaf Tagged.word (operations p) r input output s call).memory
        (call.input.storageBase+offset)=api.byte call.before (call.input.storageBase+offset)) := by
  obtain ⟨hd,_,hv,hbytes,_⟩ := whole_bits api p r input output s call (Native.checked_index_abi call binding) codec
  refine ⟨hd,call.outputShape.trans call.inputShape.symm,call.sameDevice,
    call.outputType.1,call.outputType.2,?_,?_,
    Native.input_backing_frame api leaf Tagged.word (operations p) r input output s call⟩
  · intro b m c hb hm hc
    exact (hv b m c hb hm hc).trans (original_input_bits api p r input s call b m c hb hm hc)
  · intro b m c hb hm hc k
    rw [hd] at hbytes
    rw [hbytes b m c hb hm hc k,original_input_bits api p r input s call b m c hb hm hc]

#axiomsClean rbe_triton_transform_repaired_fp32_returns

theorem rbe_triton_transform_native_addresses
    (api : Memory.API Word Store (BitVec 8)) (r : Request) (input : RegionName)
    (s : State Tagged) (call : Native.Preparation api leaf r input s) (binding : Native.Binding call)
    (b m c : Nat) (hb : b<r.batches) (hm : m<r.rows) (hc : c<r.columns) :
    (BitVec.ofNat 64 (call.input.base+4*r.offset b m c)).toNat=Native.logicalAddress call.input b m c ∧
    (BitVec.ofNat 64 (call.output.base+4*r.offset b m c)).toNat=Native.logicalAddress call.output b m c ∧
    Native.logicalAddress call.input b m c%4=0 ∧ Native.logicalAddress call.output b m c%4=0 ∧
    Native.logicalAddress call.input b m c+4≤call.input.allocationEnd ∧
    Native.logicalAddress call.output b m c+4≤call.output.allocationEnd :=
  Native.native_addresses call binding b m c hb hm hc
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RbeTritonTransform.FPEquiv
