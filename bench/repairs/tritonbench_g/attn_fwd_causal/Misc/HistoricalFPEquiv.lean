import bench.repairs.tritonbench_g.attn_fwd_causal.Proofs.EmptyReturn
import bench.repairs.tritonbench_g.attn_fwd_causal.Proofs.AcceptedFP
import bench.repairs.tritonbench_g.attn_fwd_causal.Proofs.Contract
import bench.repairs.tritonbench_g.attn_fwd_causal.Proofs.NativeOperands
namespace VeriTile.Bench.Repairs.AttnFwdCausal.FPEquiv
open VeriTile Triton FP.Structural Source Whole Native
noncomputable section
specification causal_native (I : Instructions) (p : Parameters) (observed : Contract.Observations p) (admitted : Contract.InputContract p observed) (storage : TensorStorage p)
    (abi : InputABI p) (before : State Word) (alloc : Allocation p abi before)
    (inputs : RawInputs p before) (api : Contract.StorageAPI p storage observed) (profile : Profile)
    (compiler : CompilerBinding p storage (Contract.placement p storage observed api admitted) profile)
    (flat : Fin (Contract.placement p storage observed api admitted).arena → BitVec 8) (binding : WordBinding p storage (Contract.placement p storage observed api admitted) flat alloc.state)
    (work : List Program) (legal : (programs (grid p)).Perm work) :
    Spec.Real (∃ result,invoke (fun _=>algebra) (fun _=>dotModel) p abi (injectState before)
        (injectAllocation p abi before alloc) work=some result ∧ result.tensor=alloc.output ∧
      result.tensor.shape=[p.batches,p.heads,p.length,128] ∧
      result.tensor.dtype=.fp16 ∧ result.tensor.device=abi.query.device ∧
      (∀ r (hr : r≠.output) off (ho : off<extent storage r),
        alloc.state.mem (region p r) off=.mk .real
          (read32 p storage (Contract.placement p storage observed api admitted) flat r hr off ho)) ∧
      WordResult I p alloc.state (projectState I p alloc.state result.state) ∧
      ByteResult I p storage (Contract.placement p storage observed api admitted) alloc.state flat
        (flattenBytes p storage (Contract.placement p storage observed api admitted) (fun _=>encode16) flat (projectState I p alloc.state result.state))) := by
  obtain ⟨result,he,ht,hw,hb⟩ := Native.host_native I p (Contract.guard p observed admitted) storage abi before alloc inputs (Contract.placement p storage observed api admitted) profile compiler flat binding work legal
  refine ⟨result,he,ht,?_,?_,?_,?_,hw,hb⟩
  · rw [ht,alloc.outputShape,abi.shapes.1]
  · rw [ht];exact alloc.outputType
  · rw [ht];exact alloc.outputDevice
  · intro r hr off ho
    exact input_from_current_bytes p storage _ flat alloc.state binding
      (allocation_raw p abi before alloc inputs) r hr off ho

#axiomsClean causal_native
specification causal_native_if_local (actual standard : Instructions)
    (agreement : InstructionAgreement actual standard)
    (p : Parameters) (observed : Contract.Observations p) (admitted : Contract.InputContract p observed) (storage : TensorStorage p)
    (abi : InputABI p) (before : State Word) (alloc : Allocation p abi before)
    (inputs : RawInputs p before) (api : Contract.StorageAPI p storage observed) (profile : Profile)
    (compiler : CompilerBinding p storage (Contract.placement p storage observed api admitted) profile)
    (flat : Fin (Contract.placement p storage observed api admitted).arena → BitVec 8) (binding : WordBinding p storage (Contract.placement p storage observed api admitted) flat alloc.state)
    (work : List Program) (legal : (programs (grid p)).Perm work) :
    Spec.Real (∃ result,invoke (fun _=>algebra) (fun _=>dotModel) p abi (injectState before)
        (injectAllocation p abi before alloc) work=some result ∧ result.tensor=alloc.output ∧
      result.tensor.shape=[p.batches,p.heads,p.length,128] ∧
      result.tensor.dtype=.fp16 ∧ result.tensor.device=abi.query.device ∧
      (∀ r (hr : r≠.output) off (ho : off<extent storage r),
        alloc.state.mem (region p r) off=.mk .real
          (read32 p storage (Contract.placement p storage observed api admitted) flat r hr off ho)) ∧
      WordResult standard p alloc.state (projectState actual p alloc.state result.state) ∧
      ByteResult standard p storage (Contract.placement p storage observed api admitted) alloc.state flat
        (flattenBytes p storage (Contract.placement p storage observed api admitted) (fun _=>encode16) flat (projectState actual p alloc.state result.state))) := by
  obtain ⟨result,he,ht,hw,hb⟩ := Native.fpequiv actual standard agreement p (Contract.guard p observed admitted) storage abi before alloc inputs (Contract.placement p storage observed api admitted) profile compiler flat binding work legal
  refine ⟨result,he,ht,?_,?_,?_,?_,hw,hb⟩
  · rw [ht,alloc.outputShape,abi.shapes.1]
  · rw [ht];exact alloc.outputType
  · rw [ht];exact alloc.outputDevice
  · intro r hr off ho
    exact input_from_current_bytes p storage _ flat alloc.state binding
      (allocation_raw p abi before alloc inputs) r hr off ho

#axiomsClean causal_native_if_local
end
end VeriTile.Bench.Repairs.AttnFwdCausal.FPEquiv
