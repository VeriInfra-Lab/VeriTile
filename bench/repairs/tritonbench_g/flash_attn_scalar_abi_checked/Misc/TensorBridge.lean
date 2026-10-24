import bench.repairs.tritonbench_g.flash_attn_scalar_abi_checked.Correct
import bench.repairs.tritonbench_g.flash_attn_scalar_abi_checked.Misc.NativeCorrect
import bench.repairs.tritonbench_g.flash_attn_scalar_abi_checked.Misc.KernelRealResults
import bench.repairs.tritonbench_g.flash_attn_scalar_abi_checked.Contract
import bench.repairs.tritonbench_g.flash_attn.KernelOnly.TensorResult

namespace VeriTile.Bench.Repairs.FlashAttnScalarABIChecked.TensorBridge
open VeriTile Triton FP.Structural FlashAttn FlashAttn.Checked FlashAttn.ActualSource FlashAttn.KernelTensor
noncomputable section

/-- Candidate assertions derive shape and output injectivity. Complete shared
source execution gives independent Real output/L targets and full cell frames. -/
theorem whole_real (q k v o : Checked.Tensor) (qo ko vo oo : SortedAxes) (scale : ScalarABI)
    (causal : Bool) (owner : Nat→Nat→Nat→Fin 4) (scratch : Nat)
    (checked : inputChecks q k v qo ko vo scale=Bool.true) (outputChecked : tensorChecks o oo=Bool.true)
    (allocation : Allocations (arguments q k v causal scale owner) o scratch) (m : Memory ℝ) :
    let x:=Host.layout (arguments q k v causal scale owner) o.tensor
    ∃ final,runGrid KernelReal.algebra KernelReal.dot (parameters x) (some (scale.project scale.raw))
      (programs (parameters x)) (liftMemory (Real.lift m))=some final ∧
      (∀ b h row col,(Work.output b h row col).active x.shape →
        final "O" (FlashAttn.address x.o b h row col)=.mk .real (some (Spec.output x m (scale.project scale.raw) b h row col))) ∧
      (∀ b h row,(Work.logsum b h row).active x.shape →
        final "L" (FlashAttn.lAddress x.shape b h row)=.mk .real (some (Spec.lse x m (scale.project scale.raw) b h row))) ∧
      (∀ r off,¬Writes (parameters x) (programs (parameters x)) r off → final r off=liftMemory (Real.lift m) r off) ∧
      (∀ r∈(["Q","K","V"] : List RegionName),∀ off,final r off=liftMemory (Real.lift m) r off) := by
  let a:=arguments q k v causal scale owner
  have valid:=admitted_inputs q k v qo ko vo scale (original_checks q k v qo ko vo scale checked) causal owner
  exact FlashAttnScalarABIChecked.KernelRealResults.whole_real (Host.layout a o.tensor) (Host.contract_shape a valid)
    (Host.output_layout a o.tensor scratch (admitted_allocations a o scratch allocation oo outputChecked)) m (scale.project scale.raw)

theorem whole_native (q k v o : Checked.Tensor) (qo ko vo oo : SortedAxes)
    (scale : ScalarABI) (causal : Bool) (owner : Nat→Nat→Nat→Fin 4)
    (scratch : Scratch) (checked : inputChecks q k v qo ko vo scale=Bool.true)
    (outputChecked : tensorChecks o oo=Bool.true)
    (allocation : Allocations (arguments q k v causal scale owner) o scratch.allocation)
    (scratchBinding : ScratchBinding (arguments q k v causal scale owner) scratch)
    (fresh : Fresh q k v o scratch)
    (scratchChecked : scratchPointerChecks (Host.shape (arguments q k v causal scale owner)) scratch.dataPointer=Bool.true)
    (actualOutput : Torch.FP32.Tensor) (outputBinding : TensorBinding o actualOutput)
    (I : KernelNative.Instructions) (before : ByteMemory) :
    let a:=arguments q k v causal scale owner
    let x:=Host.layout a o.tensor
    let base:=pointers q k v o scratch
    Host.Contract a ∧ ∃ result,run x base I scale before=some result ∧
      KernelNative.run x scale (Physical.load (callerBytes base before))=some result.graph ∧
      (∀ b h row col,(Work.output b h row col).active (Host.shape a) →
        (Torch.FP32.logicalValue (read result.bytes) actualOutput outputBinding.laws
          (flat o.tensor.shape b h row col)).bits=
          nativeTarget x base I scale before "O" (FlashAttn.address o.tensor.strides b h row col)) ∧
      (∀ b h row,(Work.logsum b h row).active (Host.shape a) →
        readWord result.bytes (scratch.dataPointer+4*FlashAttn.lAddress (Host.shape a) b h row)=
          nativeTarget x base I scale before "L" (FlashAttn.lAddress (Host.shape a) b h row)) ∧
      (∀ location,¬StoredByte x base .o location.val → ¬StoredByte x base .l location.val →
        result.bytes location=before location) ∧
      (∀ t∈[q,k,v],∀ location,t.storageBase≤location.val →
        location.val<t.storageBase+t.tensor.storageBytes → result.bytes location=before location) := by
  dsimp only
  let a:=arguments q k v causal scale owner
  let x:=Host.layout a o.tensor
  let base:=pointers q k v o scratch
  obtain ⟨graph,hgraph,heq,values,frame⟩:=FlashAttnScalarABIChecked.KernelNativeResults.whole_native I x scale (Physical.load (callerBytes base before))
  refine ⟨FlashAttn.Checked.admitted_inputs q k v qo ko vo scale
      (original_checks q k v qo ko vo scale checked) causal owner,
    ⟨graph,finish x base I graph before⟩,?_,hgraph,?_,?_,?_,?_⟩
  · change run x base I scale before=some ⟨graph,finish x base I graph before⟩
    simp [KernelTensor.run,hgraph]
  · intro b h row col active
    have ha : (Work.output b h row col).active o.tensor.shape := by
      simpa only [Work.active,allocation.outputShape,Host.shape] using active
    rw [logical_read o actualOutput outputBinding _ b h row col ha]
    unfold readWord
    have hb:=funext (output_byte a q k v o scratch allocation oo outputChecked I graph before b h row col active)
    rw [hb,Physical.decode_encode]
    exact values _ _
  · intro b h row active
    unfold readWord
    have hb:=funext (scratch_byte a q k v o scratch allocation oo outputChecked scratchBinding fresh scratchChecked
      I graph before b h row active)
    rw [hb,Physical.decode_encode]
    exact values _ _
  · exact fun location ho hl=>finish_frame x base I graph before location ho hl
  · exact fun t ht location hlo hhi=>input_frame a q k v o scratch allocation oo outputChecked
      scratchBinding fresh I graph before t ht location hlo hhi


/-- Actual input Tensor binding, supplementary to the complete native kernel result. -/
alias input_tensor := VeriTile.Bench.Repairs.FlashAttn.KernelTensor.input_tensor

#axiomsClean input_tensor
#axiomsClean whole_real
#axiomsClean whole_native
end
end VeriTile.Bench.Repairs.FlashAttnScalarABIChecked.TensorBridge
