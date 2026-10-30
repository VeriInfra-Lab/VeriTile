import bench.repairs.tritonbench_g.int8_matmul_quantization_i32_safe.Proofs.PublicComposed
import bench.repairs.tritonbench_g.int8_matmul_quantization_i32_safe.Proofs.ProducerLifecycle

namespace VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.Public
open VeriTile.Triton FP.Structural Direct
noncomputable section
set_option maxHeartbeats 6000000
variable {F H : Type} [Inhabited F]

def RowPost (r : RowRequest) (caller : Memory F H) (live : Nat → Prop) (alloc : RowAllocation r caller live)
    (actual : Memory F H) (bytes : Nat → Nat → Byte) (scales : Nat → H) : Prop :=
  RowViewBinding r alloc.data alloc.scales ∧
  (∀ g : Fin r.G, ∀ k : Fin r.K,actual (logicalAddress alloc.data (g.val*r.K+k.val))=.inl (bytes g k)) ∧
  (∀ g : Fin r.G,actual (logicalAddress alloc.scales g.val)=.inr (.inr (scales g))) ∧
  (∀ p,live p.1 → actual p=caller p)

local instance : Inhabited Word32 := ⟨FP.Value.zero .fp32⟩

theorem producerNative (f : KernelNativeModel.FloatingInstructions) (base : Nat → Nat) (r : RowRequest)
    (laws : TensorLaws base r.input) (caller : Memory Word32 Word16) (live : Nat → Prop)
    (inputLive : live r.input.info.storage) (alloc : RowAllocation r caller live)
    (dataLaws : TensorLaws base alloc.data) (scaleLaws : TensorLaws base alloc.scales)
    (template : State Word32) (width : Nat) (positive : 0<width) (covers : r.K≤width)
    (selection : Producer.Selection (ProducerTensor.geometry r width positive)) (rest : Algebra Word32)
    (trials : List (ProducerLifecycle.Invocation Word32 (ProducerTensor.geometry r width positive)))
    (order : Producer.Schedule (ProducerTensor.geometry r width positive)) :
    InputPointerBinding base r.input ∧ InputPointerBinding base alloc.data ∧ InputPointerBinding base alloc.scales ∧
    ∃ before final, ProducerLifecycle.trials (ProducerTensor.geometry r width positive) trials
      (ProducerTensor.bind KernelScalarProfile.pack16 r alloc.memory template)=some before ∧
      KernelProducerGrid.run (KernelScalarProfile.native f selection.tree rest) KernelNativeModel.integer8
        (ProducerTensor.geometry r width positive) order.order before=some final ∧
      RowPost r caller live alloc
        (ProducerTensor.memory KernelScalarProfile.unpack16 r alloc.data alloc.scales alloc.memory final)
        (fun g k => KernelProducerTargets.nativeByte f (ProducerTensor.geometry r width positive) selection
          (ProducerTensor.originalValues KernelScalarProfile.pack16 r caller g) k)
        (fun g => KernelProducerTargets.nativeStoredScale f (ProducerTensor.geometry r width positive) selection
          (ProducerTensor.originalValues KernelScalarProfile.pack16 r caller g)) := by
  have reads:=ProducerTensor.original_reads KernelScalarProfile.pack16 base r laws caller template width positive
  rw [Admission.producer_prepared_binding KernelScalarProfile.pack16 r caller live inputLive alloc template] at reads
  obtain ⟨before,ht,hr⟩:=ProducerLifecycle.input_reads _ _ trials _ reads
  obtain ⟨final,he,hq,hs,_⟩:=Producer.native_complete f _ selection covers rest _ before hr order
  refine ⟨input_pointer_binding base r.input laws,input_pointer_binding base alloc.data dataLaws,
    input_pointer_binding base alloc.scales scaleLaws,before,final,ht,he,alloc.binding,?_,?_,
    ProducerTensor.frame KernelScalarProfile.unpack16 r caller live alloc final⟩
  · intro g k
    exact ProducerTensor.data_readback KernelScalarProfile.unpack16 r caller live alloc final g k _
      (hq g ⟨k,lt_of_lt_of_le k.isLt covers⟩ k.isLt)
  · intro g
    simpa only [KernelScalarProfile.half_roundtrip] using
      ProducerTensor.scale_readback KernelScalarProfile.unpack16 r caller live alloc final g _ (hs g)

theorem producerReal (base : Nat → Nat) (r : RowRequest) (laws : TensorLaws base r.input)
    (caller : Memory ℝ ℝ) (live : Nat → Prop) (inputLive : live r.input.info.storage) (alloc : RowAllocation r caller live)
    (dataLaws : TensorLaws base alloc.data) (scaleLaws : TensorLaws base alloc.scales)
    (template : State ℝ) (width : Nat) (positive : 0<width) (covers : r.K≤width)
    (selection : Producer.Selection (ProducerTensor.geometry r width positive)) (rest : Algebra ℝ)
    (trials : List (ProducerLifecycle.Invocation ℝ (ProducerTensor.geometry r width positive)))
    (order : Producer.Schedule (ProducerTensor.geometry r width positive)) :
    InputPointerBinding base r.input ∧ InputPointerBinding base alloc.data ∧ InputPointerBinding base alloc.scales ∧
    ∃ before final, ProducerLifecycle.trials (ProducerTensor.geometry r width positive) trials
      (ProducerTensor.bind id r alloc.memory template)=some before ∧
      KernelProducerGrid.run (KernelScalarProfile.mathematical selection.tree rest) realInteger8
        (ProducerTensor.geometry r width positive) order.order before=some final ∧
      RowPost r caller live alloc (ProducerTensor.memory id r alloc.data alloc.scales alloc.memory final)
        (fun g k => KernelProducerTargets.mathematicalByte (ProducerTensor.originalValues id r caller g) r.K k)
        (fun g => KernelProducerTargets.mathematicalScale (ProducerTensor.originalValues id r caller g) r.K) := by
  have reads:=ProducerTensor.original_reads id base r laws caller template width positive
  rw [Admission.producer_prepared_binding id r caller live inputLive alloc template] at reads
  obtain ⟨before,ht,hr⟩:=ProducerLifecycle.input_reads _ _ trials _ reads
  obtain ⟨final,he,hq,hs,_⟩:=Producer.mathematical_complete _ selection covers rest _ before hr order
  refine ⟨input_pointer_binding base r.input laws,input_pointer_binding base alloc.data dataLaws,
    input_pointer_binding base alloc.scales scaleLaws,before,final,ht,he,alloc.binding,?_,?_,
    ProducerTensor.frame id r caller live alloc final⟩
  · intro g k
    exact ProducerTensor.data_readback id r caller live alloc final g k _
      (hq g ⟨k,lt_of_lt_of_le k.isLt covers⟩ k.isLt)
  · intro g
    exact ProducerTensor.scale_readback id r caller live alloc final g _ (hs g)

#axiomsClean producerNative
#axiomsClean producerReal
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.Public
