import bench.tritonbench_g.matmul_dequant_int4.Sources.Host
import bench.tritonbench_g.matmul_dequant_int4.Sources.HelperReal
namespace VeriTile.Bench.Repairs.MatmulDequantInt4.HelperHost
open Host Helpers Physical HelperBytes
set_option maxHeartbeats 4000000
noncomputable section

def finite16 (w : Half) : Prop := ((w >>> 10) &&& (31:Half)).toNat≠31
def positive16 (w : Half) : Prop := w.toNat<2^15 ∧ w.toNat≠0 ∧ finite16 w
def finite32 (w : Word) : Prop := ((w >>> 23) &&& (255:Word)).toNat≠255
def positive32 (w : Word) : Prop := w.toNat<2^31 ∧ w.toNat≠0 ∧ finite32 w

structure QuantDomain16 (s : Shape) (p : Native.Instructions) (order : ReductionOrder s.group)
    (initial : Bytes) (weight : Tensor) (tpRank : Nat) : Prop where
  shape : s.Valid
  layout : weight.Ordinary .fp16 s.k s.n weight.device
  rank : tpRank=weight.device
  indices : s.k*s.n<2^31
  finite : ∀ k n,k<s.k → n<s.n → finite16 ((load halfCodec initial).values .weight (k*s.n+n))
  positiveScale : ∀ g n,g<s.k/s.group → n<s.n →
    positive16 (scale s (Native.quant16 p) order (load halfCodec initial) g n)
structure QuantDomain32 (s : Shape) (p : Native.Instructions) (order : ReductionOrder s.group)
    (initial : Bytes) (weight : Tensor) (tpRank : Nat) : Prop where
  shape : s.Valid
  layout : weight.Ordinary .fp32 s.k s.n weight.device
  rank : tpRank=weight.device
  indices : s.k*s.n<2^31
  finite : ∀ k n,k<s.k → n<s.n → finite32 ((load wordCodec initial).values .weight (k*s.n+n))
  positiveScale : ∀ g n,g<s.k/s.group → n<s.n →
    positive32 (scale s (Native.quant32 p) order (load wordCodec initial) g n)

/-- Output descriptors use the actual transpose/contiguous singleton layouts.
The quantizer preserves scale dtype from the selected input profile. -/
structure QuantAllocation (s : Shape) (dtype : DType) (input : Tensor) where
  weight : Tensor
  scale : Tensor
  zero : Tensor
  weightLayout : weight.QuantLayout .int32 (s.k/8) s.n input.device
  scaleLayout : scale.QuantLayout dtype (s.k/s.group) s.n input.device
  zeroLayout : zero.QuantLayout .int32 (s.k/s.group) (s.n/8) input.device
  callerDisjoint : ∀ t∈[weight,scale,zero],Disjoint input t
  independent : Independent [weight,scale,zero]
def QuantAllocation.packed {s : Shape} {dtype : DType} {input : Tensor}
    (a : QuantAllocation s dtype input) : PackedArguments :=
  ⟨s,a.weight,a.scale,a.zero⟩
theorem quantizer_consumer_layout (s : Shape) (dtype : DType) (input : Tensor)
    (a : QuantAllocation s dtype input) (selected : dtype=.fp16 ∨ dtype=.fp32)
    (hs : s.Valid) (indices : Strides.Guard a.packed.strides s) :
    PackedContract a.packed dtype := by
  have dev : a.weight.device=input.device := a.weightLayout.1.2.2.2.2.2.2.1
  refine ⟨selected,hs,?_,?_,?_,a.independent,indices⟩
  · simpa [QuantAllocation.packed,dev] using a.weightLayout.1
  · simpa [QuantAllocation.packed,dev] using a.scaleLayout.1
  · simpa [QuantAllocation.packed,dev] using a.zeroLayout.1

structure QuantReturn where
  weight : Tensor
  scale : Tensor
  zero : Tensor
  group : Nat
  data : QuantizedBytes
  callerDevice : Nat
def quant16Call (s : Shape) (p : Native.Instructions) (order : ReductionOrder s.group)
    (initial : Bytes) (scratch : Fin 3 → Nat → Byte) (input : Tensor)
    (allocation : QuantAllocation s .fp16 input) (caller : Nat) : QuantReturn :=
  ⟨allocation.weight,allocation.scale,allocation.zero,s.group,
    quantizeBytes halfCodec s (Native.quant16 p) order initial scratch,caller⟩
def quant32Call (s : Shape) (p : Native.Instructions) (order : ReductionOrder s.group)
    (initial : Bytes) (scratch : Fin 3 → Nat → Byte) (input : Tensor)
    (allocation : QuantAllocation s .fp32 input) (caller : Nat) : QuantReturn :=
  ⟨allocation.weight,allocation.scale,allocation.zero,s.group,
    quantizeBytes wordCodec s (Native.quant32 p) order initial scratch,caller⟩
theorem quant16_metadata (s : Shape) (p : Native.Instructions) (order : ReductionOrder s.group)
    (initial : Bytes) (scratch : Fin 3 → Nat → Byte) (input : Tensor)
    (a : QuantAllocation s .fp16 input) (caller : Nat) :
    (quant16Call s p order initial scratch input a caller).weight=a.weight ∧
    (quant16Call s p order initial scratch input a caller).scale=a.scale ∧
    (quant16Call s p order initial scratch input a caller).zero=a.zero ∧
    (quant16Call s p order initial scratch input a caller).group=s.group ∧
    (quant16Call s p order initial scratch input a caller).callerDevice=caller := ⟨rfl,rfl,rfl,rfl,rfl⟩
theorem quant32_metadata (s : Shape) (p : Native.Instructions) (order : ReductionOrder s.group)
    (initial : Bytes) (scratch : Fin 3 → Nat → Byte) (input : Tensor)
    (a : QuantAllocation s .fp32 input) (caller : Nat) :
    (quant32Call s p order initial scratch input a caller).weight=a.weight ∧
    (quant32Call s p order initial scratch input a caller).scale=a.scale ∧
    (quant32Call s p order initial scratch input a caller).zero=a.zero ∧
    (quant32Call s p order initial scratch input a caller).group=s.group ∧
    (quant32Call s p order initial scratch input a caller).callerDevice=caller := ⟨rfl,rfl,rfl,rfl,rfl⟩

theorem helper_input_indices (s : Shape) (guard : s.k*s.n<2^31) (k n : Nat)
    (hk : k<s.k) (hn : n<s.n) : k*s.n+n<2^31 ∧ n*s.k+k<2^31 := by
  constructor
  · exact (rowMajor_bound _ _ _ _ hk hn).trans guard
  · have h := rowMajor_bound _ _ _ _ hn hk
    rw [Nat.mul_comm s.n s.k] at h;exact h.trans guard
theorem helper_pack_shift (i : Fin 8) : i.val*4<32 := by omega
theorem positive16_one : positive16 0x3c00 := by unfold positive16 finite16;decide +kernel
theorem positive32_one : positive32 0x3f800000 := by unfold positive32 finite32;decide +kernel
theorem zero_rejected : ¬positive16 0 ∧ ¬positive32 0 := by unfold positive16 positive32 finite16 finite32;decide +kernel
theorem infinity_rejected : ¬finite16 0x7c00 ∧ ¬finite32 0x7f800000 := by unfold finite16 finite32;decide +kernel

#axiomsClean quantizer_consumer_layout
#axiomsClean quant16_metadata
#axiomsClean quant32_metadata
#axiomsClean helper_input_indices
#axiomsClean helper_pack_shift
#axiomsClean positive16_one
#axiomsClean positive32_one
#axiomsClean zero_rejected
#axiomsClean infinity_rejected
end
end VeriTile.Bench.Repairs.MatmulDequantInt4.HelperHost
