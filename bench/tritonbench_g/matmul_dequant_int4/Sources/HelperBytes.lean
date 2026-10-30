import bench.tritonbench_g.matmul_dequant_int4.Sources.Physical
import bench.tritonbench_g.matmul_dequant_int4.Sources.Helpers
namespace VeriTile.Bench.Repairs.MatmulDequantInt4.HelperBytes
open Physical Helpers
set_option maxHeartbeats 4000000

def serialize (codec : Codec α) (count : Nat) (values : Nat → α) (scratch : Nat → Byte) :=
  fun b => if b/codec.width<count then
    codec.encode (values (b/codec.width)) ⟨b%codec.width,Nat.mod_lt _ codec.positive⟩ else scratch b
theorem serialize_bytes (codec : Codec α) (count : Nat) (values : Nat → α) (scratch : Nat → Byte)
    (a : Nat) (ha : a<count) (i : Fin codec.width) :
    serialize codec count values scratch (codec.width*a+i.val)=codec.encode (values a) i := by
  have hd : (codec.width*a+i.val)/codec.width=a := by
    rw [Nat.mul_comm codec.width a]
    exact div_coordinate _ _ _ codec.positive i.isLt
  have hm : (codec.width*a+i.val)%codec.width=i.val := by simp [Nat.add_mod,Nat.mod_eq_of_lt i.isLt]
  simp only [serialize,hd,ha,ite_true]
  congr 1;exact Fin.ext hm
theorem serialize_frame (codec : Codec α) (count : Nat) (values : Nat → α) (scratch : Nat → Byte)
    (b : Nat) (hb : count≤b/codec.width) : serialize codec count values scratch b=scratch b := by
  simp [serialize,not_lt_of_ge hb]

structure QuantizedBytes where
  weights : Nat → Byte
  scales : Nat → Byte
  zeros : Nat → Byte
  caller : Bytes
  weightShape : Nat × Nat
  scaleShape : Nat × Nat
  zeroShape : Nat × Nat
  groupSize : Nat

/-- Source-specific quantizer results serialized into ordinary fresh Tensor
allocations. This is a payload codec under Torch copy/slice/allocation semantics,
not an interpreter of internal Torch machine writes. Q/Z are always int32;
the scale/input codec is separately FP16 or FP32. -/
def quantizeBytes (codec : Codec α) (s : Shape) (a : Arithmetic α) (order : ReductionOrder s.group)
    (initial : Bytes) (scratch : Fin 3 → Nat → Byte) : QuantizedBytes :=
  let q := quantize s a order (load codec initial)
  ⟨serialize wordCodec ((s.k/8)*s.n) q.weights (scratch 0),
   serialize codec ((s.k/s.group)*s.n) q.scales (scratch 1),
   serialize wordCodec ((s.k/s.group)*(s.n/8)) q.zeros (scratch 2),initial,
   q.weightShape,q.scaleShape,q.zeroShape,q.groupSize⟩

theorem weight_bytes (codec : Codec α) (s : Shape) (hs : s.Valid) (a : Arithmetic α)
    (order : ReductionOrder s.group) (initial : Bytes) (scratch : Fin 3 → Nat → Byte)
    (p n : Nat) (hp : p<s.k/8) (hn : n<s.n) (i : Fin 4) :
    (quantizeBytes codec s a order initial scratch).weights (4*(p*s.n+n)+i.val)=
      encode32 (weightSpec s a order (load codec initial) p n) i := by
  change serialize wordCodec _ _ _ (wordCodec.width*(p*s.n+n)+i.val)=_
  rw [serialize_bytes _ _ _ _ _ (rowMajor_bound _ _ _ _ hp hn),quantize_result s hs a order (load codec initial) p n hp hn]
  rfl
theorem scale_bytes (codec : Codec α) (s : Shape) (hs : s.Valid) (a : Arithmetic α)
    (order : ReductionOrder s.group) (initial : Bytes) (scratch : Fin 3 → Nat → Byte)
    (g n : Nat) (hg : g<s.k/s.group) (hn : n<s.n) (i : Fin codec.width) :
    (quantizeBytes codec s a order initial scratch).scales (codec.width*(g*s.n+n)+i.val)=
      codec.encode (scale s a order (load codec initial) g n) i := by
  change serialize codec _ _ _ (codec.width*(g*s.n+n)+i.val)=_
  rw [serialize_bytes _ _ _ _ _ (rowMajor_bound _ _ _ _ hg hn),quantize_scales s hs a order (load codec initial) g n hn]
theorem zero_bytes (codec : Codec α) (s : Shape) (hs : s.Valid) (a : Arithmetic α)
    (order : ReductionOrder s.group) (initial : Bytes) (scratch : Fin 3 → Nat → Byte)
    (g p : Nat) (hg : g<s.k/s.group) (hp : p<s.n/8) (i : Fin 4) :
    (quantizeBytes codec s a order initial scratch).zeros (4*(g*(s.n/8)+p)+i.val)=
      encode32 (zeroSpec s a order (load codec initial) g p) i := by
  change serialize wordCodec _ _ _ (wordCodec.width*(g*(s.n/8)+p)+i.val)=_
  rw [serialize_bytes _ _ _ _ _ (rowMajor_bound _ _ _ _ hg hp),quantize_zeros s hs a order (load codec initial) g p hp]
  rfl
theorem caller_frame (codec : Codec α) (s : Shape) (a : Arithmetic α)
    (order : ReductionOrder s.group) (initial : Bytes) (scratch : Fin 3 → Nat → Byte) :
    (quantizeBytes codec s a order initial scratch).caller=initial := rfl
theorem geometry (codec : Codec α) (s : Shape) (a : Arithmetic α)
    (order : ReductionOrder s.group) (initial : Bytes) (scratch : Fin 3 → Nat → Byte) :
    let result := quantizeBytes codec s a order initial scratch
    result.weightShape=(s.k/8,s.n) ∧ result.scaleShape=(s.k/s.group,s.n) ∧
    result.zeroShape=(s.k/s.group,s.n/8) ∧ result.groupSize=s.group := ⟨rfl,rfl,rfl,rfl⟩

#axiomsClean serialize_bytes
#axiomsClean serialize_frame
#axiomsClean weight_bytes
#axiomsClean scale_bytes
#axiomsClean zero_bytes
#axiomsClean caller_frame
#axiomsClean geometry
end VeriTile.Bench.Repairs.MatmulDequantInt4.HelperBytes
