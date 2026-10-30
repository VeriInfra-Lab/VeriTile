import bench.tritonbench_g.int4_matmul.Proofs.Physical
import bench.tritonbench_g.int4_matmul.Proofs.Helpers
namespace VeriTile.Bench.Repairs.Int4Matmul.HelperBytes
open Physical Helpers
abbrev Half := BitVec 16

def encode16 (x : Half) (i : Fin 2) : Byte := x.extractLsb' (8*i.val) 8
def decode16 (x : Fin 2 → Byte) : Half := x 1++x 0

theorem decode_encode16 (x : Half) : decode16 (encode16 x)=x := by
  change (x.extractLsb' 8 8++x.extractLsb' 0 8)=x
  rw [BitVec.extractLsb'_append_extractLsb'_eq_extractLsb' (by decide : 8=0+8)]
  exact BitVec.extractLsb'_eq_self

structure Unpacked where
  payload : Nat → Byte
  caller : Bytes
  rows : Nat
  columns : Nat
  rowStride : Nat
  columnStride : Nat

def unpack (s : Shape) (a : HalfArithmetic Word Half) (initial : Bytes) (scratch : Nat → Byte) : Unpacked :=
  ⟨(fun b => if b/2<s.n*s.k then encode16
      (unpackLoop s a (load initial) ((b/2)%s.k) ((b/2)/s.k) (s.k/s.group))
      ⟨b%2,Nat.mod_lt _ (by decide)⟩ else scratch b),initial,s.k,s.n,1,s.k⟩

theorem unpack_bytes (s : Shape) (hs : s.Valid) (a : HalfArithmetic Word Half)
    (initial : Bytes) (scratch : Nat → Byte) (k n : Nat) (hk : k<s.k) (hn : n<s.n) (i : Fin 2) :
    (unpack s a initial scratch).payload (2*(n*s.k+k)+i.val)=
      encode16 (unpackSpec s a (load initial) k n) i := by
  have hd : (2*(n*s.k+k)+i.val)/2=n*s.k+k := by omega
  have hm : (2*(n*s.k+k)+i.val)%2=i.val := by omega
  have hbound := unpack_address s k n hk hn
  simp only [unpack,hd,hbound,ite_true,div_coordinate n s.k k hs.kpos hk,
    Nat.mul_add_mod_self_right,Nat.mod_eq_of_lt hk,whole_unpack_loop s hs a (load initial) k n hk]
  congr 1
  exact Fin.ext hm

theorem unpack_caller_frame (s : Shape) (a : HalfArithmetic Word Half) (initial : Bytes)
    (scratch : Nat → Byte) : (unpack s a initial scratch).caller=initial := rfl

theorem unpack_scratch_frame (s : Shape) (a : HalfArithmetic Word Half) (initial : Bytes)
    (scratch : Nat → Byte) (b : Nat) (hb : s.n*s.k≤b/2) :
    (unpack s a initial scratch).payload b=scratch b := by simp [unpack,not_lt_of_ge hb]

theorem unpack_geometry (s : Shape) (a : HalfArithmetic Word Half) (initial : Bytes) (scratch : Nat → Byte) :
    (unpack s a initial scratch).rows=s.k ∧ (unpack s a initial scratch).columns=s.n ∧
    (unpack s a initial scratch).rowStride=1 ∧ (unpack s a initial scratch).columnStride=s.k := ⟨rfl,rfl,rfl,rfl⟩
def serialize32 (count : Nat) (values : Nat → Word) (scratch : Nat → Byte) :=
  fun b => if b/4<count then encode (values (b/4)) ⟨b%4,Nat.mod_lt _ (by decide)⟩ else scratch b

theorem serialize32_bytes (count : Nat) (values : Nat → Word) (scratch : Nat → Byte)
    (a : Nat) (ha : a<count) (i : Fin 4) : serialize32 count values scratch (4*a+i.val)=encode (values a) i := by
  have hd : (4*a+i.val)/4=a := by omega
  have hm : (4*a+i.val)%4=i.val := by omega
  simp only [serialize32,hd,ha,ite_true]
  congr 1;exact Fin.ext hm

structure QuantizedBytes where
  weights : Nat → Byte
  scales : Nat → Byte
  zeros : Nat → Byte
  caller : Bytes
  weightShape : Nat × Nat
  scaleShape : Nat × Nat
  zeroShape : Nat × Nat
  groupSize : Nat

def quantizeBytes (s : Shape) (a : Arithmetic Word) (order : ReductionOrder s.group)
    (initial : Bytes) (scratch : Fin 3 → Nat → Byte) : QuantizedBytes :=
  let q := quantize s a order (load initial)
  ⟨serialize32 ((s.k/8)*s.n) q.weights (scratch 0),
   serialize32 ((s.k/s.group)*s.n) q.scales (scratch 1),
   serialize32 ((s.k/s.group)*(s.n/8)) q.zeros (scratch 2),initial,
   q.weightShape,q.scaleShape,q.zeroShape,q.groupSize⟩

theorem quantize_weight_bytes (s : Shape) (hs : s.Valid) (a : Arithmetic Word) (order : ReductionOrder s.group)
    (initial : Bytes) (scratch : Fin 3 → Nat → Byte) (p n : Nat) (hp : p<s.k/8) (hn : n<s.n) (i : Fin 4) :
    (quantizeBytes s a order initial scratch).weights (4*(p*s.n+n)+i.val)=
      encode (weightSpec s a order (load initial) p n) i := by
  simp only [quantizeBytes]
  rw [serialize32_bytes _ _ _ _ (rowMajor_bound _ _ _ _ hp hn),quantize_result s hs a order (load initial) p n hp hn]

theorem quantize_scale_bytes (s : Shape) (hs : s.Valid) (a : Arithmetic Word) (order : ReductionOrder s.group)
    (initial : Bytes) (scratch : Fin 3 → Nat → Byte) (g n : Nat) (hg : g<s.k/s.group) (hn : n<s.n) (i : Fin 4) :
    (quantizeBytes s a order initial scratch).scales (4*(g*s.n+n)+i.val)=
      encode (scale s a order (load initial) g n) i := by
  simp only [quantizeBytes]
  rw [serialize32_bytes _ _ _ _ (rowMajor_bound _ _ _ _ hg hn),quantize_scales s hs a order (load initial) g n hn]

theorem quantize_zero_bytes (s : Shape) (hs : s.Valid) (a : Arithmetic Word) (order : ReductionOrder s.group)
    (initial : Bytes) (scratch : Fin 3 → Nat → Byte) (g p : Nat) (hg : g<s.k/s.group) (hp : p<s.n/8) (i : Fin 4) :
    (quantizeBytes s a order initial scratch).zeros (4*(g*(s.n/8)+p)+i.val)=
      encode (zeroSpec s a order (load initial) g p) i := by
  simp only [quantizeBytes]
  rw [serialize32_bytes _ _ _ _ (rowMajor_bound _ _ _ _ hg hp),quantize_zeros s hs a order (load initial) g p hp]

theorem quantize_caller_frame (s : Shape) (a : Arithmetic Word) (order : ReductionOrder s.group)
    (initial : Bytes) (scratch : Fin 3 → Nat → Byte) :
    (quantizeBytes s a order initial scratch).caller=initial := rfl

#axiomsClean quantize_weight_bytes
#axiomsClean quantize_scale_bytes
#axiomsClean quantize_zero_bytes
#axiomsClean quantize_caller_frame
#axiomsClean decode_encode16
#axiomsClean unpack_bytes
#axiomsClean unpack_caller_frame
#axiomsClean unpack_scratch_frame
#axiomsClean unpack_geometry
end VeriTile.Bench.Repairs.Int4Matmul.HelperBytes
