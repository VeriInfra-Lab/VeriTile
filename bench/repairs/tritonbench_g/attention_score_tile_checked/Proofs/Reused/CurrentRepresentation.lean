import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.Physical
import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.Widened

namespace VeriTile.Bench.Repairs.AttentionScoreTileChecked.Reused.Whole002.Representation
open VeriTile Triton Geometry Physical
noncomputable section
set_option maxHeartbeats 1000000

def i32max : Nat := 2147483647
def i64max : Nat := 9223372036854775807
def Signed32 (x : Int) : Prop := -2147483648  ≤  x ∧ x  ≤  2147483647
def Signed64 (x : Int) : Prop := -9223372036854775808  ≤  x ∧ x  ≤  9223372036854775807

def qPad (c : Shape) : Nat := RealLoop.tripCount c.N c.B * c.B
def kPad (c : Shape) : Nat := c.programs * c.B

def metadata (c : Shape) : List Nat :=
  [c.Z,c.H,c.HK,c.N,c.R,c.NK,c.D,c.H*c.N*c.D,c.N*c.D,
   c.HK*c.NK*c.D,c.NK*c.D,c.H*c.NK]

/-- End-exclusive formation extents. Q and M include the final advance and
its vector lanes, K and O include masked tail lanes. M also covers its allocation. -/
def extent (c : Shape) (r : RegionName) : Nat :=
  if r="q" then (c.Z*c.H-1)*c.N*c.D+(qPad c+c.B)*c.D
  else if r="k" then (c.Z*c.HK-1)*c.NK*c.D+kPad c*c.D
  else if r="m" then max ((c.Z*c.H-1)*c.R+qPad c+c.B) c.normalizerSize
  else if r="out" then (c.Z*c.H-1)*c.NK+kPad c else 0

/-- Exactly the nonempty-launch checks in the successor wrapper. The signed32
metadata bound fixes the selected runtime scalar ABI; D has no ad hoc resource
cap, but block_shape and arange must be representable. -/
structure Launch (c : Shape) : Prop where
  params : ∀ x ∈ metadata c, x  ≤  i32max
  paddedQuery : qPad c+c.B  ≤  i32max
  paddedKey : kPad c  ≤  i32max
  gridX : c.programs  ≤  i32max
  gridY : c.Z*c.H  ≤  65535
  bytes : ∀ r, tracked r → 4*extent c r  ≤  i64max

/-- Finite FP32 word storage, with a checked four-byte address range. This is
not a raw-byte endian/decode theorem. Empty return does not require a launch. -/
structure Domain (c : Shape) (abi : ABI c) (offset size : Int) (window : Bool) : Prop where
  scalars : window=Bool.true → Signed32 offset ∧ Signed32 size
  launch : c.N ≠ 0 → c.outputSize ≠ 0 → Launch c
  addresses : c.N ≠ 0 → c.outputSize ≠ 0 → ∀ r, tracked r →
    4*((abi.alloc r).base+extent c r) < 18446744073709551616

/-- Existing launch admission already bounds the retained source int32 cast. -/
theorem Domain.key_offsets_fit (c : Shape) (abi : ABI c) (offset size : Int) (window : Bool)
    (domain : Domain c abi offset size window) :
    c.N ≠ 0 → c.outputSize ≠ 0 → KeyOffsetsFit c := by
  intro hN hOut block
  have bound := (domain.launch hN hOut).paddedKey
  have blockBound := Nat.mul_le_mul_right c.B (Nat.le_of_lt block.isLt)
  dsimp [kPad, i32max] at bound
  omega

theorem ofInt64_exact (x : Int) (h : Signed64 x) : (BitVec.ofInt 64 x).toInt=x := by
  exact BitVec.toInt_ofInt_eq_self (by decide) h.1 (by dsimp [Signed64] at h; norm_num; omega)

theorem ofInt32_exact (x : Int) (h : Signed32 x) : (BitVec.ofInt 32 x).toInt=x := by
  exact BitVec.toInt_ofInt_eq_self (by decide) h.1 (by dsimp [Signed32] at h; norm_num; omega)

theorem natural32 (x : Nat) (h : x ≤ i32max) : Signed32 (x : Int) := by
  dsimp [Signed32,i32max] at *; omega

theorem widen32_exact (x : Int) (hx : Signed32 x) : (BitVec.ofInt 64 x).toInt=x :=
  ofInt64_exact x (by dsimp [Signed64,Signed32] at *; omega)

theorem metadata_exact (c : Shape) (h : Launch c) (x : Nat) (hx : x∈metadata c) :
    (BitVec.ofInt 32 (x : Int)).toInt=(x : Int) := ofInt32_exact _ (natural32 x (h.params x hx))

theorem supported_le (c : Shape) : c.B ≤ 64 := by
  rcases c.block with h|h|h  <;> omega

theorem padded_coordinates (c : Shape) (hc : Launch c)
    (t : Nat) (ht : t ≤ RealLoop.tripCount c.N c.B) (i : Fin c.B)
    (block : Fin c.programs) (j : Fin c.B) :
    t*c.B+i.val < i32max ∧ block.val*c.B+j.val < i32max ∧
    t*c.B ≤ i32max ∧ block.val*c.B ≤ i32max := by
  have hq := Nat.mul_le_mul_right c.B ht
  have hk := digit_bound block.val c.programs j.val c.B block.isLt j.isLt
  have hp := hc.paddedQuery
  have hpk := hc.paddedKey
  dsimp [qPad,kPad] at *
  omega

/-- Every left-associated signed window intermediate fits int64, including
inactive padded query/key lanes. Offset and size retain the explicit i32 API. -/
theorem distance_intermediates (c : Shape) (hc : Launch c)
    (offset : Int) (hoff : Signed32 offset) (start : Nat)
    (hs : start ≤ qPad c) (block : Fin c.programs) (i j : Fin c.B) :
    Signed64 ((i.val : Int)-(j.val : Int)) ∧
    Signed64 ((i.val : Int)-(j.val : Int)+(start : Int)) ∧
    Signed64 ((i.val : Int)-(j.val : Int)+(start : Int)-(block.val*c.B : Nat)) ∧
    Signed64 ((i.val : Int)-(j.val : Int)+(start : Int)-(block.val*c.B : Nat)+offset) := by
  have hb := supported_le c
  have hq := hc.paddedQuery
  have hk := hc.paddedKey
  have hn := Nat.mul_le_mul_right c.B (Nat.le_of_lt block.isLt)
  dsimp [Signed64,Signed32,i32max,kPad] at *
  omega

/-- Fixed-width expression in the same source operand/association order. -/
def distance64 (i j start keyBase : Nat) (offset : Int) : BitVec 64 :=
  BitVec.ofInt 64 i - BitVec.ofInt 64 j + BitVec.ofInt 64 start -
    BitVec.ofInt 64 keyBase + BitVec.ofInt 64 offset

theorem ofInt_sub (x y : Int) : BitVec.ofInt 64 (x-y)=BitVec.ofInt 64 x-BitVec.ofInt 64 y := by
  apply BitVec.eq_of_toInt_eq
  simp

theorem distance64_exact (c : Shape) (hc : Launch c) (offset : Int)
    (hoff : Signed32 offset) (start : Nat) (hs : start ≤ qPad c)
    (block : Fin c.programs) (i j : Fin c.B) :
    (distance64 i.val j.val start (block.val*c.B) offset).toInt=
      (i.val : Int)-(j.val : Int)+(start : Int)-(block.val*c.B : Nat)+offset := by
  unfold distance64
  rw [← ofInt_sub, ← BitVec.ofInt_add, ← ofInt_sub, ← BitVec.ofInt_add]
  exact ofInt64_exact _ (distance_intermediates c hc offset hoff start hs block i j).2.2.2

theorem window_equivalence (c : Shape) (hc : Launch c) (offset size : Int)
    (hoff : Signed32 offset) (hsize : Signed32 size) (start : Nat) (hs : start ≤ qPad c)
    (block : Fin c.programs) (i j : Fin c.B) (complement : Bool) :
    (if complement then
      decide ((BitVec.ofInt 64 size).toInt ≤ (distance64 i.val j.val start (block.val*c.B) offset).toInt)
     else decide (0 ≤ (distance64 i.val j.val start (block.val*c.B) offset).toInt ∧
       (distance64 i.val j.val start (block.val*c.B) offset).toInt < (BitVec.ofInt 64 size).toInt)) =
    decide (RealFormula.windowAllows Bool.true complement offset size (start+i.val) (block.val*c.B+j.val)) := by
  rw [widen32_exact size hsize]
  rw [distance64_exact c hc offset hoff start hs block i j]
  have he : (i.val : Int)-(j.val : Int)+(start : Int)-(block.val*c.B : Nat)+offset =
      RealFormula.distance (start+i.val) (block.val*c.B+j.val) offset := by
    simp [RealFormula.distance]
    omega
  rw [he]
  cases complement <;> simp [RealFormula.windowAllows]

theorem padded_linear (head heads rows row pad dim d : Nat)
    (hh : head < heads) (hr : row < pad) (hd : d < dim) :
    head*rows*dim+row*dim+d < (heads-1)*rows*dim+pad*dim := by
  have hhead : head≤heads-1 := by omega
  have hb := Nat.mul_le_mul_right dim (Nat.mul_le_mul_right rows hhead)
  have hv := digit_bound row pad d dim hr hd
  omega

theorem source_addresses (c : Shape) (z : Fin c.Z) (h : Fin c.H)
    (row : Nat) (hr : row < qPad c+c.B) (key : Nat) (hk : key < kPad c) (d : Fin c.D) :
    c.queryBase z.val h.val+row*c.D+d.val < extent c "q" ∧
    c.keyBase z.val h.val+key*c.D+d.val < extent c "k" ∧
    c.normalizerBase z.val h.val+row < extent c "m" ∧
    c.outputBase z.val h.val+key < extent c "out" := by
  have hzh := digit_bound z.val c.Z h.val c.H z.isLt h.isLt
  have hzk := digit_bound z.val c.Z (h.val/(c.H/c.HK)) c.HK z.isLt (grouped_head c h)
  have hq := padded_linear _ _ c.N row _ c.D d.val hzh hr d.isLt
  have hkey := padded_linear _ _ c.NK key _ c.D d.val hzk hk d.isLt
  have hhead : z.val*c.H+h.val≤c.Z*c.H-1 := by omega
  have hm := Nat.mul_le_mul_right c.R hhead
  have ho := Nat.mul_le_mul_right c.NK hhead
  simp only [extent,Shape.queryBase,Shape.keyBase,Shape.normalizerBase,Shape.outputBase]
  simp only [show ("q" : RegionName)≠"k" by decide,
    show ("k" : RegionName)≠"q" by decide,
    show ("m" : RegionName)≠"q" by decide,
    show ("m" : RegionName)≠"k" by decide,
    show ("out" : RegionName)≠"q" by decide,
    show ("out" : RegionName)≠"k" by decide,
    show ("out" : RegionName)≠"m" by decide, ite_true, ite_false]
  refine ⟨hq,hkey,?_,?_⟩
  · have hx := Nat.le_max_left ((c.Z*c.H-1)*c.R+qPad c+c.B) c.normalizerSize
    omega
  · omega

/-- A source formation bound implies both signed64 element/byte displacement
and the checked unsigned64 physical byte-address range. -/
theorem formed_address (c : Shape) (abi : ABI c) (hc : Launch c)
    (ha : ∀ r, tracked r → 4*((abi.alloc r).base+extent c r) < 18446744073709551616)
    (r : RegionName) (hr : tracked r) (a : Nat) (hb : a < extent c r) :
    Signed64 (a : Int) ∧ Signed64 (4*(a : Int)) ∧
      4*((abi.alloc r).base+a) < 18446744073709551616 := by
  have he := hc.bytes r hr
  have hp := ha r hr
  dsimp [Signed64,i64max] at *
  omega

/-- The historical small input is preserved as a concrete bit-vector witness. -/
theorem F1_old_wrap : (BitVec.ofInt 32 2147483647+BitVec.ofInt 32 1).toInt = -2147483648 := by decide

theorem F1_widened : (distance64 1 0 0 0 2147483647).toInt=2147483648 := by decide

#axiomsClean distance64_exact
#axiomsClean window_equivalence
#axiomsClean formed_address
#axiomsClean F1_old_wrap
#axiomsClean F1_widened
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttentionScoreTileChecked.Reused.Whole002.Representation
