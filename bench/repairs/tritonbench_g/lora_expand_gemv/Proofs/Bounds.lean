import bench.repairs.tritonbench_g.lora_expand_gemv.Proofs.Geometry
namespace VeriTile.Bench.Repairs.LoraExpandGemv.Bounds
open VeriTile.Triton
noncomputable section
abbrev I32 := BitVec 32
def word (n : Nat) : I32 := BitVec.ofNat 32 n

theorem signed_word (n : Nat) (h : n<2^31) : (word n).toInt=(n : Int) := by
  rw [show word n=BitVec.ofInt 32 (n : Int) from (BitVec.ofInt_natCast _ _).symm]
  apply BitVec.toInt_ofInt_eq_self (by decide)
  · norm_num
  · exact_mod_cast h

theorem sdiv_word (x y : Nat) (hx : x<2^31) (hy : y<2^31) :
    (word x).sdiv (word y)=word (x/y) := by
  have nx : (word x).msb=Bool.false := by
    apply BitVec.msb_eq_false_iff_two_mul_lt.mpr
    change 2*(x%2^32)<2^32
    rw [Nat.mod_eq_of_lt (by omega)]
    omega
  have ny : (word y).msb=Bool.false := by
    apply BitVec.msb_eq_false_iff_two_mul_lt.mpr
    change 2*(y%2^32)<2^32
    rw [Nat.mod_eq_of_lt (by omega)]
    omega
  rw [BitVec.sdiv_eq,nx,ny]
  apply BitVec.eq_of_toNat_eq
  simp only [BitVec.toNat_udiv]
  change (x%2^32)/(y%2^32)=(x/y)%2^32
  rw [Nat.mod_eq_of_lt (by omega),Nat.mod_eq_of_lt (by omega),Nat.mod_eq_of_lt (by
    have := Nat.div_le_self x y; omega)]

theorem cdiv_word (g : Geometry) (raw : Nat → Int) (v : Valid g raw) (hn : 0<g.N) :
    (word g.N+word g.N-word 1).sdiv (word g.N)=word 1 := by
  rw [show word g.N+word g.N-word 1=word (g.N+g.N-1) by
    simp only [word,BitVec.ofNat_add_ofNat]
    rw [BitVec.ofNat_sub_ofNat_of_le _ _ (by norm_num) (by omega)]]
  rw [sdiv_word _ _ (by have := v.cdiv; omega) v.grid.2]
  congr 1
  exact length_one g hn

def pointer (width base off : Nat) : BitVec 64 :=
  BitVec.ofNat 64 base + BitVec.ofInt 64 ((width : Int)*(word off).toInt)

theorem pointer_projection (width base off : Nat) (ho : off<2^31)
    (h : base+width*off<2^64) : (pointer width base off).toNat=base+width*off := by
  unfold pointer
  rw [signed_word off ho]
  rw [←Int.natCast_mul,BitVec.ofInt_natCast,BitVec.ofNat_add_ofNat]
  exact Nat.mod_eq_of_lt h

theorem signed_index (g : Geometry) (raw : Nat → Int) (v : Valid g raw) (b : Nat) (hb : b<g.B) :
    (BitVec.ofInt 32 (logicalIndex g raw b)).toInt=logicalIndex g raw b := by
  apply BitVec.toInt_ofInt_eq_self (by decide)
  · rcases v.ids b hb with h|h <;> omega
  · have := v.index32
    rcases v.ids b hb with h|h <;> omega

def xword (g : Geometry) (b k : Nat) := word b*word g.xm+word k*word g.xk
def wword (g : Geometry) (l n k : Nat) := word l*word g.l0+word n*word g.wn+word k*word g.wk
def yword (g : Geometry) (b n : Nat) := word b*word g.ym+word n

theorem x_projection (g : Geometry) (raw : Nat → Int) (v : Valid g raw)
    (b k : Nat) (hb : b<g.B) (hk : k<g.block) :
    (xword g b k).toInt=(xoff g b k : Int) := by
  have h1 := Nat.mul_le_mul_right g.xm (Nat.le_of_lt hb)
  have h2 := Nat.mul_le_mul_right g.xk (Nat.le_of_lt hk)
  have bound : xoff g b k<2^31 := by have := v.xArithmetic; unfold xoff; omega
  simpa [xword,word,BitVec.ofNat_mul_ofNat,BitVec.ofNat_add_ofNat,xoff] using signed_word (xoff g b k) bound

theorem w_projection (g : Geometry) (raw : Nat → Int) (v : Valid g raw)
    (l n k : Nat) (hl : l<g.L) (hn : n<g.N) (hk : k<g.block) :
    (wword g l n k).toInt=(woff g l n k : Int) := by
  have h1 := Nat.mul_le_mul_right g.l0 (Nat.le_of_lt hl)
  have h2 := Nat.mul_le_mul_right g.wn (Nat.le_of_lt hn)
  have h3 := Nat.mul_le_mul_right g.wk (Nat.le_of_lt hk)
  have bound : woff g l n k<2^31 := by have := v.wArithmetic; unfold woff; omega
  simpa [wword,word,BitVec.ofNat_mul_ofNat,BitVec.ofNat_add_ofNat,woff] using signed_word (woff g l n k) bound

theorem y_projection (g : Geometry) (raw : Nat → Int) (v : Valid g raw)
    (b n : Nat) (hb : b<g.B) (hn : n<g.N) :
    (yword g b n).toInt=(yoff g b n : Int) := by
  have h1 := Nat.mul_le_mul_right g.ym (Nat.le_of_lt hb)
  have bound : yoff g b n<2^31 := by have := v.yArithmetic; unfold yoff; omega
  simpa [yword,word,BitVec.ofNat_mul_ofNat,BitVec.ofNat_add_ofNat,yoff] using signed_word (yoff g b n) bound

theorem input_mask (g : Geometry) (raw : Nat → Int) (v : Valid g raw) (k : Fin g.block) :
    (g.even || decide (k.val<g.K))=Bool.true ↔ k.val<g.K := by
  constructor
  · intro h
    by_cases he : g.even=Bool.true
    · have := v.even he; have := k.isLt; omega
    · simpa [he,Bool.eq_false_of_not_eq_true he] using h
  · intro h; simp [h]

/-- Actual host split length is one: only lane zero is active. All other 255
rows are masked, including weight accesses. There is no partial-row overflow. -/
theorem masks (g : Geometry) (n lane k : Nat) (hn : n<g.N) :
    (lane<Branches.length g.parameters ↔ lane=0) ∧
    (lane<Branches.length g.parameters ∧ k<g.K ↔ lane=0 ∧ k<g.K) := by
  rw [length_one g (by omega)]
  omega

theorem active_bounds (g : Geometry) (raw : Nat → Int) (v : Valid g raw)
    (b n k : Nat) (hb : b<g.B) (hn : n<g.N) (hk : k<g.K)
    (ha : logicalIndex g raw b≠-1) :
    xoff g b k<g.xCapacity ∧
      woff g (logicalIndex g raw b).toNat n k<g.wCapacity ∧
      yoff g b n<g.yCapacity := by
  have hi := (v.ids b hb).resolve_left ha
  have hl : (logicalIndex g raw b).toNat<g.L := by omega
  exact ⟨v.x b k hb hk,v.w _ n k hl hn hk,v.y b n hb hn⟩

theorem byte_bound (base capacity off width j : Nat) (ho : off<capacity)
    (hj : j<width) (h : base+width*capacity<2^64) :
    base+width*off+j<base+width*capacity ∧ base+width*off+j<2^64 := by
  have := Nat.mul_le_mul_left width (Nat.succ_le_of_lt ho)
  constructor <;> nlinarith

#axiomsClean sdiv_word
#axiomsClean cdiv_word
#axiomsClean pointer_projection
#axiomsClean signed_index
#axiomsClean x_projection
#axiomsClean w_projection
#axiomsClean y_projection
#axiomsClean input_mask
#axiomsClean masks
#axiomsClean active_bounds
#axiomsClean byte_bound
end
end VeriTile.Bench.Repairs.LoraExpandGemv.Bounds
