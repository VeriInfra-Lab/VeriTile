import bench.tritonbench_g.attention_fwd_triton3.Proofs.Return
import Init.Data.BitVec.Bitblast

namespace VeriTile.Bench.Repairs.AttentionFwdTriton3.Machine
open VeriTile.Triton

abbrev word (x : Int) : BitVec 64 := BitVec.ofInt 64 x

def signed32 (x : Int) : Prop := -(2^31)≤x ∧ x<2^31

theorem signed_exact (x : Int) (hl : -(2^63)≤x) (hh : x<2^63) : (word x).toInt=x :=
  BitVec.toInt_ofInt_eq_self (by decide) hl hh

theorem widening (x : Int) (hx : signed32 x) :
    ((BitVec.ofInt 32 x).signExtend 64).toInt=x := by
  have h : (BitVec.ofInt 32 x).toInt=x := BitVec.toInt_ofInt_eq_self (by decide) hx.1 hx.2
  change (BitVec.ofInt 64 (BitVec.ofInt 32 x).toInt).toInt=x
  rw [h]
  exact signed_exact x (by have := hx.1; omega) (by have := hx.2; omega)

def lowerNumerator (row offset size : Int) : BitVec 64 := word row+word offset-word size+word 1

def upperNumerator (row bm offset bn : Int) : BitVec 64 := word row+word bm-word 1+word offset+word bn

def distance (row key offset : Int) : BitVec 64 := word row-word key+word offset

theorem lower_exact (row offset size : Int) (hr : 0≤row ∧ row<2^31)
    (ho : signed32 offset) (hs : signed32 size) :
    (lowerNumerator row offset size).toInt=row+offset-size+1 := by
  have he : lowerNumerator row offset size=word (row+offset-size+1) := by
    simp only [lowerNumerator,word,sub_eq_add_neg,BitVec.sub_eq_add_neg,BitVec.ofInt_add,BitVec.ofInt_neg]
  rw [he]
  exact signed_exact _ (by have:=ho.1; have:=hs.2; omega) (by have:=ho.2; have:=hs.1; omega)

theorem upper_exact (row bm offset bn : Int) (hr : 0≤row ∧ row<2^31)
    (hm : 0<bm ∧ bm≤64) (ho : signed32 offset) (hn : 0<bn ∧ bn≤64) :
    (upperNumerator row bm offset bn).toInt=row+bm-1+offset+bn := by
  have he : upperNumerator row bm offset bn=word (row+bm-1+offset+bn) := by
    simp only [upperNumerator,word,sub_eq_add_neg,BitVec.sub_eq_add_neg,BitVec.ofInt_add,BitVec.ofInt_neg]
  rw [he]
  exact signed_exact _ (by have:=ho.1; omega) (by have:=ho.2; omega)

theorem distance_exact (row key offset : Int) (hr : 0≤row ∧ row<2^31)
    (hk : 0≤key ∧ key<2^31) (ho : signed32 offset) :
    (distance row key offset).toInt=row-key+offset := by
  have he : distance row key offset=word (row-key+offset) := by
    simp only [distance,word,sub_eq_add_neg,BitVec.sub_eq_add_neg,BitVec.ofInt_add,BitVec.ofInt_neg]
  rw [he]
  exact signed_exact _ (by have:=ho.1; omega) (by have:=ho.2; omega)

theorem division_exact (x : BitVec 64) (b : Int) (hb : 0<b ∧ b≤64) :
    (x.sdiv (word b)).toInt=x.toInt.tdiv b := by
  have hbb : (word b).toInt=b := signed_exact _ (by omega) (by omega)
  have hne : word b≠(-1 : BitVec 64) := by
    intro he
    have hi:=congrArg BitVec.toInt he
    have hminus : (-1 : BitVec 64).toInt = -1 := by decide
    rw [hbb,hminus] at hi
    omega
  rw [BitVec.toInt_sdiv_of_ne_or_ne _ _ (Or.inr hne),hbb]

/-- The block-pointer ABI narrows only the initial advance. If its loop is
active, clipping gives this bound; otherwise no K/V access uses that pointer. -/
theorem active_advance_exact (lo hi nk : Int) (hlo : 0≤lo)
    (hloop : lo<hi) (hhi : hi≤nk) (hn : nk<2^31) :
    (BitVec.ofInt 32 lo).toInt=lo :=
  BitVec.toInt_ofInt_eq_self (by decide) (by omega) (by omega)

theorem carried_address_exact (hz rounded row : Int)
    (hh : 0≤hz ∧ hz<65536) (hr : 0≤rounded ∧ rounded<2^31)
    (hi : 0≤row ∧ row<rounded) :
    (word hz*word rounded+word row).toInt=hz*rounded+row := by
  have hprod : 0≤hz*rounded := mul_nonneg hh.1 hr.1
  have htop : hz*rounded+row<65536*(2^31) := by nlinarith
  have he : word hz*word rounded+word row=word (hz*rounded+row) := by
    simp only [word,BitVec.ofInt_add,BitVec.ofInt_mul]
  rw [he]
  exact signed_exact _ (by omega) (by norm_num at htop ⊢; omega)

/-- Q/O block offsets and the final KV backedge remain in their INT32 ABI.
The workspace bound is rounded allocation capacity, not logical NQ. -/
theorem block_offset_exact (row : Int) (hr : 0≤row ∧ row<2^31) :
    (BitVec.ofInt 32 row).toInt=row :=
  BitVec.toInt_ofInt_eq_self (by decide) (by omega) hr.2

theorem tail_and_backedge (nk start bn lane : Int) (hn : 0≤nk ∧ nk≤2^31-65)
    (hs : 0≤start ∧ start<nk) (hb : 0<bn ∧ bn≤64) (hl : 0≤lane ∧ lane<bn) :
    0≤start+lane ∧ start+lane<2^31 ∧ start+bn<2^31 := by omega

#print axioms upper_exact
#print axioms carried_address_exact
end VeriTile.Bench.Repairs.AttentionFwdTriton3.Machine
