import bench.repairs.tritonbench_g.attention_fwd_triton1.Whole001.Proofs.Geometry
import Mathlib.Tactic.NormNum

namespace VeriTile.Bench.Repairs.AttentionFwdTriton1.Whole001.Machine
noncomputable section

def i32max : Nat := 2147483647
def i64max : Nat := 9223372036854775807
def Signed32 (x : Int) : Prop := -2147483648≤x ∧ x≤2147483647
def Signed64 (x : Int) : Prop := -9223372036854775808≤x ∧ x≤9223372036854775807

def metadata (c : Shape) : List Nat :=
  [c.B,c.H,c.T,c.D,c.chunks,c.padded,c.headSize,c.chunks*c.D,c.historyHead,c.heads]

structure Launch (c : Shape) : Prop where
  metadata : ∀ x∈metadata c, x≤i32max
  mainBytes : 4*c.mainExtent≤i64max
  historyBytes : 4*c.historyExtent≤i64max

/-- Formal bases are FP32 words. Multiplication by four is the aligned
host byte-address check, not a separate endian/byte-encoding theorem. -/
structure Domain (c : Shape) (abi : ABI c) : Prop where
  launch : c.T≠0 → c.heads≠0 → Launch c
  mainAddress : c.T≠0 → c.heads≠0 → ∀ a∈[abi.q,abi.k,abi.v,abi.o],
    4*(a.base+c.mainExtent)<18446744073709551616
  historyAddress : c.T≠0 → c.heads≠0 →
    4*(abi.h.base+c.historyExtent)<18446744073709551616

theorem natural32 (x : Nat) (hx : x≤i32max) : Signed32 (x : Int) := by
  dsimp [Signed32,i32max] at *
  omega

theorem exact32 (x : Int) (h : Signed32 x) : (BitVec.ofInt 32 x).toInt=x := by
  exact BitVec.toInt_ofInt_eq_self (by decide) h.1 (by dsimp [Signed32] at h; norm_num; omega)

theorem exact64 (x : Int) (h : Signed64 x) : (BitVec.ofInt 64 x).toInt=x := by
  exact BitVec.toInt_ofInt_eq_self (by decide) h.1 (by dsimp [Signed64] at h; norm_num; omega)

theorem main_formation (c : Shape) (head row col : Nat)
    (hh : head<c.heads) (hr : row<c.padded) (hd : col<c.D) :
    c.mainOffset head row col<c.mainExtent := by
  have hhead : head≤c.heads-1 := by omega
  have hm := Nat.mul_le_mul_right c.headSize hhead
  have hrow := digit_bound row c.padded col c.D hr hd
  dsimp [Shape.mainOffset,Shape.mainExtent]
  omega

theorem padded_coordinates (c : Shape) (launch : Launch c)
    (chunk : Nat) (hc : chunk<c.chunks) (row : Fin 32) (d : Fin c.D) :
    Signed32 (chunk*32+row.val : Nat) ∧ Signed32 (chunk*c.D+d.val : Nat) ∧
    Signed32 (chunk+1 : Nat) := by
  have ht := digit_bound chunk c.chunks row.val 32 hc row.isLt
  have hd := digit_bound chunk c.chunks d.val c.D hc d.isLt
  have hpad := launch.metadata c.padded (by simp [metadata])
  have hhist := launch.metadata (c.chunks*c.D) (by simp [metadata])
  have hn := launch.metadata c.chunks (by simp [metadata])
  refine ⟨natural32 _ ?_,natural32 _ ?_,natural32 _ ?_⟩
  · dsimp [Shape.padded] at hpad
    omega
  · omega
  · omega

def PointerExact (base extent offset : Nat) : Prop :=
  ∀ part, part≤offset → Signed64 (part : Int) ∧ Signed64 (4*(part : Int)) ∧
    (BitVec.ofInt 64 (part : Int)).toInt=(part : Int) ∧
    (BitVec.ofInt 64 (4*(part : Int))).toInt=4*(part : Int) ∧
    4*(base+part)<18446744073709551616

theorem pointer_exact (base extent offset : Nat) (hoff : offset<extent)
    (hbyte : 4*extent≤i64max) (haddr : 4*(base+extent)<18446744073709551616) :
    PointerExact base extent offset := by
  intro part hp
  have hpart : part<extent := lt_of_le_of_lt hp hoff
  have hb : Signed64 (4*(part : Int)) := by dsimp [Signed64,i64max] at *; omega
  have hw : Signed64 (part : Int) := by dsimp [Signed64,i64max] at *; omega
  exact ⟨hw,hb,exact64 _ hw,exact64 _ hb,by omega⟩

/-- All source lanes, including temporal padding and the terminal loop index.
Compiler speculative prefetch/scheduling registers stay an ordinary boundary. -/
def Closed (c : Shape) (abi : ABI c) : Prop :=
  c.T≠0 → c.heads≠0 →
  (∀ x∈metadata c, (BitVec.ofInt 32 (x : Int)).toInt=(x : Int)) ∧
  (∀ (head : Fin c.heads) (chunk : Fin c.chunks) (row : Fin 32) (i j : Fin c.D),
    Signed32 (chunk.val*32+row.val : Nat) ∧
    Signed32 (chunk.val*c.D+i.val : Nat) ∧ Signed32 (chunk.val+1 : Nat) ∧
    (∀ a∈[abi.q,abi.k,abi.v,abi.o], PointerExact a.base c.mainExtent
      (c.mainOffset head.val (chunk.val*32+row.val) j.val)) ∧
    PointerExact abi.h.base c.historyExtent
      (c.historyOffset head.val chunk.val i.val j.val))

theorem closed (c : Shape) (abi : ABI c) (domain : Domain c abi) : Closed c abi := by
  intro ht hh
  have launch := domain.launch ht hh
  refine ⟨fun x hx => exact32 _ (natural32 _ (launch.metadata x hx)),?_⟩
  intro head chunk row i j
  obtain ⟨hrow,hi,hnext⟩ := padded_coordinates c launch chunk.val chunk.isLt row i
  have hpad := digit_bound chunk.val c.chunks row.val 32 chunk.isLt row.isLt
  refine ⟨hrow,hi,hnext,?_,?_⟩
  · intro a ha
    exact pointer_exact _ _ _ (main_formation c _ _ _ head.isLt hpad j.isLt)
      launch.mainBytes (domain.mainAddress ht hh a ha)
  · exact pointer_exact _ _ _ (c.history_bound _ _ _ _ head.isLt chunk.isLt i.isLt j.isLt)
      launch.historyBytes (domain.historyAddress ht hh)

#axiomsClean padded_coordinates
#axiomsClean pointer_exact
#axiomsClean closed
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttentionFwdTriton1.Whole001.Machine
