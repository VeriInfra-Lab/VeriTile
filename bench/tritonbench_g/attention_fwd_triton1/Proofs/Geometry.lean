import VeriTile.Triton.DSL
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.AttentionFwdTriton1.Whole001
noncomputable section

structure Shape where
  B : Nat
  H : Nat
  T : Nat
  D : Nat
  feature : 16 ≤ D
  power : ∃ n, D = 2^n

namespace Shape
def heads (c : Shape) := c.B*c.H
def chunks (c : Shape) := (c.T+31)/32
def padded (c : Shape) := c.chunks*32
def headSize (c : Shape) := c.T*c.D
def mainSize (c : Shape) := c.heads*c.headSize
def historyHead (c : Shape) := c.chunks*c.D*c.D
def historySize (c : Shape) := c.heads*c.historyHead
def mainOffset (c : Shape) (head row col : Nat) := head*c.headSize+row*c.D+col
def historyOffset (c : Shape) (head chunk row col : Nat) :=
  head*c.historyHead+(chunk*c.D+row)*c.D+col
def mainExtent (c : Shape) := (c.heads-1)*c.headSize+c.padded*c.D
def historyExtent (c : Shape) := c.historySize

theorem dim_pos (c : Shape) : 0<c.D := by have := c.feature; omega
theorem padded_cover (c : Shape) : c.T≤c.padded := by
  have h := Nat.mod_lt (c.T+31) (by decide : 0<32)
  have e := Nat.div_add_mod (c.T+31) 32
  dsimp [padded,chunks]
  omega
theorem chunks_zero (c : Shape) : c.chunks=0 ↔ c.T=0 := by
  dsimp [chunks]
  omega
theorem chunk_of_row (c : Shape) (row : Nat) (hr : row<c.T) : row/32<c.chunks := by
  have h := c.padded_cover
  have e := Nat.div_add_mod row 32
  dsimp [padded] at h
  omega
theorem chunk_start (c : Shape) (chunk : Nat) (hc : chunk<c.chunks) : chunk*32<c.T := by
  have h := Nat.div_mul_le_self (c.T+31) 32
  dsimp [chunks] at hc
  omega
end Shape

theorem digit_bound (a A b B : Nat) (ha : a<A) (hb : b<B) : a*B+b<A*B := by
  have h := Nat.mul_le_mul_right B (Nat.succ_le_iff.mpr ha)
  simp only [Nat.succ_mul] at h
  omega

theorem digit_injective (B : Nat) (a b a' b' : Nat) (hb : b<B) (hb' : b'<B)
    (he : a*B+b=a'*B+b') : a=a' ∧ b=b' := by
  have ha : a=a' := by
    by_contra h
    rcases lt_or_gt_of_ne h with h|h
    · have hm := Nat.mul_le_mul_right B (Nat.succ_le_iff.mpr h)
      simp only [Nat.succ_mul] at hm
      omega
    · have hm := Nat.mul_le_mul_right B (Nat.succ_le_iff.mpr h)
      simp only [Nat.succ_mul] at hm
      omega
  exact ⟨ha,by simpa [ha] using he⟩

structure Allocation where
  base : Nat
  length : Nat
def Allocation.Contains (a : Allocation) (point : Nat) : Prop :=
  a.base≤point ∧ point<a.base+a.length
def Allocation.Separate (a b : Allocation) : Prop :=
  ∀ point, a.Contains point → ¬b.Contains point

structure ABI (c : Shape) where
  q : Allocation
  k : Allocation
  v : Allocation
  h : Allocation
  o : Allocation
  qLength : q.length=c.mainSize
  kLength : k.length=c.mainSize
  vLength : v.length=c.mainSize
  hLength : h.length=c.historySize
  oLength : o.length=c.mainSize
  outputSeparate : ∀ a∈[q,k,v,h], o.Separate a
  historySeparate : ∀ a∈[q,k,v,o], h.Separate a

def Memory (α : Type) := Nat → α

namespace Memory
def store (m : Memory α) (base count : Nat) (value : Nat → α) : Memory α :=
  fun point => if base≤point ∧ point<base+count then value (point-base) else m point

theorem store_read (m : Memory α) (base count i : Nat) (hi : i<count) (value : Nat → α) :
    store m base count value (base+i)=value i := by
  simp [store,hi,Nat.add_lt_add_left hi]

theorem store_frame (m : Memory α) (base count : Nat) (value : Nat → α) (point : Nat)
    (outside : ¬(base≤point ∧ point<base+count)) : store m base count value point=m point := by
  simp [store,outside]

theorem store_ext (m : Memory α) (base count : Nat) (f g : Nat → α)
    (h : ∀ i, i<count → f i=g i) : store m base count f=store m base count g := by
  funext point
  unfold store
  split_ifs with hb
  · exact h _ (by omega)
  · rfl
end Memory

namespace Shape
theorem main_bound (c : Shape) (head row col : Nat)
    (hh : head<c.heads) (hr : row<c.T) (hd : col<c.D) :
    c.mainOffset head row col<c.mainSize := by
  have hrow := digit_bound row c.T col c.D hr hd
  have h := digit_bound head c.heads (row*c.D+col) c.headSize hh hrow
  simpa [mainOffset,mainSize,Nat.add_assoc] using h

theorem history_bound (c : Shape) (head chunk row col : Nat)
    (hh : head<c.heads) (hc : chunk<c.chunks) (hr : row<c.D) (hd : col<c.D) :
    c.historyOffset head chunk row col<c.historySize := by
  have hrow := digit_bound chunk c.chunks row c.D hc hr
  have hcell := digit_bound (chunk*c.D+row) (c.chunks*c.D) col c.D hrow hd
  have h := digit_bound head c.heads ((chunk*c.D+row)*c.D+col) c.historyHead hh hcell
  simpa [historyOffset,historySize,Nat.add_assoc] using h
end Shape

#axiomsClean Shape.padded_cover
#axiomsClean Shape.main_bound
#axiomsClean Shape.history_bound
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttentionFwdTriton1.Whole001
