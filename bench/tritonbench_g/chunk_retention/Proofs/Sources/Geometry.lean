import VeriTile.Triton.DSL
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001
noncomputable section

structure Shape where
  B : Nat
  H : Nat
  T : Nat
  K : Nat
  V : Nat
  BK : Nat
  BV : Nat
  keyPositive : 0 < K
  valuePositive : 0 < V
  keyTile : BK = 16 ∨ BK = 32 ∨ BK = 64
  valueTile : BV = 16 ∨ BV = 32 ∨ BV = 64

namespace Shape
def heads (c : Shape) := c.B*c.H
def chunks (c : Shape) := (c.T+63)/64
def keyBlocks (c : Shape) := (c.K+c.BK-1)/c.BK
def valueBlocks (c : Shape) := (c.V+c.BV-1)/c.BV
def paddedTime (c : Shape) := c.chunks*64
def paddedKey (c : Shape) := c.keyBlocks*c.BK
def paddedValue (c : Shape) := c.valueBlocks*c.BV
def keyHead (c : Shape) := c.T*c.K
def valueHead (c : Shape) := c.T*c.V
def stateHead (c : Shape) := c.K*c.V
def historyHead (c : Shape) := c.chunks*c.stateHead
def keySize (c : Shape) := c.heads*c.keyHead
def valueSize (c : Shape) := c.heads*c.valueHead
def stateSize (c : Shape) := c.heads*c.stateHead
def historySize (c : Shape) := c.heads*c.historyHead
def partialSize (c : Shape) := c.keyBlocks*c.valueSize
def keyOffset (c : Shape) (head row key : Nat) := head*c.keyHead+row*c.K+key
def valueOffset (c : Shape) (head row value : Nat) := head*c.valueHead+row*c.V+value
def stateOffset (c : Shape) (head key value : Nat) := head*c.stateHead+key*c.V+value
def historyOffset (c : Shape) (head chunk key value : Nat) :=
  head*c.historyHead+chunk*c.stateHead+key*c.V+value
def partialOffset (c : Shape) (block head row value : Nat) :=
  block*c.valueSize+c.valueOffset head row value
/-- Valid time boundary after c chunks. -/
def boundary (c : Shape) (chunk : Nat) := min (chunk*64) c.T
def length (c : Shape) (chunk : Nat) := min 64 (c.T-chunk*64)

theorem key_tile_pos (c : Shape) : 0<c.BK := by rcases c.keyTile with h|h|h <;> omega
theorem value_tile_pos (c : Shape) : 0<c.BV := by rcases c.valueTile with h|h|h <;> omega
theorem time_cover (c : Shape) : c.T≤c.paddedTime := by
  have h := Nat.mod_lt (c.T+63) (by decide : 0<64)
  have e := Nat.div_add_mod (c.T+63) 64
  dsimp [paddedTime,chunks]
  omega
theorem chunks_zero (c : Shape) : c.chunks=0 ↔ c.T=0 := by dsimp [chunks]; omega
theorem chunk_of_row (c : Shape) (row : Nat) (hr : row<c.T) : row/64<c.chunks := by
  have h := c.time_cover
  have e := Nat.div_add_mod row 64
  dsimp [paddedTime] at h
  omega
theorem chunk_start (c : Shape) (chunk : Nat) (hc : chunk<c.chunks) : chunk*64<c.T := by
  have h := Nat.div_mul_le_self (c.T+63) 64
  dsimp [chunks] at hc
  omega
theorem boundary_at_start (c : Shape) (chunk : Nat) (hc : chunk<c.chunks) :
    c.boundary chunk=chunk*64 := by simp [boundary,Nat.le_of_lt (c.chunk_start chunk hc)]
theorem boundary_final (c : Shape) : c.boundary c.chunks=c.T := by
  exact Nat.min_eq_right c.time_cover
theorem boundary_succ (c : Shape) (chunk : Nat) (hc : chunk<c.chunks) :
    c.boundary (chunk+1)=chunk*64+c.length chunk := by
  have h := c.chunk_start chunk hc
  simp only [boundary,length,Nat.add_mul,Nat.one_mul]
  omega
theorem length_pos (c : Shape) (chunk : Nat) (hc : chunk<c.chunks) : 0<c.length chunk := by
  have h := c.chunk_start chunk hc
  dsimp [length]
  omega
theorem length_le (c : Shape) (chunk : Nat) : c.length chunk≤64 := Nat.min_le_left _ _
theorem length_source (c : Shape) (chunk : Nat) (hc : chunk<c.chunks) :
    c.length chunk=(if chunk=c.chunks-1 ∧ c.T%64≠0 then c.T%64 else 64) := by
  have h := c.chunk_start chunk hc
  have d := Nat.div_add_mod c.T 64
  have m := Nat.mod_lt c.T (by decide : 0<64)
  dsimp [length,chunks] at *
  split_ifs <;> omega
end Shape

theorem digit_bound (a A b B : Nat) (ha : a<A) (hb : b<B) : a*B+b<A*B := by
  have h := Nat.mul_le_mul_right B (Nat.succ_le_iff.mpr ha)
  simp only [Nat.succ_mul] at h
  omega

theorem digit_injective (B a b a' b' : Nat) (hb : b<B) (hb' : b'<B)
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
def Allocation.Contains (a : Allocation) (point : Nat) : Prop := a.base≤point ∧ point<a.base+a.length
def Allocation.Separate (a b : Allocation) : Prop := ∀ point, a.Contains point → ¬b.Contains point

def Memory (α : Type) := Nat → α

def Memory.store (m : Memory α) (base count : Nat) (value : Nat → α) : Memory α :=
  fun point => if base≤point ∧ point<base+count then value (point-base) else m point

theorem Memory.store_read (m : Memory α) (base count i : Nat) (hi : i<count) (value : Nat → α) :
    store m base count value (base+i)=value i := by simp [store,hi,Nat.add_lt_add_left hi]
theorem Memory.store_frame (m : Memory α) (base count : Nat) (value : Nat → α) (point : Nat)
    (outside : ¬(base≤point ∧ point<base+count)) : store m base count value point=m point := by
  simp [store,outside]

#axiomsClean Shape.length_source
#axiomsClean Shape.boundary_succ
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkRetention.Whole001
