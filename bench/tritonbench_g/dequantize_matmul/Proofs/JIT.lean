import bench.tritonbench_g.dequantize_matmul.Proofs.Source

namespace VeriTile.Bench.Repairs.DequantizeMatmul
noncomputable section
set_option maxHeartbeats 3000000

/-- Contiguity only constrains a stride when its dimension has more than one
entry. Empty dimensions have no live indices. Offsets remain in Location. -/
structure Dense2 (rows cols : Nat) where
  rowStride : Nat
  colStride : Nat
  rowDense : 0<rows → 0<cols → rows≤1 ∨ rowStride=cols
  colDense : 0<rows → 0<cols → cols≤1 ∨ colStride=1
structure Dense1 (size : Nat) where
  stride : Nat
  dense : size≤1 ∨ stride=1

theorem Dense2.offset (d : Dense2 rows cols) (r c : Nat) (hr : r<rows) (hc : c<cols) :
    r*d.rowStride+c*d.colStride=r*cols+c := by
  have hrow : r*d.rowStride=r*cols := by
    rcases d.rowDense (Nat.zero_lt_of_lt hr) (Nat.zero_lt_of_lt hc) with h|h
    · have : r=0 := by omega
      simp [this]
    · rw [h]
  have hcol : c*d.colStride=c := by
    rcases d.colDense (Nat.zero_lt_of_lt hr) (Nat.zero_lt_of_lt hc) with h|h
    · have : c=0 := by omega
      simp [this]
    · simp [h]
  rw [hrow,hcol]
theorem Dense1.offset (d : Dense1 size) (n : Nat) (hn : n<size) : n*d.stride=n := by
  rcases d.dense with h|h
  · have : n=0 := by omega
    simp [this]
  · simp [h]

structure Lane (tile : Tile) where
  programK : Nat
  programN : Nat
  localK : Fin tile.rows
  localN : Fin tile.columns

def Lane.k (l : Lane tile) := l.programK*tile.rows+l.localK.val
def Lane.n (l : Lane tile) := l.programN*tile.columns+l.localN.val
def Lane.active (s : Shape) (l : Lane tile) : Prop := l.k<s.K ∧ l.n<s.N
instance (s : Shape) (l : Lane tile) : Decidable (l.active s) := inferInstanceAs (Decidable (_ ∧ _))

/-- These are the repaired Python's exact load/multiply/store operands.
The scale load has the N-only mask, even for inactive K tail lanes. -/
def laneValue (a : Arithmetic α) (s : Shape) (b : Buffers)
    (db : Dense2 s.K s.N) (ds : Dense1 s.N) (m : Memory (Cell α)) (l : Lane tile) : α :=
  let q := if l.active s then (m (b.b.addr (l.k*db.rowStride+l.n*db.colStride))).signed else 0
  let scale := if l.n<s.N then a.load ((m (b.scale.addr (l.n*ds.stride))).float a.zero) else a.zero
  a.store (a.mul (a.signed8 q) scale)

theorem lane_active (a : Arithmetic α) (s : Shape) (b : Buffers)
    (db : Dense2 s.K s.N) (ds : Dense1 s.N) (m : Memory (Cell α))
    (l : Lane tile) (h : l.active s) :
    Cell.fp32 (laneValue a s b db ds m l)=dequantValue a s b m l.k l.n := by
  simp only [laneValue,if_pos h,if_pos h.2,db.offset _ _ h.1 h.2,ds.offset _ h.2,dequantValue]

theorem lane_owner (l : Lane tile) : l.k/tile.rows=l.programK ∧ l.n/tile.columns=l.programN := by
  constructor
  · unfold Lane.k
    rw [Nat.mul_comm,Nat.mul_add_div tile.rowsPositive,Nat.div_eq_of_lt l.localK.isLt,Nat.add_zero]
  · unfold Lane.n
    rw [Nat.mul_comm,Nat.mul_add_div tile.columnsPositive,Nat.div_eq_of_lt l.localN.isLt,Nat.add_zero]

def coveringLane (tile : Tile) (k n : Nat) : Lane tile :=
  ⟨k/tile.rows,n/tile.columns,⟨k%tile.rows,Nat.mod_lt _ tile.rowsPositive⟩,
    ⟨n%tile.columns,Nat.mod_lt _ tile.columnsPositive⟩⟩
theorem coveringLane_coordinates (tile : Tile) (k n : Nat) :
    (coveringLane tile k n).k=k ∧ (coveringLane tile k n).n=n := by
  constructor
  · simpa [coveringLane,Lane.k,Nat.mul_comm,Nat.add_comm] using Nat.mod_add_div k tile.rows
  · simpa [coveringLane,Lane.n,Nat.mul_comm,Nat.add_comm] using Nat.mod_add_div n tile.columns

theorem active_offsets_in_bounds (s : Shape) (db : Dense2 s.K s.N)
    (ds : Dense1 s.N) (df : Dense2 s.K s.N) (l : Lane tile) (h : l.active s) :
    l.k*db.rowStride+l.n*db.colStride<s.K*s.N ∧
    l.n*ds.stride<s.N ∧ l.k*df.rowStride+l.n*df.colStride<s.K*s.N := by
  rw [db.offset _ _ h.1 h.2,ds.offset _ h.2,df.offset _ _ h.1 h.2]
  have hk : l.k*s.N+s.N≤s.K*s.N := by
    calc l.k*s.N+s.N = (l.k+1)*s.N := by simp [Nat.add_mul]
         _ ≤ s.K*s.N := Nat.mul_le_mul_right _ h.1
  have hn := h.2
  omega

/-- The wider scale mask is safe independently of the K mask. -/
theorem scale_mask_bound (s : Shape) (ds : Dense1 s.N) (l : Lane tile) (hn : l.n<s.N) :
    l.n*ds.stride<s.N := by rw [ds.offset _ hn]; exact hn

#axiomsClean lane_active
#axiomsClean lane_owner
#axiomsClean coveringLane_coordinates
#axiomsClean active_offsets_in_bounds
#axiomsClean scale_mask_bound
end
end VeriTile.Bench.Repairs.DequantizeMatmul
