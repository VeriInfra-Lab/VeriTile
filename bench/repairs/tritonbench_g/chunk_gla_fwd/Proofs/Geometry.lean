import VeriTile.Triton
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.ChunkGlaFwd.Geometry
open VeriTile.Triton

def ceilDiv (n width : Nat) : Nat := (n + width - 1) / width

theorem owner_lt {n width i : Nat} (hw : 0 < width) (hi : i < n) :
    i / width < ceilDiv n width := by
  have h := Nat.div_mul_le_self i width
  have hm : (i / width + 1) * width ≤ n + width - 1 := by
    rw [Nat.add_mul, Nat.one_mul]
    omega
  have hd := (Nat.le_div_iff_mul_le hw).2 hm
  exact Nat.lt_of_lt_of_le (Nat.lt_succ_self _) hd

theorem packed_lt {a A b B : Nat} (ha : a < A) (hb : b < B) :
    a * B + b < A * B := by
  calc
    a * B + b < a * B + B := Nat.add_lt_add_left hb _
    _ = (a + 1) * B := by ring
    _ ≤ A * B := Nat.mul_le_mul_right B ha

theorem packed_injective {a a' b b' B : Nat} (hb : b < B) (hb' : b' < B)
    (he : a * B + b = a' * B + b') : a = a' ∧ b = b' := by
  have hB : 0 < B := Nat.zero_lt_of_lt hb
  have hmod := congrArg (fun x => x % B) he
  have hbb : b = b' := by simpa [Nat.add_mod, Nat.mod_eq_of_lt hb, Nat.mod_eq_of_lt hb'] using hmod
  subst b'
  have hmul := Nat.add_right_cancel he
  exact ⟨Nat.eq_of_mul_eq_mul_right hB hmul,rfl⟩

def attentionAddress (T BT head row column : Nat) := (head * T + row) * BT + column
def scratchAddress (heads T feature head row column : Nat) :=
  ((feature * heads + head) * T + row) * 16 + column
def outputAddress (T V head row value : Nat) := (head * T + row) * V + value
def historyAddress (chunks K V head chunk key value : Nat) :=
  ((head * chunks + chunk) * K + key) * V + value

theorem attention_bound {heads T BT head row column : Nat}
    (hh : head < heads) (hr : row < T) (hc : column < BT) :
    attentionAddress T BT head row column < heads * T * BT :=
  packed_lt (packed_lt hh hr) hc

theorem scratch_bound {features heads T feature head row column : Nat}
    (hf : feature < features) (hh : head < heads) (hr : row < T) (hc : column < 16) :
    scratchAddress heads T feature head row column < features * heads * T * 16 :=
  packed_lt (packed_lt (packed_lt hf hh) hr) hc

theorem output_bound {heads T V head row value : Nat}
    (hh : head < heads) (hr : row < T) (hv : value < V) :
    outputAddress T V head row value < heads * T * V :=
  packed_lt (packed_lt hh hr) hv

theorem history_bound {heads chunks K V head chunk key value : Nat}
    (hh : head < heads) (ht : chunk < chunks) (hk : key < K) (hv : value < V) :
    historyAddress chunks K V head chunk key value < heads * chunks * K * V :=
  packed_lt (packed_lt (packed_lt hh ht) hk) hv

theorem attention_injective {T BT h h' r r' c c' : Nat}
    (hr : r < T) (hr' : r' < T) (hc : c < BT) (hc' : c' < BT)
    (he : attentionAddress T BT h r c = attentionAddress T BT h' r' c') :
    h = h' ∧ r = r' ∧ c = c' := by
  obtain ⟨hhr,hcc⟩ := packed_injective hc hc' he
  obtain ⟨hhh,hrr⟩ := packed_injective hr hr' hhr
  exact ⟨hhh,hrr,hcc⟩

theorem scratch_source_address (heads T feature head row column : Nat) :
    (head + feature * heads) * T * 16 + row * 16 + column =
      scratchAddress heads T feature head row column := by
  unfold scratchAddress
  ring

theorem history_source_address (chunks K V head chunk key value : Nat) :
    head * (chunks * K * V) + chunk * K * V + key * V + value =
      historyAddress chunks K V head chunk key value := by
  unfold historyAddress
  ring

/-- Every causal consumer column has either one earlier-subblock writer or
the same-subblock writer. Tails do not alter this partition. -/
theorem causal_partition {row column : Nat} (hc : column ≤ row) :
    column / 16 < row / 16 ∨ column / 16 = row / 16 := by
  have h : column / 16 ≤ row / 16 := Nat.div_le_div_right hc
  omega

/-- The column used by a causal output is among the actual streamed diagonal
columns, including the last partial subblock. -/
theorem diagonal_tail_covered {T start i j : Nat}
    (hi : i < 16) (hrow : start + i < T) (hcausal : j ≤ i) :
    j < min 16 (T - start) := by
  omega

/-- The source-declared scratch extent covers the old K=512 counterexample. -/
theorem repaired_scratch_counterexample :
    scratchAddress 4 128 1 0 0 0 = 8192 ∧
    4096 ≤ scratchAddress 4 128 1 0 0 0 ∧
    scratchAddress 4 128 1 0 0 0 < 4 * 4 * 128 * 16 := by decide

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.ChunkGlaFwd.Geometry
