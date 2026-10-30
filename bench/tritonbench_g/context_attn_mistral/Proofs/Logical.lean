import bench.tritonbench_g.context_attn_mistral.Proofs.Kernels
import bench.tritonbench_g.context_attn_mistral.Proofs.Mathematical_HostReadback

namespace VeriTile.Bench.Repairs.ContextAttnMistral.Logical
open VeriTile VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnMistral
open Formula Program
noncomputable section
set_option maxHeartbeats 2000000

/-- Logical tensor-library coordinates, independent of raw storage, strides,
lazy-negative flags, intermediate registers and kernel output. -/
structure Tensors where
  q : Nat → Nat → Nat → ℝ
  k : Nat → Nat → Nat → ℝ
  v : Nat → Nat → Nat → ℝ

/-- Ordinary binding of current input coordinates to logical source roles.
The historical name is retained for shared imports; the checked wrapper uses
the supplied views directly and performs no resolution, cloning or copyback. -/
structure Resolved (c : Host.Call) (s : BlockState) (t : Tensors) : Prop where
  q : ∀ b h row d, b < c.batches → h < c.heads → row < c.length b → d < c.layout.D →
    s.readMem c.regions.query ((c.start b+row)*c.layout.qRow+h*c.layout.qHead+d*c.layout.qDim) =
      t.q (c.start b+row) h d
  k : ∀ b h row d, b < c.batches → h < c.heads → row < c.length b → d < c.layout.D →
    s.readMem c.regions.key ((c.start b+row)*c.layout.kRow+(h/c.layout.kvGroup)*c.layout.kHead+d*c.layout.kDim) =
      t.k (c.start b+row) (h/c.layout.kvGroup) d
  v : ∀ b h row d, b < c.batches → h < c.heads → row < c.length b → d < c.layout.D →
    s.readMem c.regions.value ((c.start b+row)*c.layout.vRow+(h/c.layout.kvGroup)*c.layout.vHead+d*c.layout.vDim) =
      t.v (c.start b+row) (h/c.layout.kvGroup) d

/-- The logical padded key/value; the finite -1e9 score retains these lanes
in the denominator, although a padded value contributes zero to the numerator. -/
def key (c : Host.Call) (t : Tensors) (b h j d : Nat) : ℝ :=
  if j < c.length b then t.k (c.start b+j) (h/c.layout.kvGroup) d else 0

def value (c : Host.Call) (t : Tensors) (b h j d : Nat) : ℝ :=
  if j < c.length b then t.v (c.start b+j) (h/c.layout.kvGroup) d else 0

def score (c : Host.Call) (t : Tensors) (b h row j : Nat) : ℝ :=
  if (j : Int) ≤ row ∧ (row : Int)-c.layout.window < j then
    (∑ d : Fin c.layout.D, t.q (c.start b+row) h d * key c t b h j d) * Host.scale c.layout.D
  else -1000000000

/-- Independent initial-logical-array closed weighted sum, with source host
scale and finite sentinel. The key range is the actual visited full tiles. -/
def expected (c : Host.Call) (t : Tensors) (b h row d : Nat) : ℝ :=
  (∑ j : Fin ((row/128+1)*128), Real.exp (score c t b h row j) * value c t b h j d) /
    ∑ j : Fin ((row/128+1)*128), Real.exp (score c t b h row j)

theorem query_eq (c : Host.Call) (A : FlatAlloc) (s : BlockState) (t : Tensors)
    (v : Host.Valid c A s) (hr : Resolved c s t) (b h row d : Nat)
    (hb : b < c.batches) (hh : h < c.heads) (hrow : row < c.length b) (hd : d < c.layout.D) :
    Formula.query c.layout ⟨row/128,b,h⟩ (inputs s c.regions) (row%128) d =
      t.q (c.start b+row) h d := by
  have hroweq : queryRow c.layout ⟨row/128,b,h⟩ (row%128) = row := by
    simp [queryRow,c.blockM,Nat.div_add_mod,Nat.mul_comm]
  unfold Formula.query Formula.sequence
  rw [hroweq,v.length_value b hb,v.start_value b hb]
  have hri : (row : Int) < (c.length b : Int) := Int.ofNat_lt.mpr hrow
  simp only [hri,↓reduceIte]
  convert hr.q b h row d hb hh hrow hd using 1

theorem key_eq (c : Host.Call) (A : FlatAlloc) (s : BlockState) (t : Tensors)
    (v : Host.Valid c A s) (hr : Resolved c s t) (b h row j d : Nat)
    (hb : b < c.batches) (hh : h < c.heads) (hd : d < c.layout.D) :
    Formula.key c.layout ⟨row/128,b,h⟩ (inputs s c.regions) j d = key c t b h j d := by
  unfold Formula.key Formula.sequence key
  rw [v.length_value b hb,v.start_value b hb]
  simp only [Int.ofNat_lt]
  split_ifs with hj
  · convert hr.k b h j d hb hh hj hd using 1
  · rfl

theorem value_eq (c : Host.Call) (A : FlatAlloc) (s : BlockState) (t : Tensors)
    (v : Host.Valid c A s) (hr : Resolved c s t) (b h row j d : Nat)
    (hb : b < c.batches) (hh : h < c.heads) (hd : d < c.layout.D) :
    Formula.value c.layout ⟨row/128,b,h⟩ (inputs s c.regions) j d = value c t b h j d := by
  unfold Formula.value Formula.sequence value
  rw [v.length_value b hb,v.start_value b hb]
  simp only [Int.ofNat_lt]
  split_ifs with hj
  · convert hr.v b h j d hb hh hj hd using 1
  · rfl

theorem score_eq (c : Host.Call) (A : FlatAlloc) (s : BlockState) (t : Tensors)
    (v : Host.Valid c A s) (hr : Resolved c s t) (b h row j : Nat)
    (hb : b < c.batches) (hh : h < c.heads) (hrow : row < c.length b) :
    Formula.score c.layout ⟨row/128,b,h⟩ (inputs s c.regions) (Host.scale c.layout.D) (row%128) j =
      score c t b h row j := by
  have hroweq : queryRow c.layout ⟨row/128,b,h⟩ (row%128) = row := by
    simp [queryRow,c.blockM,Nat.div_add_mod,Nat.mul_comm]
  unfold Formula.score score
  rw [hroweq]
  congr 1
  congr 1
  apply Finset.sum_congr rfl
  intro d _
  rw [query_eq c A s t v hr b h row d hb hh hrow d.isLt,
    key_eq c A s t v hr b h row j d hb hh d.isLt]

theorem expected_eq (c : Host.Call) (A : FlatAlloc) (s : BlockState) (t : Tensors)
    (v : Host.Valid c A s) (hr : Resolved c s t) (b h row d : Nat)
    (hb : b < c.batches) (hh : h < c.heads) (hrow : row < c.length b) (hd : d < c.layout.D) :
    c.logicalExpected s b h row d = expected c t b h row d := by
  have hblock : row/128*128 < c.length b := by omega
  have hc : Formula.count c.layout ⟨row/128,b,h⟩ (inputs s c.regions) = row/128+1 := by
    unfold Formula.count Formula.sequence
    rw [v.length_value b hb,c.blockM,c.blockN]
    have hbi : (row/128 : Int)*128 < (c.length b : Int) := by exact_mod_cast hblock
    simp [hbi]
  unfold Host.Call.logicalExpected Formula.expected Formula.numerator Formula.denominator expected
  rw [hc,c.blockN]
  congr 1
  · apply Finset.sum_congr rfl
    intro j _
    rw [score_eq c A s t v hr b h row j hb hh hrow,
      value_eq c A s t v hr b h row j d hb hh hd]
  · apply Finset.sum_congr rfl
    intro j _
    rw [score_eq c A s t v hr b h row j hb hh hrow]
end
end VeriTile.Bench.Repairs.ContextAttnMistral.Logical
