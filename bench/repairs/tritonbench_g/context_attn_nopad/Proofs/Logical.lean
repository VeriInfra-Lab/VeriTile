import bench.repairs.tritonbench_g.context_attn_nopad.Proofs.Source
import bench.tritonbench_g.context_attn_nopad.Proofs.HostCoverage

namespace VeriTile.Bench.Repairs.ContextAttnNopad.Logical
open VeriTile.Triton VeriTile.Bench.TritonBenchG.ContextAttnNopad.Formula VeriTile.Bench.TritonBenchG.ContextAttnNopad.Program VeriTile.Bench.TritonBenchG.ContextAttnNopad.Host
noncomputable section
set_option maxHeartbeats 1200000

/-- The pre-call logical tensor values; no pointer layout, launch or result. -/
structure Inputs where
  query : Nat → Nat → Nat → ℝ
  key : Nat → Nat → Nat → ℝ
  value : Nat → Nat → Nat → ℝ

def score (c : Call) (a : Inputs) (b h t j : Nat) : ℝ :=
  (∑ d : Fin c.q.dims, a.query (c.start b+t) h d.val *
    a.key (c.start b+j) h d.val) * c.scale

def expected (c : Call) (a : Inputs) (b h t d : Nat) : ℝ :=
  (∑ j ∈ Finset.range (t+1), Real.exp (score c a b h t j) *
    a.value (c.start b+j) h d) /
    ∑ j ∈ Finset.range (t+1), Real.exp (score c a b h t j)

/-- Ordinary resolve_neg coordinate contract, for each actual read role. The
normalization can change all three strides independently. Integer metadata's
coordinate contract is already explicit in Host.Memory.start/length; Python's
contiguous() supplies that ABI. These are setup contracts, not kernel laws. -/
structure Normalizes (c : Call) (physical : VeriTile.Bench.TritonBenchG.ContextAttnNopad.Formula.Inputs) (a : Inputs) : Prop where
  query : ∀ row head dim, row < c.q.rows → head < c.q.heads → dim < c.q.dims →
    physical.q (c.q.offset row head dim) = a.query row head dim
  key : ∀ row head dim, row < c.k.rows → head < c.k.heads → dim < c.k.dims →
    physical.k (c.k.offset row head dim) = a.key row head dim
  value : ∀ row head dim, row < c.v.rows → head < c.v.heads → dim < c.v.dims →
    physical.v (c.v.offset row head dim) = a.value row head dim

theorem score_normalizes {c : Call} {physical : VeriTile.Bench.TritonBenchG.ContextAttnNopad.Formula.Inputs} {a : Inputs}
    (hs : Shape c) (hn : Normalizes c physical a)
    (b h t j : Nat) (hb : b < c.batches) (hh : h < c.q.heads)
    (ht : t < c.length b) (hj : j ≤ t) :
    c.score physical b h t j = score c a b h t j := by
  unfold Call.score score
  congr 1
  apply Finset.sum_congr rfl
  intro d _
  rw [hn.query _ _ _ (by have := hs.qRows b hb; omega) hh d.isLt,
      hn.key _ _ _ (by have := hs.kRows b hb; omega)
        (lt_of_lt_of_le hh hs.kHeads) (by simpa [hs.kDims] using d.isLt)]

theorem expected_normalizes {c : Call} {physical : VeriTile.Bench.TritonBenchG.ContextAttnNopad.Formula.Inputs} {a : Inputs}
    (hs : Shape c) (hn : Normalizes c physical a)
    (b h t d : Nat) (hb : b < c.batches) (hh : h < c.q.heads)
    (ht : t < c.length b) (hd : d < c.q.dims) :
    c.expected physical b h t d = expected c a b h t d := by
  unfold Call.expected expected
  congr 1
  · apply Finset.sum_congr rfl
    intro j hj
    have hj' : j ≤ t := by have := Finset.mem_range.mp hj; omega
    rw [score_normalizes hs hn b h t j hb hh ht hj',
      hn.value _ _ _ (by have := hs.vRows b hb; omega)
        (lt_of_lt_of_le hh hs.vHeads) (by simpa [hs.vDims] using hd)]
  · apply Finset.sum_congr rfl
    intro j hj
    have hj' : j ≤ t := by have := Finset.mem_range.mp hj; omega
    rw [score_normalizes hs hn b h t j hb hh ht hj']

/-- All logical coordinates of the supplied output, including cells outside
q's head/channel shape. Those extra cells must survive initialized writeback. -/
abbrev Coord (c : Call) := Fin c.o.rows × Fin c.o.heads × Fin c.o.dims

def coordOffset (c : Call) (i : Coord c) : Nat := c.o.offset i.1.val i.2.1.val i.2.2.val

def signed (negative : Bool) (x : ℝ) : ℝ := if negative then -x else x

@[simp] theorem signed_twice (negative : Bool) (x : ℝ) :
    signed negative (signed negative x) = x := by cases negative <;> simp [signed]

/-- Original caller-owned view, including identity and physical placement.
No normalized scratch allocation replaces these identity fields. -/
structure OutputView (c : Call) where
  objectId : Nat
  allocationId : Nat
  base : Nat
  rowStride : Nat
  headStride : Nat
  dimStride : Nat
  negative : Bool

def OutputView.address {c : Call} (v : OutputView c) (i : Coord c) : Nat :=
  v.base + i.1.val*v.rowStride + i.2.1.val*v.headStride + i.2.2.val*v.dimStride

def OutputView.read {c : Call} (v : OutputView c) (mem : Nat → ℝ) (i : Coord c) : ℝ :=
  signed v.negative (mem (v.address i))

/-- Tensor-library copy_ into the original logical view. It writes coordinates
and leaves every other cell of the original allocation untouched. -/
def OutputView.writeback {c : Call} (v : OutputView c) (before : Nat → ℝ)
    (values : Coord c → ℝ) : Nat → ℝ := fun addr =>
  if h : ∃ i : Coord c, v.address i = addr then
    signed v.negative (values (Classical.choose h)) else before addr

theorem writeback_read {c : Call} (v : OutputView c)
    (hinj : Function.Injective v.address) (before : Nat → ℝ)
    (values : Coord c → ℝ) (i : Coord c) :
    v.read (v.writeback before values) i = values i := by
  unfold OutputView.read OutputView.writeback
  have he : ∃ j : Coord c, v.address j = v.address i := ⟨i, rfl⟩
  rw [dif_pos he]
  have hi : Classical.choose he = i := hinj (Classical.choose_spec he)
  rw [hi, signed_twice]

theorem writeback_frame {c : Call} (v : OutputView c) (before : Nat → ℝ)
    (values : Coord c → ℝ) (written : Coord c → Prop)
    (hstable : ∀ i, ¬ written i → values i = v.read before i)
    (addr : Nat) (houtside : ¬ ∃ i, written i ∧ v.address i = addr) :
    v.writeback before values addr = before addr := by
  unfold OutputView.writeback
  split
  next h =>
    have hi : ¬ written (Classical.choose h) := by
      intro hw
      exact houtside ⟨_, hw, Classical.choose_spec h⟩
    rw [hstable _ hi]
    simp only [OutputView.read, signed_twice, Classical.choose_spec h]
  next => rfl

/-- Output initialization done by resolve_neg. All coordinates are copied,
including every under-dispatched row and every extra output head/channel. -/
def Initialized (c : Call) (r : Regions) (s : BlockState) (v : OutputView c)
    (before : Nat → ℝ) : Prop :=
  ∀ i : Coord c, s.readMem r.out (coordOffset c i) = v.read before i

structure CallerResult (c : Call) where
  objectId : Nat
  allocationId : Nat
  memory : Nat → ℝ
  pythonReturn : Option Unit

def finish {c : Call} (v : OutputView c) (before : Nat → ℝ)
    (values : Coord c → ℝ) : CallerResult c :=
  ⟨v.objectId, v.allocationId, v.writeback before values, none⟩

theorem finish_identity {c : Call} (v : OutputView c) (before : Nat → ℝ)
    (values : Coord c → ℝ) :
    (finish v before values).objectId = v.objectId ∧
    (finish v before values).allocationId = v.allocationId ∧
    (finish v before values).pythonReturn = none := ⟨rfl, rfl, rfl⟩

end
end VeriTile.Bench.Repairs.ContextAttnNopad.Logical
