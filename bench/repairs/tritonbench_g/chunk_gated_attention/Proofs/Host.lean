import bench.repairs.tritonbench_g.chunk_gated_attention.Kernels
import bench.tritonbench_g.chunk_gated_attention.Proofs.RealHMath

/-! Independent logical target and reusable adapter facts. These are proved
facts about logical-coordinate normalization and copy-back. No premise supplies
a kernel's expected output or successful execution. Whole host/grid composition
is proved in HostAPI; native address and control refinements are in Address and
NativeControl. The historical normalization-only counterexample is retained
separately in NativeGap. -/
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.Host
open scoped BigOperators
noncomputable section

structure Inputs where
  rawGate : Nat → Nat → ℝ
  key : Nat → Nat → ℝ
  value : Nat → Nat → ℝ
  seed : Nat → Nat → ℝ

/-- Independent within-chunk prefix; indices are logical time and feature. -/
def chunkPrefix (BT : Nat) (a : Nat → Nat → ℝ) (t f : Nat) : ℝ :=
  ∑ u ∈ Finset.range (t+1), if t / BT * BT ≤ u then a u f else 0

def endpoint (T BT c : Nat) : Nat := min (c*BT+BT-1) (T-1)

def weight (a : Inputs) (T BT c f : Nat) : ℝ :=
  Real.exp (chunkPrefix BT a.rawGate (endpoint T BT c) f)

/-- The contraction ranges over valid logical times of this chunk only. -/
def contribution (a : Inputs) (T BT : Nat) (gateK : Bool) (c k v : Nat) : ℝ :=
  ∑ t ∈ Finset.range T,
    if c*BT ≤ t ∧ t < (c+1)*BT then
      if gateK then
        (a.key t k * Real.exp (chunkPrefix BT a.rawGate (endpoint T BT c) k -
          chunkPrefix BT a.rawGate t k)) * a.value t v
      else a.key t k * (a.value t v *
        Real.exp (chunkPrefix BT a.rawGate (endpoint T BT c) v - chunkPrefix BT a.rawGate t v))
    else 0

/-- Closed form independently specifies each pre-update history and final
state. Fresh history returns c=0..NT-1; optional ht receives c=NT. -/
def state (a : Inputs) (T BT : Nat) (gateK hasSeed : Bool) (c k v : Nat) : ℝ :=
  let f := if gateK then k else v
  (if hasSeed then a.seed k v else 0) * (∏ u ∈ Finset.range c, weight a T BT u f) +
    ∑ t ∈ Finset.range c, contribution a T BT gateK t k v *
      ∏ u ∈ Finset.Ico (t+1) c, weight a T BT u f

/-- Independent inputs of fwd_inner: its gate argument already contains
within-chunk prefixes. This function is also meaningful when called separately
from fwd_pre with an arbitrary supplied cumulative gate tensor. -/
structure InnerInputs where
  key : Nat → Nat → ℝ
  value : Nat → Nat → ℝ
  cumulativeGate : Nat → Nat → ℝ
  seed : Nat → Nat → ℝ

def innerWeight (a : InnerInputs) (T BT : Nat) (gateK : Bool) (c k v : Nat) : ℝ :=
  Real.exp (a.cumulativeGate (endpoint T BT c) (if gateK then k else v))

/-- Mathematical contraction over valid positions within a logical chunk. -/
def innerContribution (a : InnerInputs) (T BT : Nat) (gateK : Bool) (c k v : Nat) : ℝ :=
  ∑ lane : Fin BT,
    let t := c*BT+lane.val
    if t < T then
      let f := if gateK then k else v
      let w := Real.exp (a.cumulativeGate (endpoint T BT c) f - a.cumulativeGate t f)
      if gateK then (a.key t k*w)*a.value t v else a.key t k*(a.value t v*w)
    else 0

def innerState (a : InnerInputs) (T BT : Nat) (gateK hasSeed : Bool) (c k v : Nat) : ℝ :=
  (if hasSeed then a.seed k v else 0) *
      (∏ u ∈ Finset.range c, innerWeight a T BT gateK u k v) +
    ∑ t ∈ Finset.range c, innerContribution a T BT gateK t k v *
      ∏ u ∈ Finset.Ico (t+1) c, innerWeight a T BT gateK u k v

/-- Ordinary materialization contract: copy logical coordinates, fill the
new allocation outside its source shape with zero. -/
def zeroPad (a : Nat → Nat → ℝ) (T S t f : Nat) : ℝ :=
  if t < T ∧ f < S then a t f else 0

theorem zeroPad_valid (a : Nat → Nat → ℝ) (T S t f : Nat)
    (ht : t < T) (hf : f < S) : zeroPad a T S t f = a t f := by
  simp [zeroPad, ht, hf]

theorem zeroPad_time_tail (a : Nat → Nat → ℝ) (T S t f : Nat)
    (ht : T ≤ t) : zeroPad a T S t f = 0 := by
  simp [zeroPad, Nat.not_lt.mpr ht]

/-- Cumulative gates repeat their final logical value, unlike raw gates. -/
def repeatGate (a : Nat → Nat → ℝ) (T t f : Nat) : ℝ :=
  if t < T then a t f else a (T-1) f

theorem repeated_endpoint_is_clipped (a : Nat → Nat → ℝ) (T BT c f : Nat) :
    repeatGate a T (c*BT+BT-1) f = a (endpoint T BT c) f := by
  unfold repeatGate endpoint
  split_ifs with h
  · rw [Nat.min_eq_left (by omega)]
  · rw [Nat.min_eq_right (by omega)]

/-- Equality that makes the shared input/output stride interface correct after
canonical staging, including independently strided B/H source views. -/
theorem canonical_flatten (H T S b h t f : Nat) :
    (b*H+h)*(T*S)+t*S+f = ((b*H+h)*T+t)*S+f := by ring

/-- Every lane of a full padded tile is inside its padded logical dimension. -/
theorem full_tile_active (tiles block tile lane : Nat)
    (ht : tile < tiles) (hl : lane < block) :
    tile*block+lane < tiles*block := by
  have h := Nat.mul_le_mul_right block (Nat.succ_le_iff.mpr ht)
  nlinarith

/-- Real zero padding adds no contraction term, regardless of the gate. This
is not an IEEE rule for nonfinite exp results; those require local FP domains. -/
theorem padded_product_zero (gate v : ℝ) : (0 * Real.exp gate) * v = 0 := by simp

/-- Functional form of the ordinary tensor copy_ contract. Caller output
storage may alias inputs: `old` is the snapshot before final copy-back. -/
def copyBack {I A V : Type} (old : A → V) (view : I → A) (values : I → V) (p : A) : V := by
  classical
  exact if h : ∃ i, view i = p then values (Classical.choose h) else old p

theorem copyBack_readback {I A V : Type} (old : A → V) (view : I → A)
    (values : I → V) (hi : Function.Injective view) (i : I) :
    copyBack old view values (view i) = values i := by
  classical
  unfold copyBack
  split_ifs with h
  · have he := Classical.choose_spec h
    rw [hi he]
  · exact False.elim (h ⟨i,rfl⟩)

theorem copyBack_frame {I A V : Type} (old : A → V) (view : I → A)
    (values : I → V) (p : A) (hp : ¬ ∃ i, view i = p) :
    copyBack old view values p = old p := by
  classical
  simp [copyBack,hp]

/-- Lazy-negative outputs preserve logical values by encoding on write and
negating on read. No Torch internals are assumed. -/
theorem negative_view_readback {I A : Type} (old : A → ℝ) (view : I → A)
    (values : I → ℝ) (hi : Function.Injective view) (i : I) :
    -(copyBack old view (fun j => -(values j)) (view i)) = values i := by
  rw [copyBack_readback old view _ hi i]
  simp

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.Host
