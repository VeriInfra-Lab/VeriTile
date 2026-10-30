import bench.tritonbench_g.token_attn_llama2.Proofs.FPTypedSource
namespace VeriTile.Bench.Repairs.TokenAttnLlama2.FPValue
open VeriTile.Triton FP.Structural
noncomputable section
abbrev Index (p : Parameters) := TileIndex [p.tokens]

def token {α : Type} (p : Parameters) (s : State α) (i : Index p) : Nat :=
  s.pids 2 * p.tokens + i.1.val

def windowIndex {α : Type} [Inhabited α] (p : Parameters) (s : State α) (i : Index p) : Nat :=
  p.maxLength - length p s + token p s i

def active {α : Type} [Inhabited α] (p : Parameters) (s : State α) (i : Index p) : Prop :=
  windowIndex p s i < p.maxLength
instance {α : Type} [Inhabited α] (p : Parameters) (s : State α) : DecidablePred (active p s) :=
  fun _ => inferInstanceAs (Decidable (_ < _))

def queryOffset {α : Type} (p : Parameters) (s : State α) (d : Fin p.dimensions) : Nat :=
  s.pids 0 * p.queryBatchStride + s.pids 1 * p.queryHeadStride + d.val * p.queryDimStride

def keyLocation {α : Type} [Inhabited α] (p : Parameters) (s : State α) (i : Index p) : Nat :=
  if active p s i then
    (s.mem (Region.cast p.locations)
      (p.locationBatchStride * s.pids 0 + p.locationSequenceStride * windowIndex p s i)).read .nat
  else 0

def keyOffset {α : Type} [Inhabited α] (p : Parameters) (s : State α)
    (i : Index p) (d : Fin p.dimensions) : Nat :=
  keyLocation p s i * p.keyBatchStride + (s.pids 1 / p.groups) * p.keyHeadStride +
    d.val * p.keyDimStride

def outputOffset {α : Type} [Inhabited α] (p : Parameters) (s : State α) (i : Index p) : Nat :=
  s.pids 1 * p.outputHeadStride +
    ((s.mem (Region.cast p.starts) (s.pids 0)).read .nat + token p s i) * p.outputSequenceStride

/-- All leaves come from the initial memory; the two input channels are fp32.
The casts materialize fp32 register loading and remain explicit for an arbitrary backend. -/
def queryValue {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α)
    (d : Fin p.dimensions) : α :=
  M.cast (some .fp32) .fp32 .real ((s.mem p.q (queryOffset p s d)).read .fp32)

def keyValue {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α)
    (i : Index p) (d : Fin p.dimensions) : α :=
  M.cast (some .fp32) .fp32 .real
    (if active p s i then (s.mem p.k (keyOffset p s i d)).read .fp32
     else M.cast (some .fp32) .real .fp32 (M.literal (some .fp32) .real 0.0))

def products {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) :
    Values α .real [p.tokens,p.dimensions] := fun i =>
  M.binary (some .fp32) .real .mul (queryValue M p s i.2.1)
    (keyValue M p s (i.1,PUnit.unit) i.2.1)

/-- Whole row reduction followed by the original scalar multiplication and
output storage conversion. Scale conversion is an explicit backend operation. -/
def score {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α)
    (i : Index p) : α :=
  M.cast none .real .fp32
    (M.binary (some .fp32) .real .mul
      (M.reduceSum (some .fp32) ⟨1, by simp⟩ Bool.false (products M p s) i)
      (M.literal (some .fp32) .real p.scale))

def written {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) : State α :=
  if blockActive p s then
    (TileShape.allIndices [p.tokens]).foldl (fun t i =>
      if active p s i then t.write p.output (outputOffset p s i) (.mk .fp32 (score M p s i))
      else t) s
  else s

/-- Address injectivity is the original output-layout condition. -/
def WritesApart {α : Type} [Inhabited α] (p : Parameters) (s : State α) : Prop :=
  ∀ i j, active p s i → active p s j → outputOffset p s i = outputOffset p s j → i = j

def Post {α : Type} [Inhabited α] (p : Parameters) (initial final : State α)
    (results : Index p → α) : Prop :=
  (∀ i, blockActive p initial → active p initial i →
    final.mem p.output (outputOffset p initial i) = .mk .fp32 (results i)) ∧
  (∀ r o, (¬ blockActive p initial ∨ r ≠ p.output ∨
    ∀ i, active p initial i → o ≠ outputOffset p initial i) → final.mem r o = initial.mem r o)
end
end VeriTile.Bench.Repairs.TokenAttnLlama2.FPValue
