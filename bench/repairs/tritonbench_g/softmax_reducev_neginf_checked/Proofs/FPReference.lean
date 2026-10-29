import bench.repairs.tritonbench_g.softmax_reducev_neginf_checked.Proofs.FPSource
import bench.repairs.tritonbench_g.softmax_reducev_neginf_checked.Proofs.Range
namespace VeriTile.Bench.Repairs.SoftmaxReduceVNegInfChecked.FPValue
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.SoftmaxReduceVNegInfChecked.Range (foldRange)
noncomputable section
abbrev Index (p : Parameters) := TileIndex [p.dimensions]

def sequenceLength {α : Type} [Inhabited α] (p : Parameters) (s : State α) : Nat :=
  (s.mem (Region.cast p.lengths) (s.pids 0)).read .nat

def sequenceStart {α : Type} [Inhabited α] (p : Parameters) (s : State α) : Nat :=
  (s.mem (Region.cast p.starts) (s.pids 0)).read .nat

def locationBase {α : Type} [Inhabited α] (p : Parameters) (s : State α) : Nat :=
  s.pids 0 * p.locationBatchStride + (p.maxLength - sequenceLength p s) * p.locationSequenceStride

def active {α : Type} [Inhabited α] (p : Parameters) (s : State α) (start : Nat) (i : Fin p.tokens) : Prop :=
  start+i.val  <  sequenceLength p s
instance {α : Type} [Inhabited α] (p : Parameters) (s : State α) (start : Nat) (i : Fin p.tokens) :
    Decidable (active p s start i) := inferInstanceAs (Decidable (_  <  _))

def location {α : Type} [Inhabited α] (p : Parameters) (s : State α) (start : Nat) (i : Fin p.tokens) : Int :=
  if active p s start i then
    (s.mem (Region.cast p.locations) (locationBase p s+(start+i.val)*p.locationSequenceStride)).read .int
  else 0

def logicOffset {α : Type} [Inhabited α] (p : Parameters) (s : State α) (start : Nat) (i : Fin p.tokens) :=
  s.pids 1 * p.logicHeadStride + (sequenceStart p s + start+i.val)*p.logicSequenceStride

def valueOffset {α : Type} [Inhabited α] (p : Parameters) (s : State α) (start : Nat)
    (i : TileIndex [p.tokens,p.dimensions]) : Nat :=
  s.pids 1*p.valueHeadStride+i.2.1.val*p.valueDimStride+
    (location p s start i.1 * (p.valueBatchStride : Int)).toNat -
    (-(location p s start i.1 * (p.valueBatchStride : Int))).toNat

def outputOffset {α : Type} (p : Parameters) (s : State α) (i : Index p) : Nat :=
  s.pids 0*p.outputBatchStride+s.pids 1*p.outputHeadStride+i.1.val*p.outputDimStride

/-- Both operand arrays are read from the original memory, before any output
store. V is zero-filled outside active key lanes; no sentinel address is read. -/
def logits {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) (start : Nat) :
    Values α .real [p.tokens] := fun i =>
  M.cast (some .fp32) .fp32 .real (if active p s start i.1 then
    (s.mem p.logics (logicOffset p s start i.1)).read .fp32
    else M.cast none .real .fp32 M.negInf)
def values {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) (start : Nat) :
    Values α .real [p.tokens,p.dimensions] := fun i =>
  M.cast (some .fp32) .fp32 .real (if active p s start i.1 then
    (s.mem p.values (valueOffset p s start i)).read .fp32
    else M.cast none .real .fp32 (M.literal none .real 0.0))

structure Accumulator (α : Type) (D : Nat) where
  maximum : α
  normalizer : α
  output : TileIndex [D] → α

def initialAccumulator {α : Type} (M : Algebra α) (p : Parameters) : Accumulator α p.dimensions :=
  ⟨M.negInf,M.literal (some .fp32) .real 0.0,fun _ => M.literal (some .fp32) .real 0⟩

/-- Independent operator-by-operator online recurrence. In particular no
exp/log identity, associativity, or cancellation at infinities is used. -/
def update {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α)
    (start : Nat) (old : Accumulator α p.dimensions) : Accumulator α p.dimensions :=
  let q := logits M p s start
  let nextMax := M.binary (some .fp32) .real .max
    (M.reduceMax (some .fp32) ⟨0,by simp⟩ Bool.false q ()) old.maximum
  let nextMax := M.binary (some .fp32) .real .max nextMax
    (M.literal (some .fp32) .real (-340282346638528859811704183484516925440.0))
  let oldScale := M.unary (some .fp32) .exp
    (M.binary (some .fp32) .real .sub old.maximum nextMax)
  let weights : Values α .real [p.tokens] := fun i => M.unary (some .fp32) .exp
    (M.binary (some .fp32) .real .sub (q i) nextMax)
  let products : Values α .real [p.tokens,p.dimensions] := fun i =>
    M.binary (some .fp32) .real .mul (weights (i.1,())) (values M p s start i)
  { maximum := nextMax
    normalizer := M.binary (some .fp32) .real .add
      (M.binary (some .fp32) .real .mul old.normalizer oldScale)
      (M.reduceSum (some .fp32) ⟨0,by simp⟩ Bool.false weights ())
    output := fun i => M.binary (some .fp32) .real .add
      (M.binary (some .fp32) .real .mul (old.output i) oldScale)
      (M.reduceSum (some .fp32) ⟨0,by simp⟩ Bool.false products i) }

def finalAccumulator {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) :=
  foldRange (update M p s) 0 (sequenceLength p s) p.tokens (initialAccumulator M p)

def outputValues {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (a : Accumulator α p.dimensions) : Index p → α := fun i =>
  M.cast none .real .fp32 (M.binary (some .fp32) .real .div (a.output i) a.normalizer)

def written {α : Type} (p : Parameters) (s : State α) (v : Index p → α) : State α :=
  (TileShape.allIndices [p.dimensions]).foldl
    (fun t i => t.write p.output (outputOffset p s i) (.mk .fp32 (v i))) s

end
end VeriTile.Bench.Repairs.SoftmaxReduceVNegInfChecked.FPValue
