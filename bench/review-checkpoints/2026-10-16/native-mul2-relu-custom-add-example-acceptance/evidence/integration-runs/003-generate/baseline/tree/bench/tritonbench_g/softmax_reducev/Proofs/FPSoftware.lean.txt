import bench.tritonbench_g.softmax_reducev.Proofs.FPReference
import VeriTile.Triton.Float.ScalarBackend
import VeriTile.Triton.Float.TermModel
namespace VeriTile.Bench.TritonBenchG.SoftmaxReducev.FPValue
open VeriTile.Triton FP.Structural
open FP.Equational (ReductionTree ReductionPlan)
open ParallelAttention.FPStages (foldRange)
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 25000
set_option linter.unusedSimpArgs false
abbrev Word := FP.Value .fp32

/-- Explicit maximum-number software policy: suppress a lone NaN, canonicalize
both NaNs, prefer +0 on mixed zero ties, and apply the selected input FTZ policy. -/
def maximumWord (cfg : FP.Config) (x y : Word) : Word :=
  let x := FP.prepare cfg x
  let y := FP.prepare cfg y
  if x.isNaN then (if y.isNaN then FP.Value.qNaN .fp32 else y)
  else if y.isNaN then x
  else if x.isZero && y.isZero then FP.Value.zero .fp32 (x.sign && y.sign)
  else if FP.lt cfg y x then x else y

def treeValue (combine : Word → Word → Word) (seed : Word) (xs : Fin n → Word) :
    ReductionTree n → Word
  | .input i => xs i
  | .zero => seed
  | .add a b => combine (treeValue combine seed xs a) (treeValue combine seed xs b)

def treeReduce (plans : ∀ n, ReductionPlan n) (combine : Word → Word → Word) (seed : Word)
    {shape : TileShape} (ax : Fin shape.length) (keep : Bool) (xs : Values Word .real shape) :
    Values Word .real (TileShape.reduceShape shape ax keep) := by
  cases keep
  · exact fun out => treeValue combine seed
      (fun k => xs (TileShape.insertAxisIndex shape ax out k)) (plans _).tree
  · exact fun out => treeValue combine seed
      (fun k => xs (TileShape.replaceAxisIndex shape ax out k)) (plans _).tree

/-- Explicit fp32 scalar operations and two separately chosen, valid reduction
plans. `exp32` is the only nonconcrete scalar primitive used by the result.
Unrelated backend callbacks remain in `rest`; their absence from the result is
proved by output_software below. -/
def softwareBackend (rest : Algebra Word) (cfg : FP.Config) (exp32 : Word → Word)
    (sumPlans maxPlans : ∀ n, ReductionPlan n) : Algebra Word :=
  let base := FP.Software.scalarFP32 rest cfg
  { base with
    literal := fun p d r => if (p = none ∨ p = some .fp32) ∧ (d = .real ∨ d = .fp32) ∧ r = 0
      then FP.Value.zero .fp32 else base.literal p d r
    binary := fun p d op a b => if FP.Software.fp32Context p d && decide (op = .max)
      then maximumWord cfg a b else base.binary p d op a b
    unary := fun p op a => if p = some .fp32 ∧ op = .exp then exp32 a else base.unary p op a
    cast := fun p src dst x =>
      if (p = some .fp32 ∧ src = .fp32 ∧ dst = .real) ∨
        ((p = none ∨ p = some .fp32) ∧ src = .real ∧ dst = .fp32) then x
      else base.cast p src dst x
    reduceSum := fun p _ ax keep xs => if p = some .fp32 then
      treeReduce sumPlans (FP.add cfg) (FP.Value.zero .fp32) ax keep xs
      else rest.reduceSum p ax keep xs
    reduceMax := fun p _ ax keep xs => if p = some .fp32 then
      treeReduce maxPlans (maximumWord cfg) (FP.Value.infinity .fp32 Bool.true) ax keep xs
      else rest.reduceMax p ax keep xs }

/-- The extra typed-storage annotations materialize only the original fp32
load-to-register and register-to-store boundaries. For every bit pattern these
same-format conversions and the direct fp32 load callback preserve payloads. -/
theorem storage_register_identity (rest : Algebra Word) (cfg : FP.Config) (exp32 : Word → Word)
    (sumPlans maxPlans : ∀ n, ReductionPlan n) (x : Word) :
    (softwareBackend rest cfg exp32 sumPlans maxPlans).cast (some .fp32) .fp32 .real x = x ∧
    (softwareBackend rest cfg exp32 sumPlans maxPlans).cast none .real .fp32 x = x ∧
    (softwareBackend rest cfg exp32 sumPlans maxPlans).cast (some .fp32) .real .fp32 x = x ∧
    (softwareBackend rest cfg exp32 sumPlans maxPlans).fp32Load x = x := by
  simp [softwareBackend,FP.Software.scalarFP32]

def wordLogits [Inhabited Word] (p : Parameters) (s : State Word) (start : Nat) :
    Values Word .real [p.tokens] := fun i =>
  if active p s start i.1 then (s.mem p.logics (logicOffset p s start i.1)).read .fp32
  else FP.Value.infinity .fp32 Bool.true

def wordValues [Inhabited Word] (p : Parameters) (s : State Word) (start : Nat) :
    Values Word .real [p.tokens,p.dimensions] := fun i =>
  (s.mem p.values (valueOffset p s start i)).read .fp32

def wordInitial (p : Parameters) : Accumulator Word p.dimensions :=
  ⟨FP.Value.infinity .fp32 Bool.true, FP.Value.zero .fp32, fun _ => FP.Value.zero .fp32⟩

def wordUpdate [Inhabited Word] (cfg : FP.Config) (exp32 : Word → Word)
    (sumPlans maxPlans : ∀ n, ReductionPlan n) (p : Parameters) (s : State Word)
    (start : Nat) (old : Accumulator Word p.dimensions) : Accumulator Word p.dimensions :=
  let q := wordLogits p s start
  let nextMax := maximumWord cfg
    (treeReduce maxPlans (maximumWord cfg) (FP.Value.infinity .fp32 Bool.true)
      ⟨0,by simp⟩ Bool.false q ()) old.maximum
  let oldScale := exp32 (FP.sub cfg old.maximum nextMax)
  let weights : Values Word .real [p.tokens] := fun i => exp32 (FP.sub cfg (q i) nextMax)
  let products : Values Word .real [p.tokens,p.dimensions] := fun i =>
    FP.mul cfg (weights (i.1,())) (wordValues p s start i)
  { maximum := nextMax
    normalizer := FP.add cfg (FP.mul cfg old.normalizer oldScale)
      (treeReduce sumPlans (FP.add cfg) (FP.Value.zero .fp32) ⟨0,by simp⟩ Bool.false weights ())
    output := fun i => FP.add cfg (FP.mul cfg (old.output i) oldScale)
      (treeReduce sumPlans (FP.add cfg) (FP.Value.zero .fp32) ⟨0,by simp⟩ Bool.false products i) }

def wordAccumulator [Inhabited Word] (cfg : FP.Config) (exp32 : Word → Word)
    (sumPlans maxPlans : ∀ n, ReductionPlan n) (p : Parameters) (s : State Word) :=
  foldRange (wordUpdate cfg exp32 sumPlans maxPlans p s) 0 (sequenceLength p s) p.tokens (wordInitial p)

def wordOutput [Inhabited Word] (cfg : FP.Config) (exp32 : Word → Word)
    (sumPlans maxPlans : ∀ n, ReductionPlan n) (p : Parameters) (s : State Word) : Index p → Word :=
  let a := wordAccumulator cfg exp32 sumPlans maxPlans p s
  fun i => FP.div cfg (a.output i) a.normalizer

theorem logits_software [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config) (exp32 : Word → Word)
    (sumPlans maxPlans : ∀ n, ReductionPlan n) (p : Parameters) (s : State Word) (start : Nat) :
    logits (softwareBackend rest cfg exp32 sumPlans maxPlans) p s start = wordLogits p s start := by
  funext i
  simp [logits,wordLogits,softwareBackend,FP.Software.scalarFP32]

theorem values_software [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config) (exp32 : Word → Word)
    (sumPlans maxPlans : ∀ n, ReductionPlan n) (p : Parameters) (s : State Word) (start : Nat) :
    values (softwareBackend rest cfg exp32 sumPlans maxPlans) p s start = wordValues p s start := by
  funext i
  simp [values,wordValues,softwareBackend,FP.Software.scalarFP32]

theorem update_software [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config) (exp32 : Word → Word)
    (sumPlans maxPlans : ∀ n, ReductionPlan n) (p : Parameters) (s : State Word) :
    update (softwareBackend rest cfg exp32 sumPlans maxPlans) p s =
      wordUpdate cfg exp32 sumPlans maxPlans p s := by
  funext start a
  unfold update
  rw [logits_software,values_software]
  simp [wordUpdate,softwareBackend,FP.Software.scalarFP32,FP.Software.fp32Context]

theorem output_software [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config) (exp32 : Word → Word)
    (sumPlans maxPlans : ∀ n, ReductionPlan n) (p : Parameters) (s : State Word) :
    outputValues (softwareBackend rest cfg exp32 sumPlans maxPlans) p
      (finalAccumulator (softwareBackend rest cfg exp32 sumPlans maxPlans) p s) =
      wordOutput cfg exp32 sumPlans maxPlans p s := by
  unfold finalAccumulator
  rw [update_software]
  funext i
  simp [wordOutput,wordAccumulator,outputValues,initialAccumulator,wordInitial,softwareBackend,
    FP.Software.scalarFP32,FP.Software.fp32Context,show (0.0 : ℝ)=0 by norm_num]
#axiomsClean storage_register_identity
#axiomsClean logits_software
#axiomsClean values_software
#axiomsClean update_software
#axiomsClean output_software
end
end VeriTile.Bench.TritonBenchG.SoftmaxReducev.FPValue
