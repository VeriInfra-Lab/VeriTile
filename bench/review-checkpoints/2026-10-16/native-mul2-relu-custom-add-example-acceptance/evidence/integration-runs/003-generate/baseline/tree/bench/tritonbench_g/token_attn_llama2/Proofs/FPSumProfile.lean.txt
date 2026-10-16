import bench.tritonbench_g.token_attn_llama2.Proofs.FPReference
import VeriTile.Triton.Float.ScalarBackend
import VeriTile.Triton.Float.TermModel
namespace VeriTile.Bench.TritonBenchG.TokenAttnLlama2.FPValue
open VeriTile.Triton FP.Structural
open FP.Equational (ReductionTree ReductionPlan)
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false
abbrev Word := FP.Value .fp32

/-- Exactly the supplied addition tree: all input leaves and every zero seed
are retained. No permutation, reassociation or zero law is used. -/
def treeValue (cfg : FP.Config) (xs : Fin n → Word) : ReductionTree n → Word
  | .input i => xs i
  | .zero => FP.Value.zero .fp32
  | .add a b => FP.add cfg (treeValue cfg xs a) (treeValue cfg xs b)

def treeSum (cfg : FP.Config) (plans : ∀ n, ReductionPlan n)
    {shape : TileShape} (ax : Fin shape.length) (keep : Bool)
    (xs : Values Word .real shape) : Values Word .real (TileShape.reduceShape shape ax keep) := by
  cases keep
  · exact fun out => treeValue cfg
      (fun k => xs (TileShape.insertAxisIndex shape ax out k)) (plans _).tree
  · exact fun out => treeValue cfg
      (fun k => xs (TileShape.replaceAxisIndex shape ax out k)) (plans _).tree

/-- Scalar conversion remains explicit. The input Real channel cannot encode
host NaNs, infinities or the sign of zero; those require a separate runtime
argument binding. All nonzero values are supplied to the named encoder. -/
def scaleWord (encode : ℝ → Word) (x : ℝ) : Word :=
  if x = 0 then FP.Value.zero .fp32 else encode x

/-- Closed fp32 arithmetic and explicit reduction schedule. The ONLY reached
nonconcrete primitive is the separately named host-scalar encoder. `rest` is
not used by the independent result proved below. -/
def softwareBackend (rest : Algebra Word) (cfg : FP.Config)
    (plans : ∀ n, ReductionPlan n) (encode : ℝ → Word) : Algebra Word :=
  { FP.Software.scalarFP32 rest cfg with
    literal := fun p d x =>
      if (p = none ∨ p = some .fp32) ∧ (d = .real ∨ d = .fp32) then scaleWord encode x
      else rest.literal p d x
    cast := fun p src dst x =>
      if (p = some .fp32 ∧ src = .fp32 ∧ dst = .real) ∨
         ((p = none ∨ p = some .fp32) ∧ src = .real ∧ dst = .fp32) then x
      else (FP.Software.scalarFP32 rest cfg).cast p src dst x
    reduceSum := fun p _ ax keep xs =>
      if p = some .fp32 then treeSum cfg plans ax keep xs else rest.reduceSum p ax keep xs }

def wordProducts [Inhabited Word] (cfg : FP.Config) (p : Parameters) (s : State Word) :
    Values Word .real [p.tokens,p.dimensions] := fun i =>
  FP.mul cfg ((s.mem p.q (queryOffset p s i.2.1)).read .fp32)
    (if active p s (i.1,PUnit.unit) then
      (s.mem p.k (keyOffset p s (i.1,PUnit.unit) i.2.1)).read .fp32
     else FP.Value.zero .fp32)

def wordScore [Inhabited Word] (cfg : FP.Config) (plans : ∀ n, ReductionPlan n)
    (scale : Word) (p : Parameters) (s : State Word) (i : Index p) : Word :=
  FP.mul cfg (treeSum cfg plans ⟨1, by simp⟩ Bool.false (wordProducts cfg p s) i) scale

theorem products_software [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config)
    (plans : ∀ n, ReductionPlan n) (encode : ℝ → Word) (p : Parameters) (s : State Word) :
    products (softwareBackend rest cfg plans encode) p s = wordProducts cfg p s := by
  funext i
  simp [products, queryValue, keyValue, wordProducts, softwareBackend,
    FP.Software.scalarFP32, FP.Software.fp32Context, scaleWord, toFloat,
    show (0.0 : ℝ) = 0 by norm_num]

theorem score_software [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config)
    (plans : ∀ n, ReductionPlan n) (encode : ℝ → Word) (p : Parameters)
    (s : State Word) (i : Index p) :
    score (softwareBackend rest cfg plans encode) p s i =
      wordScore cfg plans (scaleWord encode p.scale) p s i := by
  unfold score
  rw [products_software]
  simp [softwareBackend, FP.Software.scalarFP32, FP.Software.fp32Context, wordScore]

/-- A source-specific binding for the only nonzero scalar literal reached in
this kernel, whose scale is q. It is not a general Real converter. Conversion
is executable direct fp32 rounding; q is arbitrary, including negatives and
overflow cases. Zero padding is handled independently by scaleWord. -/
def rationalEncoder (cfg : FP.Config) (q : Rat) (_ : ℝ) : Word := FP.round .fp32 cfg q

theorem rational_scale (cfg : FP.Config) (q : Rat) :
    scaleWord (rationalEncoder cfg q) (q : ℝ) = FP.round .fp32 cfg q := by
  by_cases hq : q = 0
  · subst q
    simp [scaleWord, rationalEncoder, FP.round]
  · have hr : (q : ℝ) ≠ 0 := by exact_mod_cast hq
    simp [scaleWord, hr, rationalEncoder]

#axiomsClean products_software
#axiomsClean score_software
#axiomsClean rational_scale
end
end VeriTile.Bench.TritonBenchG.TokenAttnLlama2.FPValue
