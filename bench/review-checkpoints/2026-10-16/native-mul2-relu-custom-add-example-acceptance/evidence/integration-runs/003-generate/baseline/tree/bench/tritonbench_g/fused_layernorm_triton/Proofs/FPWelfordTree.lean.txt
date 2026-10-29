import VeriTile.Triton
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.FusedLayernormTriton.FPStages
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

structure WelfordTriple (α : Type) where
  mean : α
  m2 : α
  weight : α

/-- A fixed reduction tree is part of the execution profile. Different trees
are not equated: floating Welford combine is not assumed associative. -/
inductive ReductionTree (N : Nat) where
  | leaf : Fin N → ReductionTree N
  | node : ReductionTree N → ReductionTree N → ReductionTree N

def ReductionTree.leaves {N : Nat} : ReductionTree N → List (Fin N)
  | .leaf i => [i]
  | .node l r => l.leaves ++ r.leaves

def ReductionTree.Complete {N : Nat} (tree : ReductionTree N) : Prop :=
  tree.leaves.Perm (List.finRange N)

/-- Total availability of the actual floating equality primitive. No
reflexivity, equivalence relation or arithmetic property is imposed. -/
def withEquality {α : Type} (M : Algebra α) (equal32 : α → α → Bool) : Algebra α :=
  { M with compareEq := fun p d => if p=some .fp32 ∧ d=.real then some equal32 else M.compareEq p d }

/-- Exact helper graph from the pinned torch triton_helpers.py, including
both guards and left-associated source multiplications/additions. -/
def combineTriple {α : Type} (M : Algebra α) (equal32 : α → α → Bool)
    (left right : WelfordTriple α) : WelfordTriple α :=
  let zero := M.literal (some .fp32) .real 0.0
  let delta := if equal32 left.mean right.mean then zero
    else M.binary (some .fp32) .real .sub right.mean left.mean
  let weight := M.binary (some .fp32) .real .add left.weight right.weight
  let ratio := if equal32 weight zero then zero
    else M.binary (some .fp32) .real .div right.weight weight
  { mean := M.binary (some .fp32) .real .add left.mean (M.binary (some .fp32) .real .mul delta ratio)
    m2 := M.binary (some .fp32) .real .add
      (M.binary (some .fp32) .real .add left.m2 right.m2)
      (M.binary (some .fp32) .real .mul
        (M.binary (some .fp32) .real .mul (M.binary (some .fp32) .real .mul delta delta) left.weight) ratio)
    weight := weight }

def treeValue {α : Type} {N : Nat} (M : Algebra α) (equal32 : α → α → Bool)
    (values : Fin N → WelfordTriple α) : ReductionTree N → WelfordTriple α
  | .leaf i => values i
  | .node l r => combineTriple M equal32 (treeValue M equal32 values l) (treeValue M equal32 values r)

/-- The eager where nodes retain computation of both branches, as in
triton_helpers.welford_combine. Only pure intermediate expression substitution
is used; no reduction or real arithmetic rewrite is involved. -/
def combineExpressions (left right : WelfordTriple (Op .real [])) : WelfordTriple (Op .real []) :=
  let zero : Op .real [] := .const 0.0
  let delta := Op.where (.eq .real .nil left.mean right.mean) zero (.sub .real .nil right.mean left.mean)
  let weight := Op.add .real .nil left.weight right.weight
  let ratio := Op.where (.eq .real .nil weight zero) zero (.div .real .nil right.weight weight)
  { mean := .add .real .nil left.mean (.mul .real .nil delta ratio)
    m2 := .add .real .nil (.add .real .nil left.m2 right.m2)
      (.mul .real .nil (.mul .real .nil (.mul .real .nil delta delta) left.weight) ratio)
    weight := weight }

def treeExpressions {N : Nat} : ReductionTree N → WelfordTriple (Op .real [])
  | .leaf i =>
    { mean := Op.remap (shape := [1,N]) [] (fun _ => ⟨⟨0,by decide⟩,i,()⟩) (.ref .real [1,N] "tmp3_mean")
      m2 := Op.remap (shape := [1,N]) [] (fun _ => ⟨⟨0,by decide⟩,i,()⟩) (.ref .real [1,N] "tmp3_m2")
      weight := Op.remap (shape := [1,N]) [] (fun _ => ⟨⟨0,by decide⟩,i,()⟩) (.ref .real [1,N] "tmp3_weight") }
  | .node l r => combineExpressions (treeExpressions l) (treeExpressions r)

def EvaluatesTriple {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (expressions : WelfordTriple (Op .real [])) (values : WelfordTriple α) : Prop :=
  evalOp M (some .fp32) expressions.mean s = some (fun _ => values.mean) ∧
  evalOp M (some .fp32) expressions.m2 s = some (fun _ => values.m2) ∧
  evalOp M (some .fp32) expressions.weight s = some (fun _ => values.weight)

private theorem eval_add32 {α : Type} [Inhabited α] (M : Algebra α) (a b : Op .real []) (s : State α) :
    evalOp M (some .fp32) (.add .real .nil a b) s = (do
      let va ← evalOp M (some .fp32) a s
      let vb ← evalOp M (some .fp32) b s
      pure (fun i => M.binary (some .fp32) .real .add (va i) (vb i))) := by
  rw [evalOp_unfold]
  rfl

private theorem eval_sub32 {α : Type} [Inhabited α] (M : Algebra α) (a b : Op .real []) (s : State α) :
    evalOp M (some .fp32) (.sub .real .nil a b) s = (do
      let va ← evalOp M (some .fp32) a s
      let vb ← evalOp M (some .fp32) b s
      pure (fun i => M.binary (some .fp32) .real .sub (va i) (vb i))) := by
  rw [evalOp_unfold]
  rfl

private theorem eval_mul32 {α : Type} [Inhabited α] (M : Algebra α) (a b : Op .real []) (s : State α) :
    evalOp M (some .fp32) (.mul .real .nil a b) s = (do
      let va ← evalOp M (some .fp32) a s
      let vb ← evalOp M (some .fp32) b s
      pure (fun i => M.binary (some .fp32) .real .mul (va i) (vb i))) := by
  rw [evalOp_unfold]
  rfl

private theorem eval_div32 {α : Type} [Inhabited α] (M : Algebra α) (a b : Op .real []) (s : State α) :
    evalOp M (some .fp32) (.div .real .nil a b) s = (do
      let va ← evalOp M (some .fp32) a s
      let vb ← evalOp M (some .fp32) b s
      pure (fun i => M.binary (some .fp32) .real .div (va i) (vb i))) := by
  rw [evalOp_unfold]
  rfl

private theorem eval_eq32 {α : Type} [Inhabited α] (M : Algebra α) (equal32 : α→α→Bool)
    (a b : Op .real []) (s : State α) :
    evalOp (withEquality M equal32) (some .fp32) (.eq .real .nil a b) s = (do
      let va ← evalOp (withEquality M equal32) (some .fp32) a s
      let vb ← evalOp (withEquality M equal32) (some .fp32) b s
      pure (bop equal32 Broadcast.nil va vb)) := by
  rw [evalOp_unfold]
  simp [numericEq, withEquality, Option.bind, bop]

private theorem eval_where32 {α : Type} [Inhabited α] (M : Algebra α)
    (cond : Op .bool []) (a b : Op .real []) (s : State α) :
    evalOp M (some .fp32) (.where cond a b) s = (do
      let vc ← evalOp M (some .fp32) cond s
      let va ← evalOp M (some .fp32) a s
      let vb ← evalOp M (some .fp32) b s
      pure (fun i => if vc i then va i else vb i)) := by
  rw [evalOp_unfold]

private theorem eval_const32 {α : Type} [Inhabited α] (M : Algebra α) (x : ℝ) (s : State α) :
    evalOp M (some .fp32) (.const x) s = some (fun _ => M.literal (some .fp32) .real x) := by
  rw [evalOp_unfold]

theorem combine_expressions_eval {α : Type} [Inhabited α] (M : Algebra α)
    (equal32 : α → α → Bool) (s : State α) (left right : WelfordTriple (Op .real []))
    (lv rv : WelfordTriple α)
    (hl : EvaluatesTriple (withEquality M equal32) s left lv)
    (hr : EvaluatesTriple (withEquality M equal32) s right rv) :
    EvaluatesTriple (withEquality M equal32) s (combineExpressions left right) (combineTriple M equal32 lv rv) := by
  rcases hl with ⟨hlm,hl2,hlw⟩
  rcases hr with ⟨hrm,hr2,hrw⟩
  simp only [EvaluatesTriple, combineExpressions, combineTriple, eval_add32,eval_sub32,eval_mul32,eval_div32,eval_eq32,eval_where32,eval_const32,
    hlm,hl2,hlw,hrm,hr2,hrw, Option.bind, bind, pure]
  simp [withEquality, bop]

theorem tree_expressions_eval {α : Type} [Inhabited α] {N : Nat} (M : Algebra α)
    (equal32 : α → α → Bool) (s : State α) (values : Fin N → WelfordTriple α)
    (hm : s.regs .real [1,N] "tmp3_mean" = some (fun i => (values i.2.1).mean))
    (h2 : s.regs .real [1,N] "tmp3_m2" = some (fun i => (values i.2.1).m2))
    (hw : s.regs .real [1,N] "tmp3_weight" = some (fun i => (values i.2.1).weight))
    (tree : ReductionTree N) :
    EvaluatesTriple (withEquality M equal32) s (treeExpressions tree) (treeValue M equal32 values tree) := by
  induction tree with
  | leaf i => simp [EvaluatesTriple, treeExpressions, treeValue, evalOp_unfold, hm,h2,hw, Option.bind]
  | node l r hl hr => exact combine_expressions_eval M equal32 s _ _ _ _ hl hr

/-- Exact welford_reduce helper, including the non-arithmetic first branch. -/
def reduceStep {α : Type} (M : Algebra α) (firstIteration : Bool) (value : α)
    (previous : WelfordTriple α) : WelfordTriple α :=
  if firstIteration then ⟨value,M.literal (some .fp32) .real 0,M.literal (some .fp32) .real 1⟩
  else
    let delta := M.binary (some .fp32) .real .sub value previous.mean
    let weight := M.binary (some .fp32) .real .add previous.weight (M.literal (some .fp32) .real 1)
    let mean := M.binary (some .fp32) .real .add previous.mean (M.binary (some .fp32) .real .div delta weight)
    ⟨mean, M.binary (some .fp32) .real .add previous.m2
      (M.binary (some .fp32) .real .mul delta (M.binary (some .fp32) .real .sub value mean)),weight⟩

def reduceExpressions (firstIteration : Bool) (value : Op .real [])
    (previous : WelfordTriple (Op .real [])) : WelfordTriple (Op .real []) :=
  if firstIteration then ⟨value,.const 0,.const 1⟩
  else
    let delta := Op.sub .real .nil value previous.mean
    let weight := Op.add .real .nil previous.weight (.const 1)
    let mean := Op.add .real .nil previous.mean (.div .real .nil delta weight)
    ⟨mean,.add .real .nil previous.m2 (.mul .real .nil delta (.sub .real .nil value mean)),weight⟩

theorem reduce_expressions_eval {α : Type} [Inhabited α] (M : Algebra α)
    (s : State α) (firstIteration : Bool) (value : Op .real []) (v : α)
    (previous : WelfordTriple (Op .real [])) (prev : WelfordTriple α)
    (hv : evalOp M (some .fp32) value s = some (fun _ => v))
    (hp : EvaluatesTriple M s previous prev) :
    EvaluatesTriple M s (reduceExpressions firstIteration value previous) (reduceStep M firstIteration v prev) := by
  rcases hp with ⟨hm,h2,hw⟩
  cases firstIteration <;> simp [EvaluatesTriple, reduceExpressions, reduceStep, eval_add32,eval_sub32,eval_mul32,eval_div32,eval_const32,
    hv,hm,h2,hw,Option.bind,bind,pure]

#axiomsClean combine_expressions_eval
#axiomsClean tree_expressions_eval
#axiomsClean reduce_expressions_eval
end
end VeriTile.Bench.TritonBenchG.FusedLayernormTriton.FPStages
