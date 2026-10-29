import bench.tritonbench_g.bgmv_expand_slice.Kernels
import VeriTile.Triton.Float.RewriteTactics
import VeriTile.Meta.StatementAudit

/-! Conditional equivalence of the complete BGMV sources for every storage
configuration. The premise is native scalar addition commutativity at the
actual dtype and compute precision, on finite operands. It is deliberately
separate from the fp32 experiment rules: those rules cannot supply the native
fp16/bf16 premises. The domain follows only reached additions, with the
original cast, reduction, masks and loop state preserved. -/
namespace VeriTile.Bench.TritonBenchG.BgmvExpandSlice.NativeFP
open VeriTile Triton FP.Structural FP.Guarded
open _root_.VeriTile.Triton.FP.ProfiledRewrite (engine Contextual)
open _root_.VeriTile.Triton.FP.GuardedRewrite (sameShape sameShape_left sameShape_right)
set_option maxHeartbeats 5000000
set_option maxRecDepth 15000

/-- Scalar backend law, not a kernel equality or an asserted hardware fact.
The operation's precision and dtype remain distinct parameters. -/
def ScalarAdditionCommutes {α : Type} (M : Algebra α) (D : Domain α)
    (p : Option ComputeDType) (dtype : FloatDType) : Prop :=
  ∀ a b, D .finite a → D .finite b →
    (engine M).binary p dtype .add a b = (engine M).binary p dtype .add b a

/-- Existing admitted rules already supply the fp32 scalar law in either
source context. They say nothing here about native half/bfloat16 addition. -/
theorem fp32_scalar_of_rules {α : Type} [Inhabited α]
    (R : FP.ScalarArithmetic.Rules) (M : Algebra α) (D : Domain α)
    (hM : FP.Guarded.Models R.assumptions M D) (s : State α) :
    ScalarAdditionCommutes M D none .real ∧
    ScalarAdditionCommutes M D (some .fp32) .real := by
  constructor <;> intro a b ha hb
  all_goals exact FP.ScalarArithmetic.add_comm R M D hM s a b ha hb

inductive Site where
  | add (before : List ComputeStmt) (precision : Option ComputeDType)
      (dtype : FloatDType) (shape : TileShape)
      (left right : Op dtype.toTileDType shape)
  | range (before : List ComputeStmt) (index : RegName)
      (start stop stride : Op .nat []) (body : List ComputeStmt) (inner : Site)
  | branch (before : List ComputeStmt) (condition : Op .bool [])
      (selected : Bool) (inner : Site)

def Site.Holds {α : Type} [Inhabited α] (M : Algebra α) (D : Domain α)
    (s : State α) : Site → Prop
  | .add before precision dtype _ left right =>
    ∀ t, run (engine M) before s = some t →
      ∀ a b, FP.Structural.evalOp (engine M) precision left t = some a →
        FP.Structural.evalOp (engine M) precision right t = some b →
        (∀ i, D .finite (toFloat dtype (a i))) ∧
        (∀ i, D .finite (toFloat dtype (b i)))
  | .range before index start stop stride body inner =>
    ∀ t, run (engine M) before s = some t →
      ∀ a b d, FP.Structural.evalOp (engine M) none start t = some a →
        FP.Structural.evalOp (engine M) none stop t = some b →
        FP.Structural.evalOp (engine M) none stride t = some d →
        ∀ cur u, RangeReachable (engine M) index (b PUnit.unit) (d PUnit.unit)
          body (a PUnit.unit) t cur u → d PUnit.unit ≠ 0 → cur < b PUnit.unit →
          inner.Holds M D (u.setReg index .nat [] (fun _ => cur))
  | .branch before condition selected inner =>
    ∀ t, run (engine M) before s = some t →
      ∀ value, FP.Structural.evalOp (engine M) none condition t = some value →
        value PUnit.unit = selected → inner.Holds M D t

structure AdditionOperands (shape : TileShape) where
  dtype : FloatDType
  left : Op dtype.toTileDType shape
  right : Op dtype.toTileDType shape
  precision : Option ComputeDType := none

/-- Find the actual addition under the explicit output conversion. -/
def additionOperands? : {dtype : TileDType} → {shape : TileShape} →
    Op dtype shape → Option (AdditionOperands shape)
  | _, _, .add .real (.consSame .nil) left right => some ⟨.real, left, right, none⟩
  | _, _, .add .fp32 (.consSame .nil) left right => some ⟨.fp32, left, right, none⟩
  | _, _, .add .fp16 (.consSame .nil) left right => some ⟨.fp16, left, right, none⟩
  | _, _, .add .bf16 (.consSame .nil) left right => some ⟨.bf16, left, right, none⟩
  | _, _, .add .f8e4 (.consSame .nil) left right => some ⟨.f8e4, left, right, none⟩
  | _, _, .add .f8e5 (.consSame .nil) left right => some ⟨.f8e5, left, right, none⟩
  | _, _, .castFloat _ _ operand => additionOperands? operand
  | _, _, .withPrecision _ scope operand =>
      (additionOperands? operand).map fun a =>
        { a with precision := (a.precision.or
            (some (match scope with | .fp32 => .fp32 | .fp64 => .fp64))) }
  | _, _, _ => none

def additionSite? (code : List ComputeStmt) : Option Site :=
  let before := Source.prefixBefore code "accumulator"
  match (code.drop before.length).head? with
  | some (.assign _ shape _ (.alg operand)) =>
      (additionOperands? operand).map (fun a => .add before a.precision a.dtype shape a.left a.right)
  | some (.assign _ shape _ (.compute (.alg p operand))) =>
      (additionOperands? operand).map (fun a => .add before (a.precision.or (some p)) a.dtype shape a.left a.right)
  | _ => none

def requirementsFor (kernel : ComputeKernel) : List Site :=
  (do
    let active ← Source.conditional? kernel.surfaceBody
      (fun body => (Source.dynamicRange? body "n").isSome)
    let loop ← Source.dynamicRange? active.yes "n"
    let accumulation ← Source.conditional? loop.body (fun _ => Bool.true)
    let addition ← additionSite? accumulation.yes
    return Site.branch active.before active.condition Bool.true
      (.range loop.before "n" loop.start loop.stop loop.stride loop.body
        (.branch accumulation.before accumulation.condition Bool.true addition))).toList

/-- Read the precision of the actual source addition, through its control flow. -/
def Site.profile : Site → Option ComputeDType × FloatDType
  | .add _ precision dtype _ _ _ => (precision, dtype)
  | .range _ _ _ _ _ _ inner => inner.profile
  | .branch _ _ _ inner => inner.profile

/-- Only native addition laws selected by the source are requested.
There are no laws for output casts, products, reductions, or unused formats. -/
def NativeAdditionCommutes {α : Type} (M : Algebra α) (D : Domain α)
    (kernel : ComputeKernel) : Prop :=
  ∀ site ∈ requirementsFor kernel,
    ScalarAdditionCommutes M D site.profile.1 site.profile.2

theorem contextual_range {α : Type} [Inhabited α] (M : Algebra α)
    (D : Domain α) (s : State α) (before after : List ComputeStmt)
    (index : RegName) (start stop stride : Op .nat [])
    (lhs rhs : List ComputeStmt) (inner : Site)
    (hd : (Site.range before index start stop stride lhs inner).Holds M D s)
    (hbody : ∀ t, inner.Holds M D t → run (engine M) lhs t = run (engine M) rhs t) :
    Contextual M s before after (.forRangeDyn index start stop stride lhs)
      (.forRangeDyn index start stop stride rhs) := by
  unfold Contextual
  rw [run_append, run_append]
  cases hp : run (engine M) before s with
  | none => rfl
  | some t =>
    simp only [bind, Option.bind_some, run, step]
    cases ha : FP.Structural.evalOp (engine M) none start t with
    | none => rfl
    | some a =>
      simp only [Option.bind_some]
      cases hb : FP.Structural.evalOp (engine M) none stop t with
      | none => rfl
      | some b =>
        simp only [Option.bind_some]
        cases hdt : FP.Structural.evalOp (engine M) none stride t with
        | none => rfl
        | some d =>
          simp only [Option.bind_some]
          rw [range_congr_on_reachable (engine M) index (a PUnit.unit)
            (b PUnit.unit) (d PUnit.unit) lhs rhs t (fun cur u hr hz hc =>
              hbody _ (hd t hp a b d ha hb hdt cur u hr hz hc))]

theorem contextual_then {α : Type} [Inhabited α] (M : Algebra α)
    (D : Domain α) (s : State α) (before after : List ComputeStmt)
    (condition : Op .bool []) (lhs rhs otherwise : List ComputeStmt) (inner : Site)
    (hd : (Site.branch before condition Bool.true inner).Holds M D s)
    (hbody : ∀ t, inner.Holds M D t → run (engine M) lhs t = run (engine M) rhs t) :
    Contextual M s before after (.ifThenElse condition lhs otherwise)
      (.ifThenElse condition rhs otherwise) := by
  unfold Contextual
  rw [run_append, run_append]
  cases hp : run (engine M) before s with
  | none => rfl
  | some t =>
    simp only [Option.bind_some, run, step]
    cases hc : FP.Structural.evalOp (engine M) none condition t with
    | none => rfl
    | some value =>
      simp only [bind, Option.bind_some]
      cases hv : value PUnit.unit with
      | «false» => rfl
      | «true» => simp only [↓reduceIte]; rw [hbody t (hd t hp value hc hv)]

theorem contextual_if_then {α : Type} [Inhabited α] (M : Algebra α)
    (D : Domain α) (s : State α) (before after : List ComputeStmt)
    (condition : Op .bool []) (lhs rhs : List ComputeStmt) (inner : Site)
    (hd : (Site.branch before condition Bool.true inner).Holds M D s)
    (hbody : ∀ t, inner.Holds M D t → run (engine M) lhs t = run (engine M) rhs t) :
    Contextual M s before after (.ifThen condition lhs) (.ifThen condition rhs) := by
  unfold Contextual
  rw [run_append, run_append]
  cases hp : run (engine M) before s with
  | none => rfl
  | some t =>
    simp only [Option.bind_some, run, step]
    cases hc : FP.Structural.evalOp (engine M) none condition t with
    | none => rfl
    | some value =>
      simp only [bind, Option.bind_some]
      cases hv : value PUnit.unit with
      | «false» => rfl
      | «true» => simp only [↓reduceIte]; rw [hbody t (hd t hp value hc hv)]

private theorem numeric_comm {α : Type} (M : Algebra α) (D : Domain α)
    (p : Option ComputeDType) (dtype : FloatDType)
    (hc : ScalarAdditionCommutes M D p dtype)
    (a b : Value α dtype.toTileDType)
    (ha : D .finite (toFloat dtype a)) (hb : D .finite (toFloat dtype b)) :
    numeric (engine M) p .add dtype.numericDType a b =
      numeric (engine M) p .add dtype.numericDType b a := by
  cases dtype <;> exact hc _ _ ha hb

private theorem eval_add_comm {α : Type} [Inhabited α]
    (M : Algebra α) (D : Domain α)
    (s : State α) (p : Option ComputeDType) (dtype : FloatDType)
    (hc : ScalarAdditionCommutes M D p dtype)
    (a b : Op dtype.toTileDType shape)
    (hd : ∀ va vb, FP.Structural.evalOp (engine M) p a s = some va →
      FP.Structural.evalOp (engine M) p b s = some vb →
      (∀ i, D .finite (toFloat dtype (va i))) ∧
      (∀ i, D .finite (toFloat dtype (vb i)))) :
    FP.Structural.evalOp (engine M) p (.add dtype.numericDType (sameShape shape) a b) s =
      FP.Structural.evalOp (engine M) p (.add dtype.numericDType (sameShape shape) b a) s := by
  have eadd (x y : Op dtype.toTileDType shape) :
      FP.Structural.evalOp (engine M) p (.add dtype.numericDType (sameShape shape) x y) s =
        (FP.Structural.evalOp (engine M) p x s).bind (fun vx =>
          (FP.Structural.evalOp (engine M) p y s).bind (fun vy =>
            some (bop (numeric (engine M) p .add dtype.numericDType) (sameShape shape) vx vy))) := rfl
  rw [eadd, eadd]
  cases ha : FP.Structural.evalOp (engine M) p a s <;>
    cases hb : FP.Structural.evalOp (engine M) p b s
  all_goals simp only [Option.bind_none, Option.bind_some]
  rename_i va vb
  congr 1
  funext i
  simp only [bop, sameShape_left, sameShape_right]
  exact numeric_comm M D p dtype hc _ _ ((hd va vb ha hb).1 i) ((hd va vb ha hb).2 i)

private theorem eval_cast_congr {α : Type} [Inhabited α] (M : Algebra α)
    (s : State α) (p : Option ComputeDType) (input output : FloatDType)
    (a b : Op input.toTileDType shape)
    (h : FP.Structural.evalOp M p a s = FP.Structural.evalOp M p b s) :
    FP.Structural.evalOp M p (.castFloat input output a) s =
      FP.Structural.evalOp M p (.castFloat input output b) s := by
  have ecast (x : Op input.toTileDType shape) :
      FP.Structural.evalOp M p (.castFloat input output x) s =
        (FP.Structural.evalOp M p x s).bind (fun v =>
          some (fun i => ofFloat output (M.cast p input output (toFloat input (v i))))) := rfl
  rw [ecast, ecast, h]

/-- The final storage conversion is identical on both sides, with no cast law. -/
theorem contextual_cast_add {α : Type} [Inhabited α]
    (M : Algebra α) (D : Domain α)
    (s : State α) (before after : List ComputeStmt) (out : RegName)
    (dtype output : FloatDType) (a b : Op dtype.toTileDType shape)
    (hc : ScalarAdditionCommutes M D none dtype)
    (hd : (Site.add before none dtype shape a b).Holds M D s) :
    Contextual M s before after
      (.assign output.toTileDType shape out
        (.alg (.castFloat dtype output (.add dtype.numericDType (sameShape shape) a b))))
      (.assign output.toTileDType shape out
        (.alg (.castFloat dtype output (.add dtype.numericDType (sameShape shape) b a)))) := by
  unfold Contextual
  rw [run_append, run_append]
  cases hp : run (engine M) before s with
  | none => rfl
  | some t =>
    simp only [Option.bind_some, run, step, FP.Structural.evalExpr]
    rw [eval_cast_congr (engine M) t none dtype output _ _
      (eval_add_comm M D t none dtype hc a b (hd t hp))]

/-- A narrow store conversion preserves the explicit fp32 evaluation scope
of the addition underneath it. The output conversion is shared, without a cast law. -/
theorem contextual_cast_scoped_add_fp32 {α : Type} [Inhabited α]
    (M : Algebra α) (D : Domain α)
    (s : State α) (before after : List ComputeStmt) (out : RegName)
    (dtype output : FloatDType) (a b : Op dtype.toTileDType shape)
    (hc : ScalarAdditionCommutes M D (some .fp32) dtype)
    (hd : (Site.add before (some .fp32) dtype shape a b).Holds M D s) :
    Contextual M s before after
      (.assign output.toTileDType shape out
        (.alg (.castFloat dtype output (.withPrecision dtype .fp32
          (.add dtype.numericDType (sameShape shape) a b)))))
      (.assign output.toTileDType shape out
        (.alg (.castFloat dtype output (.withPrecision dtype .fp32
          (.add dtype.numericDType (sameShape shape) b a))))) := by
  unfold Contextual
  rw [run_append, run_append]
  cases hp : run (engine M) before s with
  | none => rfl
  | some t =>
    simp only [Option.bind_some, run, step, FP.Structural.evalExpr]
    have hs : FP.Structural.evalOp (engine M) none
        (.withPrecision dtype .fp32 (.add dtype.numericDType (sameShape shape) a b)) t =
        FP.Structural.evalOp (engine M) none
        (.withPrecision dtype .fp32 (.add dtype.numericDType (sameShape shape) b a)) t :=
      eval_add_comm M D t (some .fp32) dtype hc a b (hd t hp)
    rw [eval_cast_congr (engine M) t none dtype output _ _ hs]

/-- Explicit fp32 computation remains around the source's output conversion. -/
theorem contextual_cast_add_fp32 {α : Type} [Inhabited α]
    (M : Algebra α) (D : Domain α)
    (s : State α) (before after : List ComputeStmt) (out : RegName)
    (dtype : FloatDType) (a b : Op dtype.toTileDType shape)
    (hc : ScalarAdditionCommutes M D (some .fp32) dtype)
    (hd : (Site.add before (some .fp32) dtype shape a b).Holds M D s) :
    Contextual M s before after
      (.assign .real shape out
        (.compute (.alg .fp32 (.castFloat dtype .real (.add dtype.numericDType (sameShape shape) a b)))))
      (.assign .real shape out
        (.compute (.alg .fp32 (.castFloat dtype .real (.add dtype.numericDType (sameShape shape) b a))))) := by
  unfold Contextual
  rw [run_append, run_append]
  cases hp : run (engine M) before s with
  | none => rfl
  | some t =>
    simp only [Option.bind_some, run, step, FP.Structural.evalExpr, evalComputeOp]
    erw [eval_cast_congr (engine M) t (some .fp32) dtype .real _ _
      (eval_add_comm M D t (some .fp32) dtype hc a b (hd t hp))]
    rfl

/-- Explicit fp64 computation remains around the source's output conversion. -/
theorem contextual_cast_add_fp64 {α : Type} [Inhabited α]
    (M : Algebra α) (D : Domain α)
    (s : State α) (before after : List ComputeStmt) (out : RegName)
    (dtype : FloatDType) (a b : Op dtype.toTileDType shape)
    (hc : ScalarAdditionCommutes M D (some .fp64) dtype)
    (hd : (Site.add before (some .fp64) dtype shape a b).Holds M D s) :
    Contextual M s before after
      (.assign .real shape out
        (.compute (.alg .fp64 (.castFloat dtype .real (.add dtype.numericDType (sameShape shape) a b)))))
      (.assign .real shape out
        (.compute (.alg .fp64 (.castFloat dtype .real (.add dtype.numericDType (sameShape shape) b a))))) := by
  unfold Contextual
  rw [run_append, run_append]
  cases hp : run (engine M) before s with
  | none => rfl
  | some t =>
    simp only [Option.bind_some, run, step, FP.Structural.evalExpr, evalComputeOp]
    erw [eval_cast_congr (engine M) t (some .fp64) dtype .real _ _
      (eval_add_comm M D t (some .fp64) dtype hc a b (hd t hp))]
    rfl

namespace ProofTactic
open Lean Meta Elab Tactic
open VeriTile.Meta.EquivDecompose
private def differences (goal : MVarId) (contextual : Name) (model state lhs rhs : Expr)
    (backwards : Bool := Bool.false) : MetaM (List MVarId) := goal.withContext do
  let some left ← listItems? lhs | throwError "the source statement list is symbolic: {lhs}; type: {← inferType lhs}"
  let some right ← listItems? rhs | throwError "the target statement list is symbolic"
  unless left.size == right.size do
    throwError "contextual decomposition requires aligned statement lists; insertion/deletion needs an execution lemma"
  let statement := mkConst ``ComputeStmt
  let mut current := left
  let mut proof : Option Expr := none
  let mut remaining := []
  for offset in [:left.size] do
    let i := if backwards then left.size - 1 - offset else offset
    if ← same current[i]! right[i]! then continue
    let before ← mkListLit statement (current.extract 0 i).toList
    let after ← mkListLit statement (current.extract (i + 1) current.size).toList
    let target ← mkAppM contextual #[model, state, before, after, current[i]!, right[i]!]
    let hole ← mkFreshExprSyntheticOpaqueMVar target
      (tag := (← goal.getTag).appendAfter s!"diff_{remaining.length + 1}")
    remaining := remaining ++ [hole.mvarId!]
    proof ← match proof with
      | none => pure (some hole)
      | some previous => pure (some (← mkAppM ``Eq.trans #[previous, hole]))
    current := current.set! i right[i]!
  match proof with
  | none => goal.refl
  | some complete => goal.assign complete
  return remaining

private def contextPlan (lhs rhs : Expr) : MetaM (Option (Name × Nat)) := do
  let left := lhs.getAppArgs
  let right := rhs.getAppArgs
  if lhs.isAppOfArity ``ComputeStmt.forRangeDyn 5 &&
      rhs.isAppOfArity ``ComputeStmt.forRangeDyn 5 then
    for i in [:4] do
      unless ← same left[i]! right[i]! do return none
    return some (``contextual_range, 4)
  if lhs.isAppOfArity ``ComputeStmt.ifThen 2 &&
      rhs.isAppOfArity ``ComputeStmt.ifThen 2 then
    unless ← same left[0]! right[0]! do return none
    return some (``contextual_if_then, 1)
  if lhs.isAppOfArity ``ComputeStmt.ifThenElse 3 &&
      rhs.isAppOfArity ``ComputeStmt.ifThenElse 3 then
    unless ← same left[0]! right[0]! do return none
    if ← same left[2]! right[2]! then
      return some (``contextual_then, 1)
  return none

/-- Descend through ranges and matching conditional arms only when the declared
site supplies that execution context's guard. Numerical obligations remain for
fp_prove; this traversal does not manufacture conditions or rewrite a test. -/
private partial def contexts (goal : MVarId) (backwards : Bool := Bool.false) : MetaM (List MVarId) := goal.withContext do
  let target ← instantiateMVars (← goal.getType)
  unless target.isAppOfArity ``FP.ProfiledRewrite.Contextual 8 do return [goal]
  let args := target.getAppArgs
  let lhs ← whnf args[6]!
  let rhs ← whnf args[7]!
  let left := lhs.getAppArgs
  let right := rhs.getAppArgs
  let some (ruleName, bodyIndex) ← contextPlan lhs rhs | return [goal]
  let saved ← saveState
  try
    let children ← goal.apply (← mkConstWithFreshMVarLevels ruleName)
    -- Resolve the actual site first: it supplies the inner domain after the
    -- original prefix and only in reachable iterations/selected arms.
    for child in children do
      unless ← child.isAssigned do
        try child.assumption catch _ => pure ()
    let mut remaining := []
    for child in children do
      if ← child.isAssigned then continue
      let type ← whnf (← child.getType)
      unless type.isForall do throwError "execution-context domain was not supplied"
      let (vars, body) ← child.introNP 2
      let nested ← differences body ``FP.ProfiledRewrite.Contextual args[2]!
        (mkFVar vars[0]!) left[bodyIndex]! right[bodyIndex]! backwards
      for difference in nested do remaining := remaining ++ (← contexts difference backwards)
    return remaining
  catch _ =>
    saved.restore
    return [goal]


elab "native_decompose " modelStx:term : tactic => do
  let goal ← getMainGoal
  let remaining ← goal.withContext do
    let model ← elabTerm modelStx none
    let target ← instantiateMVars (← goal.getType)
    let target := target.consumeMData
    unless target.isAppOfArity ``Eq 3 do
      throwError "native decomposition expects equality, got {target}"
    let args := target.getAppArgs
    let lhs := args[1]!.consumeMData
    let rhs := args[2]!.consumeMData
    let left := lhs.getAppArgs
    let right := rhs.getAppArgs
    unless lhs.getAppFn.isConstOf ``FP.Structural.run && left.size ≥ 2 do
      throwError "native decomposition expects a structural run, got {lhs} ({left.size} arguments)"
    unless rhs.getAppFn.isConstOf ``FP.Structural.run && right.size ≥ 2 do
      throwError "native decomposition expects a target run, got {rhs} ({right.size} arguments)"
    let nested ← differences goal ``FP.ProfiledRewrite.Contextual model
      left[left.size - 1]! left[left.size - 2]! right[right.size - 2]!
    let mut result := []
    for child in nested do result := result ++ (← contexts child)
    return result
  replaceMainGoal remaining
end ProofTactic

section
variable (input : TypedSource.InputStorage) (weight : TypedSource.WeightStorage)
    (cast : Bool) (output : FloatDType)
    (x w out : RegionName) (N K : Nat) (indices : Region .int)
    (sx sk sw0 swn swk sy syn slice BN BK splits : Nat) (even add : Bool)

def original : ComputeKernel :=
  TypedSource.original input weight cast output x w out N K indices
    sx sk sw0 swn swk sy syn slice BN BK splits even add

def outputFirst : ComputeKernel :=
  TypedSource.outputFirst input weight cast output x w out N K indices
    sx sk sw0 swn swk sy syn slice BN BK splits even add

/-- All 72 storage/cast/output specializations retain their actual addition guard. -/
theorem requirements_present :
    (requirementsFor (original input weight cast output x w out N K indices
      sx sk sw0 swn swk sy syn slice BN BK splits even add)).length = 1 := by
  cases input <;> cases weight <;> cases cast <;> cases output <;> rfl

/-- Output fp32 promotes the source addition to the admitted fp32 profile. -/
theorem fp32_profiles :
    (requirementsFor (original input weight cast .fp32 x w out N K indices
      sx sk sw0 swn swk sy syn slice BN BK splits even add)).map Site.profile =
      [(some .fp32, .real)] := by
  cases input <;> cases weight <;> cases cast <;> rfl

theorem fp32_native_of_rules {α : Type} [Inhabited α]
    (R : FP.ScalarArithmetic.Rules) (M : Algebra α) (D : Domain α)
    (hM : FP.Guarded.Models R.assumptions M D) (s : State α) :
    NativeAdditionCommutes M D (original input weight cast .fp32 x w out N K indices
      sx sk sw0 swn swk sy syn slice BN BK splits even add) := by
  intro site hs
  have member := List.mem_map_of_mem (f := Site.profile) hs
  rw [fp32_profiles] at member
  have hp : site.profile = (some .fp32, .real) := by simpa only [List.mem_singleton] using member
  rw [hp]
  exact (fp32_scalar_of_rules R M D hM s).2

/-- Conditional, whole-state equivalence on the original source's reached
finite operands. No hypothesis mentions the candidate or a kernel result. -/
theorem output_first_execution {α : Type} [Inhabited α]
    (M : Algebra α) (D : Domain α)
    (hc : NativeAdditionCommutes M D (original input weight cast output x w out N K indices
      sx sk sw0 swn swk sy syn slice BN BK splits even add)) (s : State α)
    (hd : ∀ site ∈ requirementsFor (original input weight cast output x w out N K indices
      sx sk sw0 swn swk sy syn slice BN BK splits even add), site.Holds M D s) :
    FP.Structural.exec (engine M) (original input weight cast output x w out N K indices
      sx sk sw0 swn swk sy syn slice BN BK splits even add) s =
    FP.Structural.exec (engine M) (outputFirst input weight cast output x w out N K indices
      sx sk sw0 swn swk sy syn slice BN BK splits even add) s := by
  cases input <;> cases weight <;> cases cast <;> cases output
  all_goals
    change run (engine M) _ s = run (engine M) _ s
    have site_guard := hd _ (List.mem_cons_self)
    have native_law := hc _ (List.mem_cons_self)
    native_decompose M
  all_goals first
    | apply contextual_cast_scoped_add_fp32 M D (output := .real)
    | apply contextual_cast_scoped_add_fp32 M D (output := .fp32)
    | apply contextual_cast_scoped_add_fp32 M D (output := .fp16)
    | apply contextual_cast_scoped_add_fp32 M D (output := .bf16)
    | apply contextual_cast_scoped_add_fp32 M D (output := .f8e4)
    | apply contextual_cast_scoped_add_fp32 M D (output := .f8e5)
    | apply contextual_cast_add M D (output := .real)
    | apply contextual_cast_add M D (output := .fp32)
    | apply contextual_cast_add M D (output := .fp16)
    | apply contextual_cast_add M D (output := .bf16)
    | apply contextual_cast_add M D (output := .f8e4)
    | apply contextual_cast_add M D (output := .f8e5)
    | apply contextual_cast_add_fp32 M D
    | apply contextual_cast_add_fp64 M D
  all_goals assumption
end

#guard_msgs (drop info) in
#specNonCircular ScalarAdditionCommutes avoiding [TypedSource.original, TypedSource.outputFirst]
#guard_msgs (drop info) in
#specNonCircular NativeAdditionCommutes avoiding [TypedSource.outputFirst]
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.BgmvExpandSlice.NativeFP
