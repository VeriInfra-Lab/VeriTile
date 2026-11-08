import VeriTile.Triton.Float.ProfiledRewrite

/-! Lift the admitted division/reciprocal rule at an explicit FP32 source site.
The numerator is an actual register read, whose lookup is precision independent.
The denominator literal is checked at its FP32 value by the existing site guard. -/
namespace VeriTile.Triton.FP.ProfiledRewrite
open Structural Guarded

@[spec_rule] theorem contextual_div_fp32_reference_mul_rcp {α : Type} [Inhabited α]
    (R : ScalarArithmetic.Rules) (M : Algebra α) (D : Domain α)
    (hM : Models R.assumptions M D) (s : State α) (before after : List ComputeStmt)
    (out name : RegName) (n : Nat) (rest : TileShape) (b : ℝ)
    (hd : (Site.div before n rest (.ref .real (n :: rest) name) b).Holds M D s) :
    Contextual M s before after
      (.assign .real (n :: rest) out (.compute (.alg .fp32
        (.div .real .scalarR (.ref .real (n :: rest) name) (.const b)))))
      (.assign .real (n :: rest) out (.compute (.alg .fp32
        (.mul .real .scalarR (.ref .real (n :: rest) name)
          (.div .real .nil (.const 1) (.const b)))))) := by
  unfold Contextual
  rw [run_append, run_append]
  cases hp : run (engine M) before s with
  | none => rfl
  | some t =>
    simp only [Option.bind_some, run, step, Structural.evalExpr, evalComputeOp]
    have ediv : Structural.evalOp (engine M) (some .fp32)
        (.div .real .scalarR (.ref .real (n :: rest) name) (.const b)) t =
        (t.regs .real (n :: rest) name).bind (fun va => some (fun i =>
          (engine M).binary (some .fp32) .real .div (va i)
            ((engine M).literal (some .fp32) .real b))) := rfl
    have emul : Structural.evalOp (engine M) (some .fp32)
        (.mul .real .scalarR (.ref .real (n :: rest) name)
          (.div .real .nil (.const 1) (.const b))) t =
        (t.regs .real (n :: rest) name).bind (fun va => some (fun i =>
          (engine M).binary (some .fp32) .real .mul (va i)
            ((engine M).binary (some .fp32) .real .div
              ((engine M).literal (some .fp32) .real 1)
              ((engine M).literal (some .fp32) .real b)))) := rfl
    erw [ediv, emul]
    cases ha : t.regs .real (n :: rest) name with
    | none => rfl
    | some va =>
      obtain ⟨hva, hb, hn⟩ := hd t hp va ha
      have hv : (fun i => (engine M).binary (some .fp32) .real .div (va i)
            ((engine M).literal (some .fp32) .real b)) =
          (fun i => (engine M).binary (some .fp32) .real .mul (va i)
            ((engine M).binary (some .fp32) .real .div
              ((engine M).literal (some .fp32) .real 1)
              ((engine M).literal (some .fp32) .real b))) := by
        funext i
        exact ScalarArithmetic.div_mul_rcp R M D hM t _ _ (hva i) hb hn
      simp only [Option.bind_some]
      rw [hv]

end VeriTile.Triton.FP.ProfiledRewrite
