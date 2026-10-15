import bench.tritonbench_g.context_attn_fwd.Proofs.FPModel
import VeriTile.Triton.Float.TypedWhereRewrite
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.TypedFP32Mul
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open scoped VeriTile.Spec
namespace Base
export VeriTile.Bench.TritonBenchG.ContextAttnFwd.FPModel (model)
end Base
noncomputable section
/-- The real channel retains every old scalar law; the fp32 channel is left projection. -/
def model : Algebra ℝ := { Base.model with
  binary := fun p d op a b => if d = .fp32 then a else Base.model.binary p d op a b }
def domain : FP.Guarded.Domain ℝ
  | .finite, _ => True
  | .nonzero, x => x ≠ 0
  | _, x => 0 < x

set_option maxHeartbeats 3200000
set_option maxRecDepth 3000
set_option linter.unusedSimpArgs false

private theorem finite_register (s : State ℝ) (gs : List FP.OperandGuard)
    (hd : FP.Guarded.ScalarDomain domain gs s) (name : RegName)
    (hm : ⟨name, .finite⟩ ∈ gs) :
    ∃ a : ℝ, s.regs .real [] name = some (fun _ => a) := by
  obtain ⟨v, hv, _⟩ := hd _ hm
  refine ⟨v PUnit.unit, hv.trans (congrArg some ?_)⟩
  funext i
  cases i
  rfl

private theorem nonzero_register (s : State ℝ) (gs : List FP.OperandGuard)
    (hd : FP.Guarded.ScalarDomain domain gs s) (name : RegName)
    (hm : ⟨name, .nonzero⟩ ∈ gs) (a : ℝ)
    (ha : s.regs .real [] name = some (fun _ => a)) : a ≠ 0 := by
  obtain ⟨v, hv, hn⟩ := hd _ hm
  have eqv := Option.some.inj (hv.symm.trans ha)
  simpa only [domain, eqv] using hn

/-- The execution witness satisfies all scalar candidates, with their guards.
Real arithmetic here establishes consistency of the abstract assumptions; the
kernel equivalence still uses the selected FP atom, not real algebra. -/
private theorem arithmetic_atom (atom : FP.ScalarArithmetic.Atom) :
    FP.FragmentSound model domain (FP.ScalarArithmetic.lhs atom)
      (FP.ScalarArithmetic.rhs atom) := by
  open FP.ScalarArithmetic in
    refine ⟨rfl, fun s hd => ?_⟩
    change FP.Guarded.ScalarDomain domain (guards atom) s at hd
    cases atom
    all_goals
      obtain ⟨a, ha⟩ := finite_register s _ hd "a" (by decide)
      try
        have h : ∃ b : ℝ, s.regs .real [] "b" = some (fun _ => b) := by
          apply finite_register s _ hd
          decide
        obtain ⟨b, hb⟩ := h
      try
        have h : ∃ c : ℝ, s.regs .real [] "c" = some (fun _ => c) := by
          apply finite_register s _ hd
          decide
        obtain ⟨c, hc⟩ := h
      try have hn := nonzero_register s _ hd "b" (by decide) b hb
      try have hn := nonzero_register s _ hd "a" (by decide) a ha
      simp only [lhs, rhs, lhsCode, rhsCode, fragment, FP.ScalarArithmetic.ref,
        FP.ScalarArithmetic.plus, FP.ScalarArithmetic.minus, FP.ScalarArithmetic.times,
        FP.ScalarArithmetic.divide, FP.Reciprocal.lhs, FP.Reciprocal.rhs, FP.Reciprocal.div,
        FP.Reciprocal.inv, FP.Reciprocal.mul, FP.Reciprocal.ref,
        run, step, FP.Structural.evalExpr, evalComputeOp, evalOp_unfold,
        ComputeDType.eraseDType, numeric, ha, State.setReg_same,
        FP.Guarded.fragmentAlgebra, FP.Guarded.scalarOutput, toFloat]
      simp_all [State.setReg, model, Base.model, bop, Function.comp_def, _root_.add_comm, _root_.add_assoc,
        _root_.mul_comm, _root_.mul_assoc, mul_add, div_eq_mul_inv]
      all_goals
        funext i
        dsimp [FP.Structural.bop]
        simp_all [_root_.add_comm, _root_.add_assoc, _root_.mul_comm, _root_.mul_assoc,
          mul_add, div_eq_mul_inv]
        all_goals ring

theorem model_satisfies_rules (R : FP.ScalarArithmetic.Rules) :
    FP.Guarded.Models R.assumptions model domain := by
  apply FP.models_of_atoms
  intro entry member
  obtain ⟨atom, _, he⟩ := List.mem_filterMap.mp member
  unfold FP.ScalarArithmetic.Atom.entry? at he
  cases hr : atom.report? with
  | none => simp [hr] at he
  | some row =>
      simp only [hr, Option.map_some, Option.some.injEq] at he
      subst entry
      exact .cons (arithmetic_atom atom) .nil



def empty : State ℝ where
  mem := fun _ _ => .mk .real 0
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => defaultValue d

def inputs : State ℝ :=
  (empty.setReg "a" .fp32 [] (fun _ => 1)).setReg "b" .fp32 [] (fun _ => 2)

theorem typed_domain_inhabited :
    FP.Guarded.TypedScalarDomain domain .fp32 FP.TypedScalarArithmetic.guards inputs := by
  intro g hg
  simp only [FP.TypedScalarArithmetic.guards, List.mem_cons, List.not_mem_nil, or_false] at hg
  rcases hg with rfl | rfl <;>
    simp [FP.Guarded.TypedScalarDomain, inputs, State.setReg, domain, toFloat]

/-- A real-register finite guard cannot validate missing typed fp32 inputs. -/
example : ¬ FP.Guarded.TypedScalarDomain domain .fp32 FP.TypedScalarArithmetic.guards
    ((empty.setReg "a" .real [] (fun _ => 1)).setReg "b" .real [] (fun _ => 2)) := by
  intro h
  obtain ⟨v, hv, _⟩ := h ⟨"a", .finite⟩ (by simp [FP.TypedScalarArithmetic.guards])
  simp [empty, State.setReg] at hv

/-- Existing scalar rules alone permit a noncommutative typed fp32 channel. -/
theorem old_rules_insufficient (R : FP.ScalarArithmetic.Rules) :
    FP.Guarded.Models R.assumptions model domain ∧
    model.binary (some .fp32) .fp32 .mul 1 2 ≠
      model.binary (some .fp32) .fp32 .mul 2 1 := by
  refine ⟨model_satisfies_rules R, ?_⟩
  norm_num [model]

/-- The exact typed atom is a genuine extra assumption, observed at its typed output. -/
theorem old_derivation_unavailable (R : FP.ScalarArithmetic.Rules) :
    ¬ Spec.Derivation R.assumptions [FP.TypedScalarArithmetic.lhs] [FP.TypedScalarArithmetic.rhs] := by
  intro h
  have obs := model_satisfies_rules R _ _ h inputs typed_domain_inhabited typed_domain_inhabited
  simp [FP.TypedScalarArithmetic.lhs, FP.TypedScalarArithmetic.rhs,
    FP.Guarded.fragmentAlgebra, FP.Guarded.scalarOutput, Algebra.withDefaultPrecision,
    run, step, evalExpr, evalComputeOp, evalOp_unfold, numeric,
    inputs, State.setReg, model, Base.model, toFloat, FP.Structural.bop] at obs
  have bad := congrFun obs PUnit.unit
  change (1 : ℝ) = 2 at bad
  norm_num at bad

/-- The compatibility defaults preserve the original input predicate and engine. -/
example (D : FP.Guarded.Domain ℝ) (gs : List FP.OperandGuard) (s : State ℝ) :
    FP.Guarded.TypedScalarDomain D .real gs s = FP.Guarded.ScalarDomain D gs s := rfl
example (gs : List FP.OperandGuard) (code : List ComputeStmt) (M : Algebra ℝ) :
    FP.Guarded.fragmentAlgebra ⟨gs, code, .real, none⟩ M = M := rfl
example (s : State ℝ) : FP.Guarded.scalarOutput .real s = s.regs .real [] "out" := by simp

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Tests.TypedFP32Mul
