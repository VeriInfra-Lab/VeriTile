import bench.tritonbench_g.matmul_persistent_triton.Proofs.SourceChecks
import VeriTile.Triton.Float.ExpressionRewrite
import VeriTile.Triton.Float.RewriteTactics
import VeriTile.Triton.Float.ProfiledSoundness

namespace VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.FPProof
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open scoped VeriTile.Spec
set_option maxHeartbeats 12000000
set_option maxRecDepth 24000

@[spec_rule] theorem signed_index_product_commutes {α : Type} [Inhabited α]
    (M : Algebra α) (s : State α) (before : List ComputeStmt) (a b : Op .int []) :
    OpEquivalent M s before (.mul .int .nil a b) (.mul .int .nil b a) := by
  intro t _
  change (evalOp (engine M) none a t >>= fun x =>
    evalOp (engine M) none b t >>= fun y => some (bop Int.mul .nil x y)) =
    (evalOp (engine M) none b t >>= fun y =>
    evalOp (engine M) none a t >>= fun x => some (bop Int.mul .nil y x))
  cases ha : evalOp (engine M) none a t <;> cases hb : evalOp (engine M) none b t
  all_goals first
    | rfl
    | apply congrArg some
      funext i
      exact Int.mul_comm _ _

def assumptions : Spec.Assumptions FP.GuardedFragment := []

def program (p : Parameters) (storage : TypedSource.Storage) (v : Bool) : Program where
  kernel := p.source storage v
  domain := []

theorem contextual_if_exact {α : Type} [Inhabited α] (M : Algebra α)
    (s : State α) (before after : List ComputeStmt)
    (condition : Op .bool []) (lhs rhs : List ComputeStmt)
    (hbody : ∀ t, run (engine M) lhs t = run (engine M) rhs t) :
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
      | «true» => simp only [↓reduceIte]; rw [hbody t]

theorem contextual_range_exact {α : Type} [Inhabited α] (M : Algebra α)
    (s : State α) (before after : List ComputeStmt)
    (index : RegName) (start stop stride : Op .nat [])
    (lhs rhs : List ComputeStmt)
    (hbody : ∀ t, run (engine M) lhs t = run (engine M) rhs t) :
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
            (b PUnit.unit) (d PUnit.unit) lhs rhs t
            (fun cur u _ _ _ => hbody (u.setReg index .nat [] (fun _ => cur)))]

local instance : Inhabited ComputeStmt := ⟨.alg (.assign .nat [] "" (.constNat 0))⟩

def loop (p : Parameters) (storage : TypedSource.Storage) (v : Bool) :=
  SourceChecks.loopBody (p.source storage v).surfaceBody[16]!

def completion (p : Parameters) (storage : TypedSource.Storage) (v : Bool) :=
  SourceChecks.thenBody (loop p storage v)[8]!

theorem completion_equivalent (p : Parameters) (storage : TypedSource.Storage)
    {α : Type} [Inhabited α] (M : Algebra α) (s : State α) :
    run (engine M) (completion p storage Bool.false) s =
      run (engine M) (completion p storage Bool.true) s := by
  cases storage
  case fp16 =>
    change Contextual M s [] (completion p .fp16 Bool.false |>.drop 1)
      (completion p .fp16 Bool.false)[0]! (completion p .fp16 Bool.true)[0]!
    fp_prove
  case f8e4 =>
    change Contextual M s [] (completion p .f8e4 Bool.false |>.drop 1)
      (completion p .f8e4 Bool.false)[0]! (completion p .f8e4 Bool.true)[0]!
    fp_prove

theorem loop_equivalent (p : Parameters) (storage : TypedSource.Storage)
    {α : Type} [Inhabited α] (M : Algebra α) (s : State α) :
    run (engine M) (loop p storage Bool.false) s =
      run (engine M) (loop p storage Bool.true) s := by
  cases storage
  case fp16 =>
    change Contextual M s (loop p .fp16 Bool.false |>.take 8) []
      (loop p .fp16 Bool.false)[8]! (loop p .fp16 Bool.true)[8]!
    apply contextual_if_exact
    intro t
    exact completion_equivalent p .fp16 M t
  case f8e4 =>
    change Contextual M s (loop p .f8e4 Bool.false |>.take 8) []
      (loop p .f8e4 Bool.false)[8]! (loop p .f8e4 Bool.true)[8]!
    apply contextual_if_exact
    intro t
    exact completion_equivalent p .f8e4 M t

/-- Exact complete typed-state equivalence under every floating backend.
The fused dot, its accumulator, casts, masks, persistent schedule, failures
and all resets are unchanged. No floating numerical assumption is used. -/
theorem source_equivalence (p : Parameters) (storage : TypedSource.Storage) :
    program p storage Bool.false ≡[assumptions] program p storage Bool.true := by
  apply Spec.FloatingPoint.ofNumerical rfl
  · cases storage <;> rfl
  intro α _ M D hM s hd
  cases storage
  case fp16 =>
    change Contextual M s ((p.source .fp16 Bool.false).surfaceBody.take 16) []
      (p.source .fp16 Bool.false).surfaceBody[16]!
      (p.source .fp16 Bool.true).surfaceBody[16]!
    apply contextual_range_exact
    intro t
    exact loop_equivalent p .fp16 M t
  case f8e4 =>
    change Contextual M s ((p.source .f8e4 Bool.false).surfaceBody.take 16) []
      (p.source .f8e4 Bool.false).surfaceBody[16]!
      (p.source .f8e4 Bool.true).surfaceBody[16]!
    apply contextual_range_exact
    intro t
    exact loop_equivalent p .f8e4 M t

#print_fp_assumptions source_equivalence
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.FPProof
