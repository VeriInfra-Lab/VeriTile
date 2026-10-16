import bench.tritonbench_g.cache_transform_triton.Proofs.IndexRewrite

namespace VeriTile.Bench.TritonBenchG.PowScalarTensor.FPProof
open VeriTile.Triton FP.Structural FP.ProfiledRewrite

/-- Lift an unconditional body equality through the actual dynamic range,
preserving the full state, empty ranges and execution failure. -/
theorem contextual_range {α : Type} [Inhabited α] (M : Algebra α)
    (s : State α) (before after : List ComputeStmt) (index : RegName)
    (start stop stride : Op .nat []) (lhs rhs : List ComputeStmt)
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
            (b PUnit.unit) (d PUnit.unit) lhs rhs t (fun c u _ _ _ =>
              hbody (u.setReg index .nat [] (fun _ => c)))]

@[spec_rule] theorem offset_loop {α : Type} [Inhabited α] (M : Algebra α)
    (s : State α) (before after : List ComputeStmt) (index : RegName)
    (start stop stride : Op .nat []) (init₁ init₂ : ComputeStmt)
    (rest : List ComputeStmt) (a b : Op .nat []) :
    Contextual M s before after
      (.forRangeDyn index start stop stride (init₁ :: init₂ ::
        .assign .nat [] "offset0" (.alg (.mul .nat .nil a b)) :: rest))
      (.forRangeDyn index start stop stride (init₁ :: init₂ ::
        .assign .nat [] "offset0" (.alg (.mul .nat .nil b a)) :: rest)) := by
  apply contextual_range
  intro t
  change Contextual M t [init₁, init₂] rest
    (.assign .nat [] "offset0" (.alg (.mul .nat .nil a b)))
    (.assign .nat [] "offset0" (.alg (.mul .nat .nil b a)))
  fp_prove

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.PowScalarTensor.FPProof
