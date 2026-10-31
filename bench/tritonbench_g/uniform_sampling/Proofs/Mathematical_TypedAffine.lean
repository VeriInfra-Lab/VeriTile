import bench.tritonbench_g.uniform_sampling.Proofs.Mathematical_TypedRandom
import bench.tritonbench_g.uniform_sampling.Proofs.Mathematical_TupleExecution

/-! The actual fp32 uniform payload and all three fp32 affine operations in
the typed evaluator. No algebra law or identification with native FMA is
required for successful evaluation. All four fields remain explicit.
-/
namespace VeriTile.Bench.TritonBenchG.UniformSampling.TypedAffine
open VeriTile.Triton TypedSource FP.Structural
open TupleExecution (fieldName)
noncomputable section
set_option maxHeartbeats 1500000
set_option maxRecDepth 10000

def values {α : Type} (M : Algebra α) {B : Nat} (words : TileIndex [B] → Random.Word)
    (lower upper : ℝ) : Values α .real [B] :=
  fun i => M.binary (some .fp32) .real .add
    (M.binary (some .fp32) .real .mul (M.fp32Bits ⟨(Random.uniform (words i)).bits⟩)
      (M.binary (some .fp32) .real .sub (M.literal (some .fp32) .real upper)
        (M.literal (some .fp32) .real lower))) (M.literal (some .fp32) .real lower)

private theorem eval_add {α : Type} [Inhabited α] (M : Algebra α) (p : Option ComputeDType)
    (bc : Broadcast a b c) (x : Op .real a) (y : Op .real b) (s : State α) :
    FP.Structural.evalOp M p (.add .real bc x y) s =
      ((FP.Structural.evalOp M p x s).bind fun vx =>
        (FP.Structural.evalOp M p y s).bind fun vy =>
          some (bop (M.binary p .real .add) bc vx vy)) := rfl

private theorem eval_mul {α : Type} [Inhabited α] (M : Algebra α) (p : Option ComputeDType)
    (bc : Broadcast a b c) (x : Op .real a) (y : Op .real b) (s : State α) :
    FP.Structural.evalOp M p (.mul .real bc x y) s =
      ((FP.Structural.evalOp M p x s).bind fun vx =>
        (FP.Structural.evalOp M p y s).bind fun vy =>
          some (bop (M.binary p .real .mul) bc vx vy)) := rfl

theorem affine_eval {α : Type} [Inhabited α] (M : Algebra α)
    {B : Nat} (words : TileIndex [B] → Random.Word) (name : RegName)
    (lower upper : ℝ) (s : State α)
    (hw : s.regs .nat [B] name = some (fun i => (words i).toNat)) :
    FP.Structural.evalOp M (some .fp32) (.add .real .scalarR
      (.mul .real .scalarR (.uniform32 (.ref .nat [B] name))
        (.sub .real .nil (.const upper) (.const lower))) (.const lower)) s =
      some (values M words lower upper) := by
  have hu := TypedRandom.eval_uniform words M (some .fp32) s (.ref .nat [B] name)
    (by simpa only [evalOp_unfold] using hw)
  rw [eval_add, eval_mul, hu]
  simp only [evalOp_unfold, bind, Option.bind, pure, numeric]
  apply congrArg some
  funext i
  simp [values, bop]

def result {α : Type} (M : Algebra α) {B : Nat}
    (words : Random.Direct.Field → TileIndex [B] → Random.Word)
    (lower upper : ℝ) (s : State α) : State α :=
  let s := s.setReg "r0" .real [B] (values M (words .c0) lower upper)
  let s := s.setReg "r1" .real [B] (values M (words .c1) lower upper)
  let s := s.setReg "r2" .real [B] (values M (words .c2) lower upper)
  s.setReg "r3" .real [B] (values M (words .c3) lower upper)

theorem four_steps {α : Type} [Inhabited α] (M : Algebra α)
    {a b c d : ComputeStmt} {s0 s1 s2 s3 s4 : State α}
    (ha : step M a s0 = some s1) (hb : step M b s1 = some s2)
    (hc : step M c s2 = some s3) (hd : step M d s3 = some s4) :
    run M [a, b, c, d] s0 = some s4 := by
  simp only [run, ha, hb, hc, hd, bind, Option.bind, pure]

theorem result_register {α : Type} (M : Algebra α) {B : Nat}
    (words : Random.Direct.Field → TileIndex [B] → Random.Word)
    (lower upper : ℝ) (s : State α) (f : Random.Direct.Field) :
    (result M words lower upper s).regs .real [B] (fieldName f) =
      some (values M (words f) lower upper) := by
  cases f <;> simp [result, fieldName]

#axiomsClean affine_eval
#axiomsClean four_steps
#axiomsClean result_register

end
end VeriTile.Bench.TritonBenchG.UniformSampling.TypedAffine
