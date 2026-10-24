import bench.tritonbench_g.uniform_sampling.Proofs.TypedRandom
import bench.tritonbench_g.uniform_sampling.Proofs.TupleExecution

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

def affineStmt (B : Nat) (name : RegName) (lower upper : ℝ) : ComputeStmt :=
  .assign .real [B] name (.compute (.alg .fp32 (.add .real .scalarR
    (.mul .real .scalarR (.uniform32 (.ref .nat [B] name))
      (.sub .real .nil (.const upper) (.const lower))) (.const lower))))

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

theorem affine_step {α : Type} [Inhabited α] (M : Algebra α)
    {B : Nat} (words : TileIndex [B] → Random.Word) (name : RegName)
    (lower upper : ℝ) (s : State α)
    (hw : s.regs .nat [B] name = some (fun i => (words i).toNat)) :
    step M (affineStmt B name lower upper) s =
      some (s.setReg name .real [B] (values M words lower upper)) := by
  simp only [affineStmt, step, evalExpr, evalComputeOp]
  exact congrArg (fun value : Option (Values α .real [B]) =>
    value.bind (fun v => some (s.setReg name .real [B] v)))
      (affine_eval M words name lower upper s hw)

def body (B : Nat) (lower upper : ℝ) : List ComputeStmt :=
  [affineStmt B "r0" lower upper, affineStmt B "r1" lower upper,
    affineStmt B "r2" lower upper, affineStmt B "r3" lower upper]

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

theorem affines_exec {α : Type} [Inhabited α] (M : Algebra α)
    {B : Nat} (words : Random.Direct.Field → TileIndex [B] → Random.Word)
    (lower upper : ℝ) (s : State α)
    (hw : ∀ f, s.regs .nat [B] (fieldName f) = some (fun i => (words f i).toNat)) :
    run M (body B lower upper) s = some (result M words lower upper s) := by
  let s0 := s.setReg "r0" .real [B] (values M (words .c0) lower upper)
  let s1 := s0.setReg "r1" .real [B] (values M (words .c1) lower upper)
  let s2 := s1.setReg "r2" .real [B] (values M (words .c2) lower upper)
  exact four_steps M
    (affine_step M (words .c0) "r0" lower upper s (hw .c0))
    (affine_step M (words .c1) "r1" lower upper s0 (by simpa [s0, fieldName] using hw .c1))
    (affine_step M (words .c2) "r2" lower upper s1 (by simpa [s1, s0, fieldName] using hw .c2))
    (affine_step M (words .c3) "r3" lower upper s2 (by simpa [s2, s1, s0, fieldName] using hw .c3))

theorem result_register {α : Type} (M : Algebra α) {B : Nat}
    (words : Random.Direct.Field → TileIndex [B] → Random.Word)
    (lower upper : ℝ) (s : State α) (f : Random.Direct.Field) :
    (result M words lower upper s).regs .real [B] (fieldName f) =
      some (values M (words f) lower upper) := by
  cases f <;> simp [result, fieldName]

theorem source_slice (storage : Storage) (swapped : Bool) (y : RegionName)
    (N : Nat) (seed offset : Int) (lower upper : ℝ) (B : Nat)
    (hB : SourceDomain.LegalBlock B) :
    ((source storage swapped y N seed offset lower upper B hB).surfaceBody.drop 11).take 4 =
      body B lower upper := by
  cases storage <;> cases swapped <;> rfl

#axiomsClean affine_eval
#axiomsClean affine_step
#axiomsClean four_steps
#axiomsClean affines_exec
#axiomsClean result_register
#axiomsClean source_slice

end
end VeriTile.Bench.TritonBenchG.UniformSampling.TypedAffine
