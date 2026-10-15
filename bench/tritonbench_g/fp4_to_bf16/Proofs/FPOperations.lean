import bench.tritonbench_g.fp4_to_bf16.Proofs.FPInteger
namespace VeriTile.Bench.TritonBenchG.Fp4ToBf16.FPValue
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

theorem eval_nat_as_i32 {α : Type} [Inhabited α] (M : Algebra α)
    (p : Option ComputeDType) (x : Op .nat [n]) (s : State α) :
    FP.Structural.evalOp M p (Op.intToSigned 32 (.castNatToInt x)) s =
      (FP.Structural.evalOp M p x s).map (fun v i => signed32 (v i)) := by
  rw [FP.Structural.eval_intToSigned]
  simp only [FP.Structural.evalOp]
  cases hx : FP.Structural.evalOp M p x s <;>
    simp only [bind, pure, Option.bind_none, Option.bind_some, Option.map_none, Option.map_some,
      signed32_bitvec]
  rfl

/-- Exact signed32 shift expansion, with both source casts retained. -/
theorem eval_shift32 {α : Type} [Inhabited α] (M : Algebra α)
    (p : Option ComputeDType) (x : Op .nat [n]) (amount : Nat) (s : State α) :
    FP.Structural.evalOp M p
      (Op.intToSigned 32 (.mul .int .scalarR
        (Op.intToSigned 32 (.castNatToInt x))
        (.castNatToInt (.shiftLeft (IntegerExpr.sameShape []) (.full [] (.constNat 1)) (.constNat amount))))) s =
      (FP.Structural.evalOp M p x s).map
        (fun v i => signed32 (signed32 (v i) * (↑(1 <<< amount : Nat) : Int))) := by
  rw [FP.Structural.eval_intToSigned]
  rw [FP.Structural.evalOp]
  rw [eval_nat_as_i32]
  simp only [FP.Structural.evalOp]
  cases hx : FP.Structural.evalOp M p x s <;>
    simp only [bind, pure, Option.map_none, Option.map_some, Option.bind_none, Option.bind_some,
      signed32_bitvec, FP.Structural.bop, FP.Structural.numeric,
      Broadcast.leftIndex, Broadcast.rightIndex, IntegerExpr.same_left, IntegerExpr.same_right]
  rfl

theorem eval_or32 {α : Type} [Inhabited α] (M : Algebra α)
    (p : Option ComputeDType) (a b : Op .int [n]) (s : State α) :
    FP.Structural.evalOp M p
      (Op.unsignedToInt 32 (.bitOr (IntegerExpr.sameShape [n])
        (Op.intToUnsigned 32 a) (Op.intToUnsigned 32 b))) s =
      (do let va ← FP.Structural.evalOp M p a s
          let vb ← FP.Structural.evalOp M p b s
          pure (fun i => or32 (va i) (vb i))) := by
  rw [FP.Structural.eval_unsignedToInt]
  rw [FP.Structural.evalOp]
  rw [FP.Structural.eval_intToUnsigned, FP.Structural.eval_intToUnsigned]
  cases ha : FP.Structural.evalOp M p a s <;>
    cases hb : FP.Structural.evalOp M p b s <;>
    simp only [bind, pure, Option.bind_none, Option.bind_some, or32, unsigned32,
      FP.Structural.bop, IntegerExpr.same_left, IntegerExpr.same_right]
  rfl

/-- Named AST nodes keep evaluator certificates small; source-slice equality
is definitionally checked against the translated 31-statement kernel. -/
def shiftExpr (x : Op .nat [n]) (amount : Nat) : Op .int [n] :=
  Op.intToSigned 32 (.mul .int .scalarR (Op.intToSigned 32 (.castNatToInt x))
    (.castNatToInt (.shiftLeft (IntegerExpr.sameShape []) (.full [] (.constNat 1)) (.constNat amount))))
def orExpr (a b : Op .int [n]) : Op .int [n] :=
  Op.unsignedToInt 32 (.bitOr (IntegerExpr.sameShape [n]) (Op.intToUnsigned 32 a) (Op.intToUnsigned 32 b))
def intConst (z : Int) : Op .int [] := Op.intToSigned 32 (.constInt z)

@[simp] theorem eval_shiftExpr {α : Type} [Inhabited α] (M : Algebra α)
    (p : Option ComputeDType) (x : Op .nat [n]) (amount : Nat) (s : State α) :
    FP.Structural.evalOp M p (shiftExpr x amount) s =
      (FP.Structural.evalOp M p x s).map
        (fun v i => signed32 (signed32 (v i) * (↑(1 <<< amount : Nat) : Int))) :=
  eval_shift32 M p x amount s
@[simp] theorem eval_orExpr {α : Type} [Inhabited α] (M : Algebra α)
    (p : Option ComputeDType) (a b : Op .int [n]) (s : State α) :
    FP.Structural.evalOp M p (orExpr a b) s =
      (do let va ← FP.Structural.evalOp M p a s
          let vb ← FP.Structural.evalOp M p b s
          pure (fun i => or32 (va i) (vb i))) := eval_or32 M p a b s
@[simp] theorem eval_intConst {α : Type} [Inhabited α] (M : Algebra α)
    (p : Option ComputeDType) (z : Int) (s : State α) :
    FP.Structural.evalOp M p (intConst z) s = some (fun _ => signed32 z) := by
  rw [intConst, FP.Structural.eval_intToSigned, FP.Structural.evalOp]
  simp only [bind, pure, Option.bind_some, signed32_bitvec]

def exponentCode (p : Parameters) : List ComputeStmt :=
  [.assign .int [p.block*2] "exp_biased_f32"
    (.alg (shiftExpr (.ref .nat [p.block*2] "exp_biased_f32") 23))]
def normalCode (p : Parameters) : List ComputeStmt :=
  [.assign .nat [p.block*2] "mantissa_f4"
    (.alg (.bitAnd .scalarR (.ref .nat [p.block*2] "x_pos") (.constNat 1))),
   .assign .int [p.block*2] "mantissa_f32"
    (.alg (shiftExpr (.ref .nat [p.block*2] "mantissa_f4") 22)),
   .assign .int [p.block*2] "output" (.alg (.ref .int [p.block*2] "mantissa_f32")),
   .assign .int [p.block*2] "result"
    (.alg (orExpr (.ref .int [p.block*2] "exp_biased_f32") (.ref .int [p.block*2] "mantissa_f32"))),
   .assign .int [p.block*2] "result" (.alg (.where (.ref .bool [p.block*2] "zero_mask")
     (.broadcast (intConst 0) [p.block*2]) (.ref .int [p.block*2] "result")))]
def signCode (p : Parameters) : List ComputeStmt :=
  [.assign .int [p.block*2] "result" (.alg (.where (.ref .bool [p.block*2] "denormal_mask")
     (.broadcast (intConst 1056964608) [p.block*2]) (.ref .int [p.block*2] "result"))),
   .assign .int [p.block*2] "sign_f32"
    (.alg (shiftExpr (.ref .nat [p.block*2] "sign_f4") 28)),
   .assign .int [p.block*2] "result"
    (.alg (orExpr (.ref .int [p.block*2] "result") (.ref .int [p.block*2] "sign_f32")))]

theorem exponentCode_source (p : Parameters) :
    ((source p).surfaceBody.drop 16 |>.take 1) = exponentCode p := rfl
theorem normalCode_source (p : Parameters) :
    ((source p).surfaceBody.drop 17 |>.take 5) = normalCode p := rfl
theorem signCode_source (p : Parameters) :
    ((source p).surfaceBody.drop 22 |>.take 3) = signCode p := rfl
def assignmentOnly : ComputeStmt → Bool
  | .assign .. => Bool.true
  | _ => Bool.false

theorem assignment_step_memory {α : Type} [Inhabited α] (M : Algebra α)
    (st : ComputeStmt) (s t : State α) (ha : assignmentOnly st = Bool.true)
    (h : step M st s = some t) : t.mem = s.mem := by
  cases st <;> simp only [assignmentOnly, Bool.false_eq_true] at ha
  case assign d sh name e =>
    cases he : evalExpr M e s with
    | none => simp [step, he] at h
    | some v =>
      simp only [step, he, bind, pure, Option.bind_some, Option.some.injEq] at h
      cases h
      rfl

theorem assignment_run_memory {α : Type} [Inhabited α] (M : Algebra α)
    (code : List ComputeStmt) (s t : State α) (ha : code.all assignmentOnly = Bool.true)
    (h : run M code s = some t) : t.mem = s.mem := by
  induction code generalizing s t with
  | nil =>
    simp only [run, Option.some.injEq] at h
    cases h
    rfl
  | cons st rest ih =>
    simp only [List.all_cons, Bool.and_eq_true] at ha
    cases hu : step M st s with
    | none => simp [run, hu] at h
    | some u =>
      have hr : run M rest u = some t := by
        simpa only [run, hu, bind, Option.bind_some] using h
      exact (ih u t ha.2 hr).trans (assignment_step_memory M st s u ha.1 hu)
#axiomsClean assignment_run_memory

#axiomsClean exponentCode_source
#axiomsClean normalCode_source
#axiomsClean signCode_source

#axiomsClean eval_nat_as_i32
#axiomsClean eval_shift32
#axiomsClean eval_or32
end
end VeriTile.Bench.TritonBenchG.Fp4ToBf16.FPValue
