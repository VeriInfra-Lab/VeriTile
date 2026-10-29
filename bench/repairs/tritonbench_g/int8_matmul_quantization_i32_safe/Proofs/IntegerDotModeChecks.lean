import bench.repairs.tritonbench_g.int8_matmul_quantization_i32_safe.Proofs.IntegerDotBridge

/-! Semantic regression checks for the closed accumulatorNat profile. The
artificial scalar primitive distinguishes incoming carry and plan fusion; it
is not a production numerical assumption or a GPU experiment. -/
namespace VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.IntegerDotModeChecks
open VeriTile.Triton FP.Structural IntegerDot
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 28000

def arithmetic : Arithmetic where
  normalize:=id
  product8:=fun a b=>a.toInt*b.toInt
  add:=fun carry product=>10*carry+product
  mma32:=fun _ _ _ carry=>carry

def scalarPlan (k : Nat) (fused : Bool) : IntegerDot.Plan k where
  operations:=(List.finRange k).map IntegerDot.Micro.scalar
  complete:=by simp [List.flatMap_map,Function.comp_def,IntegerDot.Micro.terms]
  fused:=fused

def context : Context where
  rows:=1
  cols:=1
  rowRegister:="pid_m"
  columnRegister:="pid_n"
  iterationRegister:="k"
  splitAxis:=1
  plans:=fun k r c split t=>scalarPlan k (decide (r=2 ∧ c=3 ∧ split=4 ∧ t=5))
  mode:=.accumulatorNat

def initial (s : State F) : State F :=
  ((((({s with pids:=fun _=>4}).setReg "pid_m" .nat [] (fun _=>2)).setReg "pid_n" .nat [] (fun _=>3)).setReg
    "k" .nat [] (fun _=>5)).setReg "left" .int [1,2] (fun _=>1)).setReg "right" .int [2,1] (fun _=>1)

def carried (s : State F) : State F :=
  (initial s).setReg "accumulator" .int [1,1] (fun _=>7)

def update (name oldName : RegName) (options : DotOptions := {}) : ComputeStmt :=
  .assign .int [1,1] name (.alg (.add .int (.consSame (.consSame .nil))
    (.ref .int [1,1] oldName) (.dotInt (batch:=[]) (.ref .int [1,2] "left") (.ref .int [2,1] "right") options)))

/-- Required Nat context, unreduced coordinates and the old carry all matter:
this is 711, while a zero-seeded separate dot followed by add would be 81. -/
theorem exact_context_and_carry [Inhabited F] (M : Algebra F) (s : State F) :
    accumulateNat M arithmetic context (.ref .int [1,2] "left")
      (.ref .int [2,1] "right") (.ref .int [1,1] "accumulator") (carried s)=
      some (fun _=>711) := by
  simp [accumulateNat,carried,initial,State.setReg,evalOp_unfold,context,
    scalarPlan,IntegerDot.Plan.evalInt,IntegerDot.Micro.eval,Arithmetic.instructions,arithmetic]
  rfl

theorem missing_nat_context_fails [Inhabited F] (M : Algebra F) (s : State F) :
    accumulateNat M arithmetic context (.ref .int [1,2] "left")
      (.ref .int [2,1] "right") (.ref .int [1,1] "accumulator")
      ((carried s).setReg "pid_m" .int [] (fun _=>2))=none := by
  simp [accumulateNat,carried,initial,State.setReg,evalOp_unfold,context]

/-- A renamed destination and nondefault dot options must use Structural. -/
theorem renamed_destination_fallback [Inhabited F] (M : Algebra F) (s : State F) :
    arithmeticStep M arithmetic context (update "result" "accumulator") s=
      FP.Structural.step M (update "result" "accumulator") s := by
  rfl

theorem renamed_old_fallback [Inhabited F] (M : Algebra F) (s : State F) :
    arithmeticStep M arithmetic context (update "accumulator" "old") s=
      FP.Structural.step M (update "accumulator" "old") s := by
  rfl

theorem options_fallback [Inhabited F] (M : Algebra F) (s : State F)
    (options : DotOptions) (nondefault : options≠{}) :
    arithmeticStep M arithmetic context (update "accumulator" "accumulator" options) s=
      FP.Structural.step M (update "accumulator" "accumulator" options) s := by
  change (if _ then _ else _) = _
  rw [if_neg (by simpa using nondefault)]
  rfl

/-- ifThen deliberately does not recursively select the contextual mode. -/
theorem ifThen_fallback [Inhabited F] (M : Algebra F) (s : State F)
    (cond : Op .bool []) (body : List ComputeStmt) :
    IntegerDot.step M arithmetic context (.ifThen cond body) s=
      FP.Structural.step M (.ifThen cond body) s := by
  rw [IntegerDot.step_unfold]
  rfl

#axiomsClean exact_context_and_carry
#axiomsClean missing_nat_context_fails
#axiomsClean renamed_destination_fallback
#axiomsClean renamed_old_fallback
#axiomsClean options_fallback
#axiomsClean ifThen_fallback
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization.I32Safe.IntegerDotModeChecks
