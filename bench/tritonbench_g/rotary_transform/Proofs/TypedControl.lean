import bench.tritonbench_g.rotary_transform.Proofs.TypedLoads

namespace VeriTile.Bench.Repairs.RotaryTransformTileChecked.TypedExecution
open VeriTile.Bench.Repairs.RotaryTransform
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 300000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

theorem kernel_carrier (s : Shape) (cumulative64 offsets64 : Bool) :
    kernel s cumulative64 offsets64 = kernel s Bool.true Bool.true := by
  simp only [kernel, KernelSource.signed_carrier]

theorem run_if_not {α : Type} [Inhabited α] (M : Algebra α) (flag : Bool)
    (yes no : List ComputeStmt) (u : State α) :
    run M [.ifThenElse (.boolNot (.constBool flag)) yes no] u =
      if flag then run M no u else run M yes u := by
  cases flag <;> simp [FP.Structural.run,FP.Structural.step,evalOp_unfold]

theorem run_if {α : Type} [Inhabited α] (M : Algebra α)
    (cond : Op .bool []) (body : List ComputeStmt) (u : State α) :
    run M [.ifThen cond body] u =
      (evalOp M none cond u).bind (fun c => if c PUnit.unit then run M body u else some u) := by
  simp [FP.Structural.run,FP.Structural.step,Option.bind_assoc]

def splitBody (s : Shape) : List ComputeStmt :=
  ((continuation s).drop 4).flatMap (branchBody Bool.false)
def interleavedBody (s : Shape) : List ComputeStmt :=
  ((continuation s).drop 4).flatMap (branchBody Bool.true)

theorem continuation_tail {α : Type} [Inhabited α] (M : Algebra α) (s : Shape)
    (u : State α) : run M ((continuation s).drop 4) u = run M (payload s) u := by
  change run M [.ifThenElse (.boolNot (.constBool s.interleaved))
    (splitBody s) (interleavedBody s)] u = _
  rw [run_if_not]
  cases hi : s.interleaved <;>
    simp only [hi, Bool.false_eq_true, if_false, if_true]
  all_goals simp only [payload,splitBody,interleavedBody,hi]

def guardOf : List ComputeStmt → Op .bool []
  | [.ifThen cond _] => cond
  | _ => .constBool Bool.false
def guard (s : Shape) : Op .bool [] :=
  guardOf ((kernel s Bool.true Bool.true).surfaceBody.drop 6)

theorem source_tail (s : Shape) :
    (kernel s Bool.true Bool.true).surfaceBody.drop 6 =
      [.ifThen (guard s) (continuation s)] := by rfl

theorem guard_eval {α : Type} [Inhabited α] (M : Algebra α) (s : Shape)
    (initial : State α) :
    evalOp M none (guard s) (setup M s Bool.true Bool.true initial).val =
      some (fun _ => decide (((initial.pids 0*s.blockM : Nat) : Int) <
        ProgramModel.sequenceLength s initial)) := by
  by_cases hv : s.varlen = Bool.true
  all_goals
    simp [guard,guardOf,kernel,KernelSource.source,KernelSource.source64_64,
      ComputeKernel.surfaceBody,setup,hv,evalOp_unfold,numeric,numericLe,bop,
      ProgramModel.sequenceLength,Int.natCast_mul]
    funext i
    simp only [← decide_not,not_le,lt_sub_iff_add_lt]

theorem setup_mem {α : Type} [Inhabited α] (M : Algebra α) (s : Shape)
    (initial : State α) : (setup M s Bool.true Bool.true initial).val.mem=initial.mem := by
  by_cases hv : s.varlen = Bool.true <;> simp [setup,hv]

#axiomsClean kernel_carrier
#axiomsClean continuation_tail
#axiomsClean guard_eval
#axiomsClean setup_mem
end
end VeriTile.Bench.Repairs.RotaryTransformTileChecked.TypedExecution
