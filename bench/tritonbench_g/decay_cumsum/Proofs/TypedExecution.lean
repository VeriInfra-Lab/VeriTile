import bench.tritonbench_g.decay_cumsum.Proofs.MaskEquivalence
import VeriTile.Triton.Float.Countermodel
import VeriTile.Triton.Float.Control
import VeriTile.Triton.Float.ProfiledSoundness
namespace VeriTile.Bench.TritonBenchG.DecayCumsum.TypedExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite TypedSource
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

/-- A local, consistent abstract arithmetic interpretation used only to
witness complete executions; it does not claim GPU numerical refinement. -/
def model : Algebra ℝ where
  literal := fun _ _ x => x
  negInf := 0
  binary := fun _ _ op a b => match op with
    | .add => a+b | .sub => a-b | .mul => a*b | .div => a/b
    | .max => max a b | .pow => 0
  unary := fun _ _ _ => 1
  cast := fun _ _ _ x => x
  fromNat := fun _ n => n
  fromInt := fun _ n => n
  fp32Bits := fun _ => 0
  fp32Load := id
  reduceMax := fun _ {_} _ _ _ _ => 0
  reduceSum := fun _ {_} _ _ _ _ => 1

theorem range_int_done (M : Algebra ℝ) (idx : RegName) (cur stop : Int)
    (stride : Nat) (body : List ComputeStmt) (s : State ℝ) (h : stop≤cur) :
    rangeInt M idx cur stop stride body s = some s := by
  rw [rangeInt]
  simp [Int.not_lt.mpr h]

theorem range_int_step (M : Algebra ℝ) (idx : RegName) (cur stop : Int)
    (stride : Nat) (body : List ComputeStmt) (s : State ℝ)
    (hs : stride≠0) (h : cur<stop) :
    rangeInt M idx cur stop stride body s =
      (run M body (s.setReg idx .int [] (fun _ => cur))).bind
        (rangeInt M idx (cur+stride) stop stride body) := by
  rw [rangeInt]
  simp [hs,h]

def initial : State ℝ where
  mem := fun r _ => if r ∈ (["g","go","dg"] : List RegionName) then .mk .bf16 1 else .mk .fp16 1
  regs := fun _ _ _ => none
  pids := fun _ => 1
  numPids := fun _ => 2
  undef := fun d _ _ => defaultValue d

/-! These executions witness inhabited model premises for full programs,
with independent data/gate storage, two loop iterations and a partial lane.
The backend is a consistent abstract interpretation, not GPU evaluation. -/

def forwardKernel := forward .bf16 Bool.false "g" "go" 16 3 1 2 2 4 2 2 3 1

def forwardExecution : {t : State ℝ // FP.Structural.exec (engine model) forwardKernel initial = some t} := by
  refine ⟨?_,?_⟩
  swap
  dsimp only [forwardKernel,forward,ComputeKernel.surfaceBody]
  dsimp only [FP.Structural.exec]
  simp [FP.Structural.exec,forwardKernel,forward,ComputeKernel.surfaceBody,
    run,step,range_int_done,range_int_step,range_zero,range_done,range_step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,
    numeric,natLt,numericLt,numericLe,numericEq,bop,initial,ofFloat,toFloat,Region.cast,
    FP.Structural.store,TileShape.allIndices,List.finRange_succ,State.write]
  rfl

theorem forward_candidate_executes :
    FP.Structural.exec (engine model) (forward .bf16 Bool.true "g" "go" 16 3 1 2 2 4 2 2 3 1)
      initial = some forwardExecution.val := by
  -- These fixed executions differ only in exact integer mask products.
  -- Execute both actual bodies directly; the abstract Real backend need not
  -- model unrelated strict software floating certificates.
  simp [forwardExecution, forwardKernel, forward, ComputeKernel.surfaceBody,
    FP.Structural.exec, run, step, range_int_done, range_int_step,
    range_zero, range_done, range_step, evalExpr, evalComputeOp, evalOp_unfold, Option.bind,
    engine, Algebra.withDefaultPrecision, resolvePrecision, model, ComputeDType.eraseDType,
    numeric, natLt, numericLt, numericLe, numericEq, bop, initial, ofFloat, toFloat, Region.cast,
    FP.Structural.store, TileShape.allIndices, List.finRange_succ, State.write, Bool.and_comm]
  try rfl

def prepareKernel := prepare .fp16 .bf16 Bool.false "q" "k" "g" "qg" "kg" 16 3 1 2 2 4 2 2 3 1

def prepareExecution : {t : State ℝ // FP.Structural.exec (engine model) prepareKernel initial = some t} := by
  refine ⟨?_,?_⟩
  swap
  dsimp only [prepareKernel,prepare,ComputeKernel.surfaceBody]
  dsimp only [FP.Structural.exec]
  simp [FP.Structural.exec,prepareKernel,prepare,ComputeKernel.surfaceBody,
    run,step,range_int_done,range_int_step,range_zero,range_done,range_step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,
    numeric,natLt,numericLt,numericLe,numericEq,bop,initial,ofFloat,toFloat,Region.cast,
    FP.Structural.store,TileShape.allIndices,List.finRange_succ,State.write]
  rfl

theorem prepare_candidate_executes :
    FP.Structural.exec (engine model) (prepare .fp16 .bf16 Bool.true "q" "k" "g" "qg" "kg" 16 3 1 2 2 4 2 2 3 1)
      initial = some prepareExecution.val := by
  -- These fixed executions differ only in exact integer mask products.
  -- Execute both actual bodies directly; the abstract Real backend need not
  -- model unrelated strict software floating certificates.
  simp [prepareExecution, prepareKernel, prepare, ComputeKernel.surfaceBody,
    FP.Structural.exec, run, step, range_int_done, range_int_step,
    range_zero, range_done, range_step, evalExpr, evalComputeOp, evalOp_unfold, Option.bind,
    engine, Algebra.withDefaultPrecision, resolvePrecision, model, ComputeDType.eraseDType,
    numeric, natLt, numericLt, numericLe, numericEq, bop, initial, ofFloat, toFloat, Region.cast,
    FP.Structural.store, TileShape.allIndices, List.finRange_succ, State.write, Bool.and_comm]
  try rfl

def backwardKernel := backward .fp16 .bf16 Bool.false "dq_inner" "dq_inter" "dk_inner" "dk_inter" "q" "k" "g" "dg" 16 3 1 2 2 4 2 2 3 1

def backwardExecution : {t : State ℝ // FP.Structural.exec (engine model) backwardKernel initial = some t} := by
  refine ⟨?_,?_⟩
  swap
  dsimp only [backwardKernel,backward,ComputeKernel.surfaceBody]
  dsimp only [FP.Structural.exec]
  simp [FP.Structural.exec,backwardKernel,backward,ComputeKernel.surfaceBody,
    run,step,range_int_done,range_int_step,range_zero,range_done,range_step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,
    numeric,natLt,numericLt,numericLe,numericEq,bop,initial,ofFloat,toFloat,Region.cast,
    FP.Structural.store,TileShape.allIndices,List.finRange_succ,State.write]
  rfl

theorem backward_candidate_executes :
    FP.Structural.exec (engine model) (backward .fp16 .bf16 Bool.true "dq_inner" "dq_inter" "dk_inner" "dk_inter" "q" "k" "g" "dg" 16 3 1 2 2 4 2 2 3 1)
      initial = some backwardExecution.val := by
  -- These fixed executions differ only in exact integer mask products.
  -- Execute both actual bodies directly; the abstract Real backend need not
  -- model unrelated strict software floating certificates.
  simp [backwardExecution, backwardKernel, backward, ComputeKernel.surfaceBody,
    FP.Structural.exec, run, step, range_int_done, range_int_step,
    range_zero, range_done, range_step, evalExpr, evalComputeOp, evalOp_unfold, Option.bind,
    engine, Algebra.withDefaultPrecision, resolvePrecision, model, ComputeDType.eraseDType,
    numeric, natLt, numericLt, numericLe, numericEq, bop, initial, ofFloat, toFloat, Region.cast,
    FP.Structural.store, TileShape.allIndices, List.finRange_succ, State.write, Bool.and_comm]
  try rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.DecayCumsum.TypedExecution
