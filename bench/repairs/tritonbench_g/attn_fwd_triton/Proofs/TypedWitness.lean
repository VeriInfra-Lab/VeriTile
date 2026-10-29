import bench.repairs.tritonbench_g.attn_fwd_triton.Proofs.TypedFP
import bench.tritonbench_g.context_attn_fwd.Proofs.FPModel

/-! Reuse an existing consistent abstract model. This is a non-vacuity witness,
not an assertion that its dot, casts or unary functions implement IEEE/CUDA. -/
namespace VeriTile.Bench.Repairs.AttnFwdTriton.TypedWitness
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.ContextAttnFwd.FPModel (model domain model_satisfies_rules)
open TypedExecution
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def support (BD : Nat) : DotSupport model BD where
  score := fun _ _ _ => 0
  score_available := fun _ _ => rfl
  value := fun _ _ _ => 0
  value_available := fun _ _ => rfl

def config : Config where
  Q := "Q"; K := "K"; V := "V"; QS := "QS"; KS := "KS"; Out := "Out"
  queryBatch := 1; queryHead := 1; queryRow := 1; queryDim := 1
  keyBatch := 1; keyHead := 1; keyRow := 1; keyDim := 1
  valueBatch := 1; valueHead := 1; valueRow := 1; valueDim := 1
  outputBatch := 1; outputHead := 1; outputRow := 1; outputDim := 1
  batches := 1; heads := 1; context := 1; headDim := 1; stage := 3
  validStage := Or.inr rfl

def initial : State ℝ where
  mem := fun _ _ => .mk .real 0
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => defaultValue d

def beforeFirst : {t : State ℝ // run (engine model) (preloop config 16) initial=some t} := by
  refine ⟨?_,?_⟩
  swap
  simp only [preloop,code,source,Bool.false_eq_true,↓reduceIte,TypedSource.repaired,
    ComputeKernel.surfaceBody,List.take_succ_cons,List.take_zero]
  simp (config := {maxSteps := 250000}) [config,initial,run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,
    engine,Algebra.withDefaultPrecision,resolvePrecision,model,
    Option.bind,ComputeDType.eraseDType,numeric,integralFloorDiv,integralMod,bop,
    natLt,natLe,numericLt,numericLe,Cell.read,ofFloat,toFloat,Region.cast,State.setReg,TileShape.axisDim]
  rfl

theorem first_done : run (engine model) (loopCode config 16 Bool.false) beforeFirst.val=some beforeFirst.val := by
  simp (config := {maxSteps := 250000}) [loopCode,lowerName,upperName,beforeFirst,config,initial,run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,
    engine,Algebra.withDefaultPrecision,resolvePrecision,model,
    Option.bind,ComputeDType.eraseDType,numeric,integralFloorDiv,integralMod,bop,
    natLt,natLe,numericLt,numericLe,Cell.read,ofFloat,toFloat,Region.cast,State.setReg,TileShape.axisDim]
  change (range (engine model) "start_n" 0 0 64 (body config 16 Bool.false) beforeFirst.val).bind some=some beforeFirst.val
  rw [Option.bind_fun_some]
  exact range_done (engine model) "start_n" 0 0 64 (body config 16 Bool.false) beforeFirst.val (by decide)

def afterSetup : {t : State ℝ // run (engine model) (setupCode config 16 Bool.true) beforeFirst.val=some t} := by
  refine ⟨?_,?_⟩
  swap
  simp only [setupCode,code,source,Bool.false_eq_true,Bool.true_eq,↓reduceIte,TypedSource.repaired,
    ComputeKernel.surfaceBody,List.drop_succ_cons,List.drop_zero,List.take_succ_cons,List.take_zero]
  simp (config := {maxSteps := 250000}) [beforeFirst,config,initial,run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,
    engine,Algebra.withDefaultPrecision,resolvePrecision,model,
    Option.bind,ComputeDType.eraseDType,numeric,integralFloorDiv,integralMod,bop,
    natLt,natLe,numericLt,numericLe,Cell.read,ofFloat,toFloat,Region.cast,State.setReg,TileShape.axisDim]
  rfl

def oneIteration : {t : State ℝ // run (engine model) (body config 16 Bool.true)
    (afterSetup.val.setReg "start_n" .nat [] (fun _ => 0))=some t} := by
  refine ⟨?_,?_⟩
  swap
  simp only [body,code,source,Bool.false_eq_true,Bool.true_eq,↓reduceIte,TypedSource.repaired,
    ComputeKernel.surfaceBody,Source.dynamicRange?,Option.map_some,Option.getD_some,beq_self_eq_true]
  simp (config := {maxSteps := 250000}) [afterSetup,beforeFirst,config,initial,run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,
    engine,Algebra.withDefaultPrecision,resolvePrecision,model,
    Option.bind,ComputeDType.eraseDType,numeric,integralFloorDiv,integralMod,bop,
    natLt,natLe,numericLt,numericLe,Cell.read,ofFloat,toFloat,Region.cast,State.setReg,TileShape.axisDim]
  rfl

theorem second_done : run (engine model) (loopCode config 16 Bool.true) afterSetup.val=some oneIteration.val := by
  have hb : run (engine model) (loopCode config 16 Bool.true) afterSetup.val =
      range (engine model) "start_n" 0 1 64 (body config 16 Bool.true) afterSetup.val := by
    simp (config := {maxSteps := 250000}) [loopCode,lowerName,upperName,afterSetup,beforeFirst,config,initial,run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,
    engine,Algebra.withDefaultPrecision,resolvePrecision,model,
    Option.bind,ComputeDType.eraseDType,numeric,integralFloorDiv,integralMod,bop,
    natLt,natLe,numericLt,numericLe,Cell.read,ofFloat,toFloat,Region.cast,State.setReg,TileShape.axisDim]
    exact Option.bind_fun_some (range (engine model) "start_n" 0 1 64 (body config 16 Bool.true) afterSetup.val)
  rw [hb,range_step (engine model) "start_n" 0 1 64 (body config 16 Bool.true) afterSetup.val (by decide) (by decide)]
  simp only [oneIteration.property,Option.bind_some]
  exact range_done (engine model) "start_n" 64 1 64 (body config 16 Bool.true) oneIteration.val (by decide)

theorem before_normalization :
    ∃ t, run (engine model) ((code config 16).take 41) initial=some t ∧
      t.regs .real [128] "l_i"=some (fun _ => 2) := by
  have hc : (code config 16).take 41=preloop config 16 ++ loopCode config 16 Bool.false ++
      setupCode config 16 Bool.true ++ loopCode config 16 Bool.true := rfl
  refine ⟨oneIteration.val,?_,?_⟩
  · rw [hc]
    simp only [run_append,beforeFirst.property,first_done,afterSetup.property,second_done,Option.bind_some]
  · simp (config := {maxSteps := 250000}) [oneIteration,afterSetup,beforeFirst,config,initial,run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,
    engine,Algebra.withDefaultPrecision,resolvePrecision,model,
    Option.bind,ComputeDType.eraseDType,numeric,integralFloorDiv,integralMod,bop,
    natLt,natLe,numericLt,numericLe,Cell.read,ofFloat,toFloat,Region.cast,State.setReg,TileShape.axisDim]
    funext i
    norm_num [bop]

theorem inhabited_domain :
    ∀ site∈(program config 16 Bool.false).domain, site.Holds model domain initial := by
  intro site hs
  change site∈[FPEquivProofs.divisionSite "Q" "K" "V" "QS" "KS" "Out"
    1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 128 64 16 3 (Or.inr rfl)] at hs
  simp only [List.mem_singleton] at hs
  subst site
  intro t ht a b ha hb
  obtain ⟨u,hu,hli⟩ := before_normalization
  have hp : Source.prefixBefore (code config 16) "acc" 1=(code config 16).take 41 := rfl
  change run (engine model) (Source.prefixBefore (code config 16) "acc" 1) initial=some t at ht
  rw [hp,hu] at ht
  have hut := Option.some.inj ht
  subst t
  have hb' : b=(fun _ => (2 : ℝ)) := by
    simp only [evalOp_unfold,hli,Option.bind_some] at hb
    exact (Option.some.inj hb).symm
  subst b
  exact ⟨fun _ => trivial,fun _ => trivial,fun _ => by norm_num [domain]⟩

/-- The same model validates every selected scalar rule, executes the entire
production-size tile graph, and inhabits the actual final normalization site. -/
theorem nonvacuous (R : FPEquivProofs.Rules) :
    FP.Guarded.Models R.assumptions model domain ∧
    (∃ t, exec (engine model) (source config 16 Bool.false) initial=some t ∧
      exec (engine model) (source config 16 Bool.true) initial=some t) ∧
    (∀ site∈(program config 16 Bool.false).domain,site.Holds model domain initial) :=
  ⟨model_satisfies_rules R,
    successful_equivalence R model domain (model_satisfies_rules R) config 16 (support 16) initial inhabited_domain,
    inhabited_domain⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttnFwdTriton.TypedWitness
