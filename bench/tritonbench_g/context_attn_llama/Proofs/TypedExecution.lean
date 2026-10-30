import bench.tritonbench_g.context_attn_llama.Proofs.FPModel
import VeriTile.Triton.Float.ProfiledSoundness

namespace VeriTile.Bench.TritonBenchG.ContextAttnLlama.TypedExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open FPModel (Number zero one two sentinel maskedScore model rest literal domain model_satisfies_typed_rules)
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

def initial : State Number where
  mem := fun r _ =>
    if r = "start" ∨ r = "prompt" then .mk .int 1
    else if r = "sequence" then .mk .int 4
    else if r = "request" ∨ r = "pages" then .mk .int 1
    else if r = "q" ∨ r = "k" ∨ r = "v" then .mk .fp16 one
    else .mk .fp16 sentinel
  regs := fun _ _ _ => none
  pids := fun d => if d = 1 then 7 else 1
  numPids := fun _ => 8
  undef := fun d _ _ => defaultValue d

def original := FPProof.original .fp16 "q" "k" "v" 2 "out" "start" "sequence" "pages" "request" "prompt"
  8 2 1 2 8 1 2 8 1 8 2 1 64 1 2 4 2 2 4

def candidate := FPProof.scaleFirst .fp16 "q" "k" "v" 2 "out" "start" "sequence" "pages" "request" "prompt"
  8 2 1 2 8 1 2 8 1 8 2 1 64 1 2 4 2 2 4

def parts : Source.IntRange :=
  match Source.intRange? original.kernel.surfaceBody "start_n" with
  | some p => p
  | none => ⟨[],.constInt 0,.constInt 0,.constNat 1,[],[]⟩

theorem source_split : original.kernel.surfaceBody =
    parts.before ++ [.forRangeInt "start_n" parts.start parts.stop parts.stride parts.body] ++ parts.after := rfl

def preloop : {t : State Number // run (engine model) parts.before initial = some t} := by
  refine ⟨?_,?_⟩
  swap
  simp only [parts,Source.intRange?,original,FPProof.original,VeriTile.Bench.TritonBenchG.ContextAttnLlama.original,ComputeKernel.surfaceBody]
  simp (config := {maxSteps := 100000}) [run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    engine,Algebra.withDefaultPrecision,resolvePrecision,model,FP.Software.strictFP32,FP.Software.scalarFP32,FP.Software.fp32Context,rest,literal,ComputeDType.eraseDType,
    numeric,natLt,natLe,numericLt,numericLe,bop,initial,ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,
    integralFloorDiv,integralMod]
  try rfl

def first : {t : State Number // run (engine model) parts.body
    (preloop.val.setReg "start_n" .int [] (fun _ => 0)) = some t} := by
  refine ⟨?_,?_⟩
  swap
  simp only [parts,Source.intRange?,original,FPProof.original,VeriTile.Bench.TritonBenchG.ContextAttnLlama.original,ComputeKernel.surfaceBody]
  simp (config := {maxSteps := 100000}) [preloop,run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    engine,Algebra.withDefaultPrecision,resolvePrecision,model,FP.Software.strictFP32,FP.Software.scalarFP32,FP.Software.fp32Context,rest,literal,ComputeDType.eraseDType,
    numeric,natLt,natLe,numericLt,numericLe,bop,initial,ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,
    integralFloorDiv,integralMod]
  try rfl


theorem loop_step : step (engine model)
    (.forRangeInt "start_n" parts.start parts.stop parts.stride parts.body) preloop.val = some first.val := by
  have hs : FP.Structural.evalOp (engine model) none parts.start preloop.val = some (fun _ => (0 : Int)) := rfl
  have he : FP.Structural.evalOp (engine model) none parts.stop preloop.val = some (fun _ => (4 : Int)) := rfl
  have hd : FP.Structural.evalOp (engine model) none parts.stride preloop.val = some (fun _ => (4 : Nat)) := rfl
  simp only [step, hs, he, hd, bind, Option.bind_some]
  rw [rangeInt]
  change (run (engine model) parts.body (preloop.val.setReg "start_n" .int [] (fun _ => 0))).bind
    (rangeInt (engine model) "start_n" 4 4 4 parts.body) = _
  simp only [first.property, Option.bind_some]
  simp [rangeInt]

def postloop : {t : State Number // run (engine model) parts.after first.val = some t} := by
  refine ⟨?_,?_⟩
  swap
  simp only [parts,Source.intRange?,original,FPProof.original,VeriTile.Bench.TritonBenchG.ContextAttnLlama.original,ComputeKernel.surfaceBody]
  simp (config := {maxSteps := 100000}) [first,preloop,run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    engine,Algebra.withDefaultPrecision,resolvePrecision,model,FP.Software.strictFP32,FP.Software.scalarFP32,FP.Software.fp32Context,rest,literal,ComputeDType.eraseDType,
    numeric,natLt,natLe,numericLt,numericLe,bop,initial,ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,
    integralFloorDiv,integralMod,FP.Structural.store]
  try rfl

theorem executes : FP.Structural.exec (engine model) original.kernel initial = some postloop.val := by
  change run (engine model) original.kernel.surfaceBody initial = _
  rw [source_split,run_append,run_append]
  simp only [preloop.property,Option.bind_some,run,loop_step,postloop.property]
  exact postloop.property

/-- The reached scaling site, with the original typed operands and loop prefix. -/
def rewriteSite : Site :=
  .intRange parts.before "start_n" parts.start parts.stop parts.stride parts.body
    (.mulScalarTypedFP32 (Source.prefixBefore parts.body "qk" 1) 2 [4]
      (.ref .fp32 [2,4] "qk") (.constFloat .fp32 2))

theorem domain_sites : original.domain = [rewriteSite] := rfl

def score : {t : State Number //
    run (engine model) (Source.prefixBefore parts.body "qk" 1)
      (preloop.val.setReg "start_n" .int [] (fun _ => 0)) = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp only [parts,Source.intRange?,original,FPProof.original,
    VeriTile.Bench.TritonBenchG.ContextAttnLlama.original,ComputeKernel.surfaceBody,Source.prefixBefore]
  simp (config := {maxSteps := 100000}) [preloop,Source.prefixBefore,String.reduceEq,run,step,FP.Structural.evalExpr,
    evalComputeOp,evalOp_unfold,Option.bind,engine,Algebra.withDefaultPrecision,
    resolvePrecision,model,FP.Software.strictFP32,FP.Software.scalarFP32,
    FP.Software.fp32Context,rest,literal,ComputeDType.eraseDType,numeric,natLt,natLe,
    numericLt,numericLe,bop,initial,ofFloat,toFloat,TileShape.axisDim,Region.cast,
    State.setReg,integralFloorDiv,integralMod]
  try rfl

private theorem first_site_holds :
    (Site.mulScalarTypedFP32 (Source.prefixBefore parts.body "qk" 1) 2 [4]
      (.ref .fp32 [2,4] "qk") (.constFloat .fp32 2)).Holds model domain
      (preloop.val.setReg "start_n" .int [] (fun _ => 0)) := by
  intro t ht a b ha hb
  have ht' := Option.some.inj (score.property.symm.trans ht)
  subst t
  have ha0 : FP.Structural.evalOp (engine model) none (.ref .fp32 [2,4] "qk") score.val =
      some (fun _ => zero) := rfl
  have hb2 : FP.Structural.evalOp (engine model) none (.constFloat .fp32 2) score.val =
      some (fun _ => two) := by
    simp [evalOp_unfold,engine,Algebra.withDefaultPrecision,resolvePrecision,model,
      FP.Software.strictFP32,FP.Software.scalarFP32,FP.Software.fp32Context,rest,literal,ofFloat]
  have ha' := Option.some.inj (ha0.symm.trans ha)
  have hb' := Option.some.inj (hb2.symm.trans hb)
  subst a
  subst b
  exact ⟨fun _ => ⟨.zero, rfl⟩, ⟨.two, rfl⟩⟩

private theorem reached_nonnegative {cur : Int} {s : State Number}
    (h : IntRangeReachable (engine model) "start_n" 4 4 parts.body 0 preloop.val cur s) :
    0 ≤ cur := by
  induction h with
  | initial => norm_num
  | next _ _ _ _ ih => omega

theorem domain_holds : ∀ site ∈ original.domain, site.Holds model domain initial := by
  intro site member
  rw [domain_sites] at member
  simp only [List.mem_singleton] at member
  subst site
  intro t ht a b d ha hb hd cur u hr hstride hcur
  have ht' := Option.some.inj (preloop.property.symm.trans ht)
  subst t
  have hs : FP.Structural.evalOp (engine model) none parts.start preloop.val =
      some (fun _ => (0 : Int)) := rfl
  have he : FP.Structural.evalOp (engine model) none parts.stop preloop.val =
      some (fun _ => (4 : Int)) := rfl
  have hstep : FP.Structural.evalOp (engine model) none parts.stride preloop.val =
      some (fun _ => (4 : Nat)) := rfl
  have ha' := Option.some.inj (hs.symm.trans ha)
  have hb' := Option.some.inj (he.symm.trans hb)
  have hd' := Option.some.inj (hstep.symm.trans hd)
  subst a
  subst b
  subst d
  cases hr with
  | initial => exact first_site_holds
  | @next previous before after reached _ _ _ =>
    have hnonnegative := reached_nonnegative reached
    change previous + (4 : Int) < 4 at hcur
    omega

theorem candidate_execution (R : FP.TypedScalarArithmetic.Rules) :
    FP.Structural.exec (engine model) candidate.kernel initial = some postloop.val := by
  have he := FP.ProfiledRewrite.sound (FPProof.equivalent .fp16 "q" "k" "v" 2 "out" "start" "sequence" "pages" "request" "prompt"
    8 2 1 2 8 1 2 8 1 8 2 1 64 1 2 4 2 2 4 R)
  exact (he Number model domain (model_satisfies_typed_rules R) initial domain_holds).symm.trans executes


/-- The one reached block writes both channels of its active row and preserves the
masked next row; the backend is the named abstract consistency witness. -/
theorem output_cells :
    (FP.Structural.exec (engine model) original.kernel initial).map
      (fun t => (t.mem "out" 30,t.mem "out" 31,t.mem "out" 38)) =
      some (.mk .fp16 zero,.mk .fp16 zero,.mk .fp16 sentinel) := by
  simp only [executes,Option.map_some,Option.some.injEq,Prod.mk.injEq]
  repeat' apply And.intro
  all_goals
    (conv_lhs => whnf) <;>
    norm_num [bop,Function.comp_def,numeric,ofFloat,toFloat,Broadcast.leftIndex,Broadcast.rightIndex,
      TileShape.dropInsertedIndex,State.write,State.setReg,TileShape.allIndices,List.finRange,
      List.ofFn_succ,String.reduceEq] <;> decide +kernel

#axiomsClean executes
#axiomsClean domain_holds
#axiomsClean candidate_execution
#axiomsClean output_cells
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ContextAttnLlama.TypedExecution
