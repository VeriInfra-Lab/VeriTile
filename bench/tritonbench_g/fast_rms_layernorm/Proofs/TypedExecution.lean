import bench.tritonbench_g.fast_rms_layernorm.Proofs.FP
import bench.tritonbench_g.context_attn_fwd.Proofs.FPModel
import VeriTile.Triton.Float.ProfiledSoundness
namespace VeriTile.Bench.TritonBenchG.FastRmsLayernorm.TypedExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite TypedSource
open VeriTile.Bench.TritonBenchG.ContextAttnFwd.FPModel
  (Number one two model rest literal domain domain_strict)
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 25000
set_option linter.unusedSimpArgs false

def storageType : Storage → FloatDType
  | .fp16 => .fp16 | .bf16 => .bf16 | .fp32 => .real

def initial (input weight : Storage) : State Number where
  mem := fun r _ => if r="x" ∨ r="y" then .mk (storageType input).toTileDType (ofFloat (storageType input) two)
    else if r="w" then .mk (storageType weight).toTileDType (ofFloat (storageType weight) (FP.Value.ofBits .fp32 0x40400000))
    else .mk .real one
  regs := fun _ _ _ => none
  pids := fun ax => if ax=0 then 1 else 0
  numPids := fun _ => 2
  undef := fun d _ _ => defaultValue d

/-! Empty-register execution witnesses for every storage combination and both
backward constexpr choices. The witness uses the shared FP32-word software
model and its proved strict interpretation. Mixed-format primitive callbacks
remain abstract; this is an execution witness, not GPU conformance. -/

theorem forward_executes (input weight : Storage) :
    ∃ v, FP.Structural.exec (engine model)
      (forward input weight Bool.false "y" "x" "w" "r" 4 4 2 1 3 4 1) (initial input weight)=some v := by
  cases input <;> cases weight
  all_goals
    simp [FP.Structural.exec,forward,ComputeKernel.surfaceBody,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
      engine,Algebra.withDefaultPrecision,resolvePrecision,model,FP.Software.strictFP32,FP.Software.scalarFP32,FP.Software.fp32Context,
      rest,literal,ComputeDType.eraseDType,
      numeric,natLt,numericLt,numericLe,numericEq,bop,initial,storageType,ofFloat,toFloat,Region.cast,
      State.setReg,FP.Structural.store,TileShape.axisDim,TileShape.eraseAxis,TileShape.insertAxisIndex,
      TileShape.allIndices,List.finRange_succ,State.write]

theorem forward_candidate_executes (input weight : Storage) :
    ∃ v, FP.Structural.exec (engine model)
      (forward input weight Bool.true "y" "x" "w" "r" 4 4 2 1 3 4 1) (initial input weight)=some v := by
  obtain ⟨v,hv⟩ := forward_executes input weight
  have h := FP.ProfiledRewrite.sound
    (FPProof.forward_equivalent input weight "y" "x" "w" "r" 4 4 2 1 3 4 1)
    Number model domain (by
      apply FP.models_of_atoms
      · intro e he; cases he
      · exact FP.strict_software_interpretation rest domain domain_strict)
    (initial input weight) (by simp [FPProof.forwardProgram])
  exact ⟨v,h.symm.trans hv⟩

theorem gemma_executes (input weight : Storage) :
    ∃ v, FP.Structural.exec (engine model)
      (gemma input weight Bool.false "y" "x" "w" "r" 4 4 2 1 3 4 1) (initial input weight)=some v := by
  cases input <;> cases weight
  all_goals
    simp [FP.Structural.exec,gemma,ComputeKernel.surfaceBody,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
      engine,Algebra.withDefaultPrecision,resolvePrecision,model,FP.Software.strictFP32,FP.Software.scalarFP32,FP.Software.fp32Context,
      rest,literal,ComputeDType.eraseDType,
      numeric,natLt,numericLt,numericLe,numericEq,bop,initial,storageType,ofFloat,toFloat,Region.cast,
      State.setReg,FP.Structural.store,TileShape.axisDim,TileShape.eraseAxis,TileShape.insertAxisIndex,
      TileShape.allIndices,List.finRange_succ,State.write]

theorem gemma_candidate_executes (input weight : Storage) :
    ∃ v, FP.Structural.exec (engine model)
      (gemma input weight Bool.true "y" "x" "w" "r" 4 4 2 1 3 4 1) (initial input weight)=some v := by
  obtain ⟨v,hv⟩ := gemma_executes input weight
  have h := FP.ProfiledRewrite.sound
    (FPProof.gemma_equivalent input weight "y" "x" "w" "r" 4 4 2 1 3 4 1)
    Number model domain (by
      apply FP.models_of_atoms
      · intro e he; cases he
      · exact FP.strict_software_interpretation rest domain domain_strict)
    (initial input weight) (by simp [FPProof.gemmaProgram])
  exact ⟨v,h.symm.trans hv⟩

theorem backward_executes (input weight : Storage) (isGemma : Bool) :
    ∃ v, FP.Structural.exec (engine model)
      (backward input weight Bool.false "y" "x" "w" "r" "dw" 4 4 2 1 5 3 4 1 isGemma) (initial input weight)=some v := by
  cases input <;> cases weight <;> cases isGemma
  all_goals
    simp [FP.Structural.exec,backward,ComputeKernel.surfaceBody,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
      engine,Algebra.withDefaultPrecision,resolvePrecision,model,FP.Software.strictFP32,FP.Software.scalarFP32,FP.Software.fp32Context,
      rest,literal,ComputeDType.eraseDType,
      numeric,natLt,numericLt,numericLe,numericEq,bop,initial,storageType,ofFloat,toFloat,Region.cast,
      State.setReg,FP.Structural.store,TileShape.axisDim,TileShape.eraseAxis,TileShape.insertAxisIndex,
      TileShape.allIndices,List.finRange_succ,State.write]

theorem backward_candidate_executes (input weight : Storage) (isGemma : Bool) :
    ∃ v, FP.Structural.exec (engine model)
      (backward input weight Bool.true "y" "x" "w" "r" "dw" 4 4 2 1 5 3 4 1 isGemma) (initial input weight)=some v := by
  obtain ⟨v,hv⟩ := backward_executes input weight isGemma
  have h := FP.ProfiledRewrite.sound
    (FPProof.backward_equivalent input weight "y" "x" "w" "r" "dw" 4 4 2 1 5 3 4 1 isGemma)
    Number model domain (by
      apply FP.models_of_atoms
      · intro e he; cases he
      · exact FP.strict_software_interpretation rest domain domain_strict)
    (initial input weight) (by simp [FPProof.backwardProgram])
  exact ⟨v,h.symm.trans hv⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.FastRmsLayernorm.TypedExecution
