import bench.tritonbench_g.flash_attn.Proofs.ActualFP
import bench.tritonbench_g.flash_attn.Proofs.NumericTypedValues

/-! Complete typed execution of the unmodified source. Entry and loop states
are shared with the earlier witness; the exact source stores are executed
separately. No erased-real program replaces this numerical execution. -/
namespace VeriTile.Bench.TritonBenchG.FlashAttn.ActualNumericExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite TypedSource NumericTypedExecution
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

def original (causal : Bool) :=
  ActualFPProof.program Bool.false .fp16 "Q" "K" "V" "L" "O" 1
    50 20 5 2 50 20 7 2 50 20 99 3 50 20 5 2 1 2 3 2 2 2 causal

def parts (causal : Bool) : Source.DynamicRange :=
  (Source.dynamicRange? (original causal).kernel.surfaceBody "start_n").getD
    ⟨[],.constNat 0,.constNat 0,.constNat 0,[],[]⟩

def post (causal : Bool) :
    {t : State (WithBot ℝ) // run (engine model) (parts causal).after
      (afterLoop causal)=some t} := by
  cases causal
  all_goals
    refine ⟨?_,?_⟩
    swap
    simp only [parts,original,ActualFPProof.program,actualKernel,actual_fp16,
      ComputeKernel.surfaceBody,Source.dynamicRange?,Option.map,Option.getD,
      String.reduceEq,beq_self_eq_true,Bool.false_eq_true,↓reduceIte]
    simp (maxSteps := 300000) [run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,evalDeltas,
      Option.bind,engine,Algebra.withDefaultPrecision,resolvePrecision,model,
      ComputeDType.eraseDType,numeric,bop,numericLe,numericLt,numericEq,natLt,
      initial,storageType,Cell.read,ofFloat,toFloat,TileShape.axisDim,Region.cast,afterLoop,entry,first,second,
      FP.Structural.store,masked_scatter_regs]
    try rfl

theorem post_eq (causal : Bool) : (post causal).val = (NumericTypedExecution.post causal).val := by
  cases causal <;> rfl

theorem source_parts (causal : Bool) :
    (original causal).kernel.surfaceBody = (parts causal).before ++
      .forRangeDyn "start_n" (.ref .nat [] "lo") (.ref .nat [] "hi") (.constNat 2)
        (parts causal).body :: (parts causal).after := by rfl

/-- Nonempty complete typed source runs, including every padded KV load and
both exact source stores, in causal and noncausal modes. -/
theorem original_execution (causal : Bool) :
    FP.Structural.exec (engine model) (original causal).kernel
      (initial .fp16 .fp16)=some (post causal).val := by
  have before : run (engine model) (parts causal).before (initial .fp16 .fp16) =
      some (entry causal).val := (entry causal).property
  have loop : step (engine model)
      (.forRangeDyn "start_n" (.ref .nat [] "lo") (.ref .nat [] "hi") (.constNat 2)
        (parts causal).body) (entry causal).val = some (afterLoop causal) := loop_executes causal
  change run (engine model) (original causal).kernel.surfaceBody _ = _
  rw [source_parts,run_append]
  simp only [before,Option.bind_some,run,loop,bind,(post causal).property]

theorem candidate_execution (causal : Bool) :
    FP.Structural.exec (engine model)
      (ActualFPProof.program Bool.true .fp16 "Q" "K" "V" "L" "O" 1
        50 20 5 2 50 20 7 2 50 20 99 3 50 20 5 2 1 2 3 2 2 2 causal).kernel
      (initial .fp16 .fp16)=some (post causal).val := by
  have h := FP.ProfiledRewrite.sound
    (ActualFPProof.equivalent .fp16 "Q" "K" "V" "L" "O" 1
      50 20 5 2 50 20 7 2 50 20 99 3 50 20 5 2 1 2 3 2 2 2 causal)
    (WithBot ℝ) model (fun _ _ => True) (by apply FP.models_of_atoms;intro e he;cases he)
    (initial .fp16 .fp16) (by simp [ActualFPProof.program])
  exact h.symm.trans (original_execution causal)

theorem noncausal_cells :
    (post Bool.false).val.mem "O" 20 = Cell.mk .fp16 (some 1) ∧
    (post Bool.false).val.mem "L" 3 = Cell.mk .real (some 2) := by
  rw [post_eq]
  exact noncausal_output_cells

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.FlashAttn.ActualNumericExecution
