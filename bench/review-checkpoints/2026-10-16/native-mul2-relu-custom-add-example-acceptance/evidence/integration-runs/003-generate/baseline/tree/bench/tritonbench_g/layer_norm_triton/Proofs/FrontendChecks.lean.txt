import bench.tritonbench_g.layer_norm_triton.Proofs.SourceBridge

namespace VeriTile.Bench.TritonBenchG.LayerNormLock.Frontend

theorem missing_backward_rejected : compile { originalTypedSource with backward := [] } = none := by decide +kernel

theorem missing_forward_output_rejected :
    compile { originalTypedSource with forward := originalTypedSource.forward.take 10 } = none := by decide +kernel

theorem missing_final_stream_rejected :
    compile { originalTypedSource with finalReduce := [.setup,.zeroDW,.zeroDB,.reduceDW,.reduceDB,.storeDW,.storeDB] } = none := by decide +kernel

def removeWhile : BStmt → BStmt
  | .whileCAS .. => .setup
  | other => other

theorem ready_lock_replacement_rejected :
    compile { originalTypedSource with backward := originalTypedSource.backward.map removeWhile } = none := by decide +kernel

theorem extra_instruction_rejected :
    compile { originalTypedSource with backward := originalTypedSource.backward ++ [.setup] } = none := by decide +kernel

def retryZero : BStmt → BStmt
  | .whileCAS expected desired _ => .whileCAS expected desired 0
  | other => other

def changedRetrySource : TypedSource :=
  { originalTypedSource with backward := originalTypedSource.backward.map retryZero }

theorem source_cas_literal_is_retained :
    compile changedRetrySource = some { originalCompiled with lock.retry := 0 } := by decide +kernel

def heldControl : Control := ⟨fun _ => 1, fun _ => .poll⟩

theorem changed_retry_changes_execution :
    decodePlan originalCompiled.lock twoRows 0 heldControl = (.poll 0, .poll) ∧
    decodePlan { originalCompiled.lock with retry := 0 } twoRows 0 heldControl = (.poll 0, .count) := by decide +kernel

def zeroC1 : BStmt → BStmt
  | .assign 7 _ => .assign 7 .zero
  | other => other

def changedArithmeticSource : TypedSource :=
  { originalTypedSource with backward := originalTypedSource.backward.map zeroC1 }

def changedArithmeticGraph : Compiled :=
  { originalCompiled with backwardProgram :=
      originalCompiled.backwardProgram.map (fun x => if x.1 = 7 then (7,.zero) else x) }

theorem source_arithmetic_is_retained : compile changedArithmeticSource = some changedArithmeticGraph := by decide +kernel

def nonzeroCentered : Memory Word := fun region _ =>
  match region with
  | .x | .w | .dy | .rstd => one
  | _ => zero

set_option maxRecDepth 16000 in
set_option maxHeartbeats 2000000 in
theorem changed_arithmetic_changes_value :
    (compiledRows originalCompiled backwardBits twoRows oneLane nonzeroCentered exampleUndef 0).dx 0 =
      VeriTile.Triton.FP.Value.ofBits .fp32 0xbf800000 ∧
    (compiledRows changedArithmeticGraph backwardBits twoRows oneLane nonzeroCentered exampleUndef 0).dx 0 = zero := by
  decide +kernel

#axiomsClean missing_backward_rejected
#axiomsClean missing_forward_output_rejected
#axiomsClean missing_final_stream_rejected
#axiomsClean ready_lock_replacement_rejected
#axiomsClean extra_instruction_rejected
#axiomsClean source_cas_literal_is_retained
#axiomsClean changed_retry_changes_execution
#axiomsClean source_arithmetic_is_retained
#axiomsClean changed_arithmetic_changes_value
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.LayerNormLock.Frontend
