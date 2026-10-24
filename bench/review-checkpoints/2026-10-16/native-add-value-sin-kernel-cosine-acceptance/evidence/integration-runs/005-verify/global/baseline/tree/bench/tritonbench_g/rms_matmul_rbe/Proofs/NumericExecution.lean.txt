import bench.tritonbench_g.rms_matmul_rbe.Proofs.NumericStore

namespace VeriTile.Bench.TritonBenchG.RmsMatmulRbe.NumericExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
open scoped Classical
set_option maxHeartbeats 4000000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false
/-- Consecutive pieces of each actual source callee, including its complete
nonempty ceil-K loop, post-loop arithmetic and masked store. -/
theorem pass_partition (fp8 variant : Bool) (p : Pass) :
    (pass fp8 variant p).surfaceBody =
      ((pass fp8 Bool.false p).surfaceBody.take 12) ++
      ((pass fp8 Bool.false p).surfaceBody.drop 12 |>.take 1) ++
      ((pass fp8 Bool.false p).surfaceBody.drop 13 |>.take 3) ++
      ((pass fp8 variant p).surfaceBody.drop 16 |>.take 4) ++
      ((pass fp8 variant p).surfaceBody.drop 20) := by
  cases fp8 <;> cases variant <;> cases p <;> rfl

/-- A single actual typed store advances the independently specified memory
phase; all other raw cells, including previous outputs and bytes, are retained. -/
theorem memory_after (fp8 : Bool) (p : Pass) :
    (fun r o => if r = outRegion p ∧ o = 0 then outputCell fp8 p
      else memory fp8 (before p) r o) = memory fp8 (before p + 1) := by
  cases fp8 <;> cases p
  all_goals
    funext r o
    by_cases hq : r = "Q" <;> by_cases hk : r = "K" <;> by_cases hv : r = "V"
    all_goals simp [memory, before, outRegion, outputCell, hq, hk, hv]

/-- The staged execution is the complete original callee from arbitrary entry
registers. Its only memory/id hypotheses identify the concrete source inputs. -/
def pass_run (fp8 variant : Bool) (p : Pass) (s : State (WithBot ℝ))
    (hp : s.pids = fun _ => 0) (hm : s.mem = memory fp8 (before p)) :
    {t : State (WithBot ℝ) // run (engine model) (pass fp8 variant p).surfaceBody s = some t ∧
      t.mem = memory fp8 (before p + 1) ∧ t.pids = fun _ => 0} := by
  have hstart : startState fp8 p s = s := by
    cases s
    simp_all [startState]
  have hi := loop_registers fp8 p s
  refine ⟨(store_run fp8 variant p s).val, ?_, ?_, ?_⟩
  · have he : run (engine model) (pass fp8 variant p).surfaceBody (startState fp8 p s) =
        some (store_run fp8 variant p s).val := by
      conv_lhs =>
        arg 2
        rw [pass_partition]
      simp only [run_append, (prefix_run fp8 p s).property, Option.bind_some,
        (loop_run fp8 p s).property, (math_run fp8 p s).property,
        (address_run fp8 variant p s).property, (store_run fp8 variant p s).property]
    exact (congrArg (run (engine model) (pass fp8 variant p).surfaceBody) hstart).symm.trans he
  · simpa only [store_run, address_run, math_run, addressState, mathState,
      State.setReg_mem, State.write, hi.2.2.2.2.2.1, startState] using memory_after fp8 p
  · simp only [store_run, address_run, math_run, addressState, mathState,
      State.setReg_pids, State.write_pids, hi.2.2.2.2.2.2, startState]

/-- Compose checked source passes without expanding a prior pass's private
register history into the next. All three actual output values are retained. -/
def numeric_run (fp8 variant : Bool) :
    {t : State (WithBot ℝ) // run (engine model) (kernelExample fp8 variant).surfaceBody
      (initial fp8) = some t ∧ t.mem = memory fp8 3} := by
  let q := pass_run fp8 variant .q (initial fp8) rfl rfl
  let k := pass_run fp8 variant .k q.val q.property.2.2 q.property.2.1
  let v := pass_run fp8 variant .v k.val k.property.2.2 k.property.2.1
  refine ⟨v.val, ?_, v.property.2.1⟩
  change run (engine model) ((pass fp8 variant .q).surfaceBody ++
    (pass fp8 variant .k).surfaceBody ++ (pass fp8 variant .v).surfaceBody) (initial fp8) = _
  simp only [run_append, q.property.1, Option.bind_some, k.property.1, v.property.1]

/-- Actual nonzero Q/K/V values and typed store destinations, for both
weight profiles and both variants. Unwritten cells retain raw integer sentinels;
byte weights remain byte weights after the complete three-pass execution. -/
theorem numeric_cells (fp8 variant : Bool) :
    (numeric_run fp8 variant).val.mem "Q" 0 = .mk .fp16 (some 2) ∧
    (numeric_run fp8 variant).val.mem "K" 0 =
      (if fp8 then .mk .f8e5 (some 3) else .mk .fp16 (some 3)) ∧
    (numeric_run fp8 variant).val.mem "V" 0 =
      (if fp8 then .mk .f8e5 (some 4) else .mk .fp16 (some 4)) ∧
    (numeric_run fp8 variant).val.mem "Q" 2 = .mk .int 37 ∧
    (numeric_run fp8 variant).val.mem "K" 2 = .mk .int 37 ∧
    (numeric_run fp8 variant).val.mem "V" 2 = .mk .int 37 ∧
    (numeric_run fp8 variant).val.mem "QW" 3 =
      (if fp8 then .mk .int 64 else .mk .fp16 (some 2)) := by
  rw [(numeric_run fp8 variant).property.2]
  cases fp8 <;> simp [memory]

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RmsMatmulRbe.NumericExecution
