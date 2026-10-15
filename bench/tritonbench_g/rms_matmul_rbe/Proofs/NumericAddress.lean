import bench.tritonbench_g.rms_matmul_rbe.Proofs.NumericMath

namespace VeriTile.Bench.TritonBenchG.RmsMatmulRbe.NumericExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
open scoped Classical
set_option maxHeartbeats 4000000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false


def outRegion : Pass → RegionName | .q => "Q" | .k => "K" | .v => "V"

def addressState (t : State (WithBot ℝ)) (p : Pass) : State (WithBot ℝ) :=
  (((t.setReg "offs_m" .nat [1] (fun _ => 0)).setReg "offs_n" .nat [2]
    (fun i => i.1.val)).setReg "out_ptrs" .ptr [1, 2]
    (fun i => (outRegion p, i.2.1.val * 2))).setReg "out_mask" .bool [1, 2]
    (fun i => decide (i.2.1.val < 1))

theorem address_step (fp8 variant : Bool) (p : Pass) (t : State (WithBot ℝ))
    (hm : t.regs .nat [] "pid_m" = some (fun _ => 0))
    (hn : t.regs .nat [] "pid_n" = some (fun _ => 0))
    (hb : t.regs .nat [] "pid_batch" = some (fun _ => 0)) :
    run (engine model) ((pass fp8 variant p).surfaceBody.drop 16 |>.take 4) t =
      some (addressState t p) := by
  cases fp8 <;> cases variant <;> cases p
  all_goals simp (maxSteps := 1000000) [addressState, outRegion, hm, hn, hb, pass, TypedSource.kernel, ComputeKernel.surfaceBody, run, step,
    FP.Structural.evalExpr, evalComputeOp, evalOp_unfold, evalOffsets, evalDeltas,
    Option.bind, engine, Algebra.withDefaultPrecision, resolvePrecision, model,
    ComputeDType.eraseDType, numeric, integralFloorDiv, integralMod, bop, natLt,
    Cell.read, ofFloat, toFloat, TileShape.axisDim, Region.cast,
    FP.Structural.store, TileShape.allIndices, List.finRange_succ, Tile.select, Tile.ptrAdd,
    Broadcast.leftIndex, Broadcast.rightIndex, Tile.scalar, -Tile.scalar_eta,
    Float8E5Bits.isInt8, Float8E5Bits.ofInt]
  all_goals
    congr 1
    · congr 1
      · congr 1
        · funext i
          simp [bop, Broadcast.leftIndex, Broadcast.rightIndex]
      · funext i
        simp [bop, Broadcast.leftIndex, Broadcast.rightIndex]
    · funext i
      simp [bop, Broadcast.leftIndex, Broadcast.rightIndex]

def address_run (fp8 variant : Bool) (p : Pass) (s : State (WithBot ℝ)) :
    {t : State (WithBot ℝ) // run (engine model) ((pass fp8 variant p).surfaceBody.drop 16 |>.take 4)
      (math_run fp8 p s).val = some t} := by
  have hi := loop_registers fp8 p s
  refine ⟨addressState (math_run fp8 p s).val p, address_step fp8 variant p _ ?_ ?_ ?_⟩
  · simpa [math_run, mathState] using hi.2.2.1
  · simpa [math_run, mathState] using hi.2.2.2.1
  · simpa [math_run, mathState] using hi.2.2.2.2.1

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RmsMatmulRbe.NumericExecution
