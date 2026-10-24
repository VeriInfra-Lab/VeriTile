import bench.tritonbench_g.rms_matmul_rbe.Proofs.NumericAddress

namespace VeriTile.Bench.TritonBenchG.RmsMatmulRbe.NumericExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
open scoped Classical
set_option maxHeartbeats 4000000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false


def outputCell (fp8 : Bool) (p : Pass) : Cell (WithBot ℝ) :=
  match p with
  | .q => .mk .fp16 (some 2)
  | .k => if fp8 then .mk .f8e5 (some 3) else .mk .fp16 (some 3)
  | .v => if fp8 then .mk .f8e5 (some 4) else .mk .fp16 (some 4)

theorem store_step (fp8 variant : Bool) (p : Pass) (t : State (WithBot ℝ))
    (hp : t.regs .ptr [1, 2] "out_ptrs" = some (fun i => (outRegion p, i.2.1.val * 2)))
    (hm : t.regs .bool [1, 2] "out_mask" = some (fun i => decide (i.2.1.val < 1)))
    (ha : t.regs .real [1, 2] "accumulator" = some (fun _ => some (value p))) :
    run (engine model) ((pass fp8 variant p).surfaceBody.drop 20) t =
      some (t.write (outRegion p) 0 (outputCell fp8 p)) := by
  cases fp8 <;> cases variant <;> cases p
  all_goals simp (maxSteps := 1000000) [outputCell, outRegion, value, hp, hm, ha, pass, TypedSource.kernel, ComputeKernel.surfaceBody, run, step,
    FP.Structural.evalExpr, evalComputeOp, evalOp_unfold, evalOffsets, evalDeltas,
    Option.bind, engine, Algebra.withDefaultPrecision, resolvePrecision, model,
    ComputeDType.eraseDType, numeric, integralFloorDiv, integralMod, bop, natLt,
    Cell.read, ofFloat, toFloat, TileShape.axisDim, Region.cast,
    FP.Structural.store, TileShape.allIndices, List.finRange_succ, Tile.select, Tile.ptrAdd,
    Broadcast.leftIndex, Broadcast.rightIndex, Tile.scalar, -Tile.scalar_eta,
    Float8E5Bits.isInt8, Float8E5Bits.ofInt]

def store_run (fp8 variant : Bool) (p : Pass) (s : State (WithBot ℝ)) :
    {t : State (WithBot ℝ) // run (engine model) ((pass fp8 variant p).surfaceBody.drop 20)
      (address_run fp8 variant p s).val = some t} := by
  refine ⟨(address_run fp8 variant p s).val.write (outRegion p) 0 (outputCell fp8 p),
    store_step fp8 variant p _ ?_ ?_ ?_⟩
  · simp [address_run, addressState]
  · simp [address_run, addressState]
  · simp [address_run, addressState, math_run, mathState]
    rfl

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RmsMatmulRbe.NumericExecution
