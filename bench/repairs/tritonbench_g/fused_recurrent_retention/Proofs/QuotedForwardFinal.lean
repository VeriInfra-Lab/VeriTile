import bench.repairs.tritonbench_g.fused_recurrent_retention.Proofs.QuotedForwardState

namespace VeriTile.Bench.Repairs.FusedRecurrentRetention.Quoted
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
set_option maxHeartbeats 7000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

def forwardFinalResult [Inhabited α] (a : Arithmetic α) (s : Shape) (b : InputBuffers)
    (initial u : State (Staged.Node α)) (seed final : Bool) (pid : ProgramId) : State (Staged.Node α) :=
  if final then
    let u := u.setReg "p_final_s" .ptr [s.BV,s.BK] (fun i : TileIndex [s.BV,s.BK] =>
      (("final_state" : RegionName),seedOffset (layout s b) pid i.2.1.val i.1.val))
    (TileShape.allIndices [s.BV,s.BK]).foldl (fun v i =>
      if pid.key s i.2.1.val<s.DK ∧ pid.value s i.1.val<s.DV then
        v.write "final_state" (seedOffset (layout s b) pid i.2.1.val i.1.val)
          (.mk .real (Staged.atom (programForwardCarry a s (inputs (layout s b) initial)
            seed pid s.T i.2.1.val i.1.val))) else v) u
  else u

theorem forward_final_values {α : Type} [Inhabited α] (a : Arithmetic α)
    (s : Shape) (b : InputBuffers) (plans : Plans) (seed final : Bool)
    (initial current : State (Staged.Node α)) (pid : ProgramId)
    (h : ForwardInvariant a s b initial seed pid s.T current) :
    run (Staged.algebra a plans)
      (Program.forwardParts (layout s b) forwardRegions seed final).after current =
      some (forwardFinalResult a s b initial current seed final pid) := by
  cases final <;>
    simp [Program.forwardParts,Program.forward,Surface.forward,
      ComputeKernel.surfaceBody,Source.dynamicRange?,run,step,FP.Structural.evalExpr,
      evalComputeOp,evalOp_unfold,Option.bind,ComputeDType.eraseDType,numeric,
      FP.Structural.store,Region.cast,layout,forwardRegions,Staged.algebra,
      h.keyId,h.valueId,h.headId,h.maskKV,h.carry,forwardFinalResult,
      seedOffset,ProgramId.key,ProgramId.value]
  unfold FP.Structural.bop
  dsimp only [Broadcast.leftIndex,Broadcast.rightIndex,ofFloat,toFloat]
  simp [carryNode,Staged.atom,Staged.tagged,Nat.add_assoc]

#axiomsClean forward_final_values
end
end VeriTile.Bench.Repairs.FusedRecurrentRetention.Quoted
