import bench.tritonbench_g.fused_recurrent_retention.Proofs.QuotedBackwardState

namespace VeriTile.Bench.Repairs.FusedRecurrentRetention.Quoted
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

theorem backward_prefix_values {α : Type} [Inhabited α] (a : Arithmetic α)
    (s : Shape) (b : InputBuffers) (plans : Plans) (seed final : Bool)
    (initial : State (Staged.Node α))
    (hscale : initial.regs .real [] "scale" =
      some (fun _ => Staged.tagged .scale (a.scale s.DK))) :
    ∃ entry, run (Staged.algebra a plans)
        (Program.backwardFirst (layout s b) backwardRegions seed final).before initial = some entry ∧
      BackwardFirstInvariant a s b initial seed (launch initial) 0 entry ∧ entry.mem=initial.mem := by
  cases seed <;>
    simp [Program.backwardFirst,Program.backward,Surface.backward,
      ComputeKernel.surfaceBody,Source.dynamicRange?,run,step,FP.Structural.evalExpr,
      evalComputeOp,evalOp_unfold,Option.bind,ComputeDType.eraseDType,numeric,bop,numericLt,
      natLt,integralFloorDiv,integralMod,Region.cast,hscale,BackwardFirstInvariant,layout,backwardRegions,
      launch,keyOffset,valueOffset,doOffset,gradientOffset,seedOffset,ProgramId.key,ProgramId.value,
      carryNode,programForwardCarry,foldN,stateLoad,inputs,readWord,
      Staged.algebra,Staged.literal,Staged.multiply,Staged.add,Staged.subtract,Staged.tagged,
      Staged.atom,Staged.product,decay,Broadcast.leftIndex,Broadcast.rightIndex]

  all_goals constructor <;>
    simp [layout,launch,keyOffset,valueOffset,doOffset,gradientOffset,seedOffset,ProgramId.key,ProgramId.value,
      carryNode,programForwardCarry,foldN,stateLoad,inputs,readWord,
      Staged.algebra,Staged.literal,Staged.multiply,Staged.add,Staged.subtract,Staged.tagged,
      Staged.atom,Staged.product,decay,FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,
      integralFloorDiv,integralMod,natLt,hscale,Nat.add_assoc]

  all_goals funext i
  all_goals norm_num [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,
    Staged.atom,Staged.tagged,Staged.product,Staged.multiply,Staged.add,Staged.subtract,
    integralFloorDiv,integralMod,natLt,ofFloat,toFloat,Nat.add_assoc,Function.comp_def]
  all_goals first | rfl | (split_ifs <;> simp_all [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,
    ofFloat,toFloat,Staged.atom,Staged.tagged,Staged.product,Staged.add,Nat.add_assoc])

#axiomsClean backward_prefix_values
end
end VeriTile.Bench.Repairs.FusedRecurrentRetention.Quoted
