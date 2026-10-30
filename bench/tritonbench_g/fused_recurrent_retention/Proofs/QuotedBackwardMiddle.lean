import bench.tritonbench_g.fused_recurrent_retention.Proofs.QuotedReverseState

namespace VeriTile.Bench.Repairs.FusedRecurrentRetention.Quoted
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
set_option maxHeartbeats 10000000
set_option maxRecDepth 28000
set_option linter.unusedSimpArgs false

theorem backward_middle_values {α : Type} [Inhabited α] (a : Arithmetic α)
    (s : Shape) (b : InputBuffers) (plans : Plans) (seed final : Bool)
    (initial current : State (Staged.Node α)) (pid : ProgramId)
    (h : BackwardFirstInvariant a s b initial seed pid s.T current) :
    ∃ entry, run (Staged.algebra a plans)
        (Program.backwardSecond (layout s b) backwardRegions seed final).before current = some entry ∧
      BackwardSecondInvariant a s b initial final pid 0 entry ∧ entry.mem=current.mem := by
  cases final <;>
    simp [Program.backwardSecond,Program.backwardFirst,Program.backward,Surface.backward,
      ComputeKernel.surfaceBody,Source.dynamicRange?,run,step,FP.Structural.evalExpr,
      evalComputeOp,evalOp_unfold,Option.bind,ComputeDType.eraseDType,numeric,bop,numericLt,
      natLt,integralFloorDiv,integralMod,Region.cast,layout,backwardRegions,
      h.scale,h.keyId,h.valueId,h.headId,h.localHead,h.beta,h.maskK,h.maskV,
      h.qmem,h.kmem,h.vmem,h.domem,h.dfmem,
      Staged.algebra,Staged.literal,Staged.multiply,Staged.add,Staged.subtract,Staged.tagged,
      Staged.atom,Staged.product,Broadcast.leftIndex,Broadcast.rightIndex]
  all_goals constructor <;>
    simp [layout,keyOffset,valueOffset,doOffset,gradientOffset,outputOffset,ProgramId.key,ProgramId.value,
      reverseKeyOffset,reverseValueOffset,reverseDoOffset,reverseGradientOffset,reverseOutputOffset,
      reverseNode,programReverseCarry,foldN,stateLoad,inputs,readWord,
      Staged.algebra,Staged.literal,Staged.tagged,Staged.atom,
      FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,integralFloorDiv,integralMod,
      h.scale,h.beta,h.maskK,h.maskV,h.qmem,h.kmem,h.vmem,h.domem,h.dfmem,Nat.add_assoc]
  all_goals funext i
  all_goals norm_num [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,
    Staged.atom,Staged.tagged,integralFloorDiv,integralMod,natLt,ofFloat,toFloat,Nat.add_assoc,Function.comp_def]
  all_goals first | rfl | (split_ifs <;> simp_all [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,
    ofFloat,toFloat,Staged.atom,Staged.tagged,Nat.add_assoc])

#axiomsClean backward_middle_values
end
end VeriTile.Bench.Repairs.FusedRecurrentRetention.Quoted
