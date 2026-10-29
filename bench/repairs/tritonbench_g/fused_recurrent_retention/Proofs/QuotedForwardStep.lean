import bench.repairs.tritonbench_g.fused_recurrent_retention.Proofs.QuotedForwardState
import bench.repairs.tritonbench_g.fused_recurrent_retention.Proofs.QuotedForwardNumbers

namespace VeriTile.Bench.Repairs.FusedRecurrentRetention.Quoted
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 28000
set_option linter.unusedSimpArgs false

private theorem update_congr {α : Type} {s t : State α} {n : RegName}
    {d : TileDType} {sh : TileShape} {v w : Values α d sh}
    (same : s=t) (values : v=w) : s.setReg n d sh v=t.setReg n d sh w := by
  cases same
  cases values
  rfl

theorem forward_body_values {α : Type} [Inhabited α] (a : Arithmetic α)
    (s : Shape) (b : InputBuffers) (plans : Plans) (seed final : Bool)
    (initial current : State (Staged.Node α)) (pid : ProgramId) (t : Nat)
    (hBK : 0<s.BK) (covered : plans.output.Covers s.BK)
    (h : ForwardInvariant a s b initial seed pid t current) :
    run (Staged.algebra a plans)
      (Program.forwardParts (layout s b) forwardRegions seed final).body
      (current.setReg "_t" .nat [] (fun _ => t)) =
      some (forwardBodyResult a s b plans initial current seed pid t) := by
  simp [Program.forwardParts,Program.forward,Surface.forward,
    ComputeKernel.surfaceBody,Source.dynamicRange?,run,step,FP.Structural.evalExpr,
    evalComputeOp,evalOp_unfold,Option.bind,ComputeDType.eraseDType,numeric,
    FP.Structural.store,Region.cast,layout,forwardRegions,Staged.algebra,
    h.scale,h.keyId,h.valueId,h.headId,h.beta,h.q,h.k,h.v,h.o,h.maskK,h.maskV,h.maskKV,h.carry,
    h.qmem,h.kmem,h.vmem]
  dsimp only [ofFloat,toFloat]
  simp only [id_eq,Function.comp_apply]
  unfold FP.Structural.bop
  dsimp only [Broadcast.leftIndex,Broadcast.rightIndex]
  try simp only [literal_zero,node_ite_value,atom_value,tagged_value,product_value,
    Staged.scaled_query,Staged.forward_step,Staged.forward_product,carryNode_value]
  simp [forwardBodyResult,forwardBeforeStore,forwardScatter,carryNode,programForwardCarry,foldN,
    programO,keyLoad,valueLoad,keyOffset,valueOffset,outputOffset,inputs,readWord,
    ProgramId.key,ProgramId.value,layout,Staged.atom,Staged.tagged,Nat.add_assoc,Nat.mul_add]
  have hred := Staged.output_reduction a plans s.BK s.BV hBK covered
    (fun v k => programForwardCarry a s (inputs (layout s b) initial) seed pid (t+1) k v)
    (fun k => a.mul (keyLoad a s (inputs (layout s b) initial).q pid.head t (pid.key s k)) (a.scale s.DK))
  simp [programForwardCarry,foldN,keyLoad,valueLoad,inputs,readWord,ProgramId.key,ProgramId.value,
    layout,Staged.atom,Staged.tagged,Nat.add_assoc,Nat.mul_add] at hred
  simp only [hred]
  repeat' apply update_congr
  all_goals first | rfl | (funext i; simp [Nat.add_assoc,Nat.mul_add,Nat.add_mul])

#axiomsClean forward_body_values
end
end VeriTile.Bench.Repairs.FusedRecurrentRetention.Quoted
