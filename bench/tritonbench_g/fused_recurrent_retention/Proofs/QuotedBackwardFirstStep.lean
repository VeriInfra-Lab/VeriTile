import bench.tritonbench_g.fused_recurrent_retention.Proofs.QuotedBackwardState
import bench.tritonbench_g.fused_recurrent_retention.Proofs.QuotedForwardNumbers

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

theorem backward_first_values {α : Type} [Inhabited α] (a : Arithmetic α)
    (s : Shape) (b : InputBuffers) (plans : Plans) (seed final : Bool)
    (initial current : State (Staged.Node α)) (pid : ProgramId) (t : Nat)
    (hBV : 0<s.BV) (covered : plans.dq.Covers s.BV)
    (h : BackwardFirstInvariant a s b initial seed pid t current) :
    run (Staged.algebra a plans)
      (Program.backwardFirst (layout s b) backwardRegions seed final).body
      (current.setReg "i" .nat [] (fun _ => t)) =
      some (backwardFirstBodyResult a s b plans initial current seed pid t) := by
  simp [Program.backwardFirst,Program.backward,Surface.backward,
    ComputeKernel.surfaceBody,Source.dynamicRange?,run,step,FP.Structural.evalExpr,
    evalComputeOp,evalOp_unfold,Option.bind,ComputeDType.eraseDType,numeric,
    FP.Structural.store,Region.cast,layout,backwardRegions,Staged.algebra,
    h.scale,h.keyId,h.valueId,h.headId,h.localHead,h.beta,h.k,h.v,h.dout,h.dq,h.maskK,h.maskV,h.carry,
    h.qmem,h.kmem,h.vmem,h.domem]
  dsimp only [ofFloat,toFloat]
  simp only [id_eq,Function.comp_apply]
  unfold FP.Structural.bop
  dsimp only [Broadcast.leftIndex,Broadcast.rightIndex]
  try simp only [literal_zero,node_ite_value,atom_value,tagged_value,product_value,
    Staged.scaled_query,Staged.forward_step,Staged.query_product,carryNode_value]
  simp [backwardFirstBodyResult,backwardFirstBeforeStore,backwardFirstScatter,carryNode,programForwardCarry,foldN,
    programQ,keyLoad,valueLoad,keyOffset,valueOffset,doOffset,gradientOffset,inputs,readWord,
    ProgramId.key,ProgramId.value,layout,Staged.atom,Staged.tagged,Nat.add_assoc,Nat.mul_add]
  have hred := Staged.query_reduction a plans s.BK s.BV hBV covered
    (fun k v => programForwardCarry a s (inputs (layout s b) initial) seed pid (t+1) k v)
    (fun v => valueLoad a s (inputs (layout s b) initial).do_ pid.head t (pid.value s v))
  simp [programForwardCarry,foldN,keyLoad,valueLoad,inputs,readWord,ProgramId.key,ProgramId.value,
    layout,Staged.atom,Staged.tagged,Nat.add_assoc,Nat.mul_add] at hred
  simp only [hred]
  simp [Staged.multiply,Staged.atom,Staged.tagged]
  repeat' apply update_congr
  all_goals first | rfl | (funext i; simp [Nat.add_assoc,Nat.mul_add,Nat.add_mul])

#axiomsClean backward_first_values
end
end VeriTile.Bench.Repairs.FusedRecurrentRetention.Quoted
