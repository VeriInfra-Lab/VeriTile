import bench.tritonbench_g.fused_recurrent_retention.Proofs.QuotedReverseAddresses

namespace VeriTile.Bench.Repairs.FusedRecurrentRetention.Quoted
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
set_option maxHeartbeats 18000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false

private theorem update_congr {α : Type} {s t : State α} {n : RegName}
    {d : TileDType} {sh : TileShape} {v w : Values α d sh}
    (same : s=t) (values : v=w) : s.setReg n d sh v=t.setReg n d sh w := by
  cases same
  cases values
  rfl

theorem reverse_body_values {α : Type} [Inhabited α] (a : Arithmetic α)
    (s : Shape) (b : InputBuffers) (plans : Plans) (seed final : Bool)
    (initial current : State (Staged.Node α)) (pid : ProgramId) (r : Nat)
    (hr : r<s.T) (hBK : 0<s.BK) (hBV : 0<s.BV)
    (coveredK : plans.dk.Covers s.BV) (coveredV : plans.dv.Covers s.BK)
    (h : BackwardSecondInvariant a s b initial final pid r current) :
    run (Staged.algebra a plans)
      (Program.backwardSecond (layout s b) backwardRegions seed final).body
      (current.setReg "_t" .nat [] (fun _ => r)) =
      some (reverseBodyResult a s b plans initial current final pid r) := by
  have htime : s.T-1-(s.T-1-r)=r := by omega
  have hk := reverseKey_at (layout s b) pid r
  have hv := reverseValue_at (layout s b) pid r
  have hdo := reverseDo_at (layout s b) pid r
  have hdk := reverseGradient_at (layout s b) pid r
  have hdv := reverseOutput_at (layout s b) pid r
  simp only [layout] at hk hv hdo hdk hdv
  simp [Program.backwardSecond,Program.backwardFirst,Program.backward,Surface.backward,
    ComputeKernel.surfaceBody,Source.dynamicRange?,run,step,FP.Structural.evalExpr,
    evalComputeOp,evalOp_unfold,Option.bind,ComputeDType.eraseDType,numeric,
    FP.Structural.store,Region.cast,layout,backwardRegions,Staged.algebra,
    h.scale,h.beta,h.q,h.k,h.v,h.dout,h.dk,h.dv,h.maskK,h.maskV,h.carry,
    h.qmem,h.kmem,h.vmem,h.domem]
  dsimp only [ofFloat,toFloat]
  simp only [id_eq,Function.comp_apply]
  unfold FP.Structural.bop
  dsimp only [Broadcast.leftIndex,Broadcast.rightIndex]
  try simp only [reverseKey_step,reverseValue_step,reverseDo_step,reverseGradient_step,reverseOutput_step]
  try simp only [hk _ hr,hv _ hr,hdo _ hr,hdk _ hr,hdv _ hr]
  try simp only [literal_zero,node_ite_value,atom_value,tagged_value,product_value,
    Staged.scaled_query,reverse_add_tagged,Staged.reverse_product,Staged.reverse_decay,reverseNode_value]
  simp [reverseBodyResult,reverseBeforeStores,reverseScatter,reverseNode,programReverseCarry,foldN,
    programAdjoint,htime,programK,programV,keyLoad,valueLoad,keyOffset,valueOffset,doOffset,
    gradientOffset,outputOffset,inputs,readWord,ProgramId.key,ProgramId.value,layout,
    Staged.atom,Staged.tagged,Nat.add_assoc,Nat.mul_add]
  have hredK := Staged.key_reduction a plans s.BK s.BV hBV coveredK
    (fun k v => programAdjoint a s (inputs (layout s b) initial) final pid (s.T-1-r) k v)
    (fun v => valueLoad a s (inputs (layout s b) initial).v pid.head (s.T-1-r) (pid.value s v))
  have hredV := Staged.value_reduction a plans s.BK s.BV hBK coveredV
    (fun k v => programAdjoint a s (inputs (layout s b) initial) final pid (s.T-1-r) k v)
    (fun k => keyLoad a s (inputs (layout s b) initial).k pid.head (s.T-1-r) (pid.key s k))
  simp [programAdjoint,htime,programReverseCarry,foldN,keyLoad,valueLoad,inputs,readWord,
    ProgramId.key,ProgramId.value,layout,Staged.atom,Staged.tagged,Nat.add_assoc,Nat.mul_add] at hredK hredV
  simp only [hredK,hredV]
  simp only [hdk _ hr,hdv _ hr]
  simp only [gradientOffset,outputOffset,Nat.add_assoc]
  repeat' apply update_congr
  all_goals first | rfl | (funext i; simp [Nat.add_assoc,Nat.mul_add,Nat.add_mul])

  all_goals rename_i i
  all_goals first
  | simpa [keyOffset,layout,Nat.add_assoc,Nat.mul_add,Nat.add_mul] using
      (Eq.trans (congrArg (fun n => n-s.DK) (reverseKey_at (layout s b) pid r i.1.val hr).symm)
        (reverseKey_step (layout s b) pid r i.1.val))
  | simpa [valueOffset,layout,Nat.add_assoc,Nat.mul_add,Nat.add_mul] using
      (Eq.trans (congrArg (fun n => n-s.DV) (reverseValue_at (layout s b) pid r i.1.val hr).symm)
        (reverseValue_step (layout s b) pid r i.1.val))
  | simpa [doOffset,layout,Nat.add_assoc,Nat.mul_add,Nat.add_mul] using
      (Eq.trans (congrArg (fun n => n-b.do_.strides 2) (reverseDo_at (layout s b) pid r i.1.val hr).symm)
        (reverseDo_step (layout s b) pid r i.1.val))
  | simpa [gradientOffset,layout,Nat.add_assoc,Nat.mul_add,Nat.add_mul] using
      (Eq.trans (congrArg (fun n => n-s.DK) (reverseGradient_at (layout s b) pid r i.1.val hr).symm)
        (reverseGradient_step (layout s b) pid r i.1.val))
  | simpa [outputOffset,layout,Nat.add_assoc,Nat.mul_add,Nat.add_mul] using
      (Eq.trans (congrArg (fun n => n-s.DV) (reverseOutput_at (layout s b) pid r i.1.val hr).symm)
        (reverseOutput_step (layout s b) pid r i.1.val))

#axiomsClean reverse_body_values
end
end VeriTile.Bench.Repairs.FusedRecurrentRetention.Quoted
