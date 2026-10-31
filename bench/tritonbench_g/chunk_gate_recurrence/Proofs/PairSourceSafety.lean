import bench.tritonbench_g.chunk_gate_recurrence.Proofs.PairSafety
import bench.tritonbench_g.chunk_gate_recurrence.Proofs.PairExecution

namespace VeriTile.Bench.Repairs.ChunkGateRecurrence.ProgramPair
open VeriTile Triton FP.Structural FP.ProfiledRewrite KernelProgram KernelCarrier
open FP.Structural.Safety
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

private theorem prefix_safe (isa : Instructions) (plan : Reduction.ProductPlan 1024) (swap : Bool)
    (kind : Kind) (c : Shape) (s : SeedStrides) (seed : Bool) (bounds : RegionName → Nat) (u : State Node) :
    traceSafeList (engine (algebra isa plan swap)) bounds (before kind c s seed) u ↔
      traceSafeList (engine (candidateAlgebra isa plan swap)) bounds (before kind c s seed) u := by
  cases kind <;> cases seed
  all_goals
    simp only [before,parts,source,Kernels.forward,Kernels.backward,ComputeKernel.surfaceBody,VeriTile.Bench.Repairs.ChunkGateRecurrence.Kernels.backwardFamily,VeriTile.Bench.Repairs.ChunkGateRecurrence.Kernels.backwardSumLeftFamily,
      Source.dynamicRange?,Option.map,Option.getD,String.reduceEq,beq_self_eq_true,↓reduceIte]
    simp (config := {maxSteps:=250000}) [traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,
      safeCompute_unfold,safeOp_unfold,safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
      run,FP.Structural.step,FP.Structural.store,evalExpr,evalComputeOp,evalOp_unfold,
      engine,Algebra.withDefaultPrecision,resolvePrecision,numeric,algebra,candidateAlgebra]

private theorem tail_before_safe (isa : Instructions) (plan : Reduction.ProductPlan 1024) (swap : Bool)
    (c : Shape) (s : SeedStrides) (seed : Bool) (bounds : RegionName → Nat) (u : State Node) :
    traceSafeList (engine (algebra isa plan swap)) bounds (tailBefore c s seed) u ↔
      traceSafeList (engine (candidateAlgebra isa plan swap)) bounds (tailBefore c s seed) u := by
  simp only [tailBefore,after,parts,source,Kernels.backward,ComputeKernel.surfaceBody,VeriTile.Bench.Repairs.ChunkGateRecurrence.Kernels.backwardFamily,VeriTile.Bench.Repairs.ChunkGateRecurrence.Kernels.backwardSumLeftFamily,
    Source.dynamicRange?,Source.prefixBefore,Option.map,Option.getD,String.reduceEq,beq_self_eq_true,↓reduceIte]
  simp [traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,safeOp_unfold,
    safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
    run,FP.Structural.step,FP.Structural.store,evalExpr,evalComputeOp,evalOp_unfold,
    engine,Algebra.withDefaultPrecision,resolvePrecision,numeric,algebra,candidateAlgebra]

private theorem tail_after_safe (isa : Instructions) (plan : Reduction.ProductPlan 1024) (swap : Bool)
    (c : Shape) (s : SeedStrides) (seed : Bool) (bounds : RegionName → Nat) (u : State Node) :
    traceSafeList (engine (algebra isa plan swap)) bounds (tailAfter c s seed) u ↔
      traceSafeList (engine (candidateAlgebra isa plan swap)) bounds (tailAfter c s seed) u := by
  simp only [tailAfter,tailBefore,after,parts,source,Kernels.backward,ComputeKernel.surfaceBody,VeriTile.Bench.Repairs.ChunkGateRecurrence.Kernels.backwardFamily,VeriTile.Bench.Repairs.ChunkGateRecurrence.Kernels.backwardSumLeftFamily,
    Source.dynamicRange?,Source.prefixBefore,Option.map,Option.getD,String.reduceEq,beq_self_eq_true,↓reduceIte]
  simp [traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,safeOp_unfold,
    safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
    run,FP.Structural.step,FP.Structural.store,evalExpr,evalComputeOp,evalOp_unfold,
    engine,Algebra.withDefaultPrecision,resolvePrecision,numeric,algebra,candidateAlgebra]

private theorem update_step (isa : Instructions) (plan : Reduction.ProductPlan 1024) (swap : Bool)
    (out left right : RegName) (u : State Node) :
    FP.Structural.step (engine (algebra isa plan swap)) (KernelOps.assign32 out (KernelOps.update left right)) u =
      FP.Structural.step (engine (candidateAlgebra isa plan swap)) (KernelOps.assign32 out (rightUpdate left right)) u := by
  simp only [FP.Structural.step,KernelOps.assign32,evalExpr,evalComputeOp]
  exact congrArg (fun (e : Option (Values Node .real [64,16])) =>
    e.bind (fun values => some (u.setReg out .real [64,16] values)))
    (ProgramPair.eval_update isa plan swap left right u)

private theorem after_safe (isa : Instructions) (plan : Reduction.ProductPlan 1024) (swap : Bool)
    (kind : Kind) (c : Shape) (s : SeedStrides) (seed : Bool) (bounds : RegionName → Nat) (u : State Node) :
    traceSafeList (engine (algebra isa plan swap)) bounds (after kind c s seed) u ↔
      traceSafeList (engine (candidateAlgebra isa plan swap)) bounds (candidateParts kind c s seed).after u := by
  cases kind with
  | forward =>
    change traceSafeList _ bounds [] u ↔ traceSafeList _ bounds [] u
    simp only [traceSafeList_nil]
  | backward =>
    have hp := funext (tail_before_run isa plan swap c s seed)
    simp only [tail_original,tail_candidate,List.append_assoc,List.cons_append,List.nil_append,
      traceSafeList_append,traceSafeList_cons,tail_before_safe,hp,update_step,tail_after_safe]
    simp [KernelOps.assign32,KernelOps.update,rightUpdate,KernelOps.matrix,KernelOps.scalar,
      stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,safeOp_unfold]

private theorem range_safe (isa : Instructions) (plan : Reduction.ProductPlan 1024) (swap : Bool)
    (kind : Kind) (c : Shape) (s : SeedStrides) (seed : Bool) (bounds : RegionName → Nat) (stop stride : Nat) :
    ∀ start u, rangeSafe (engine (algebra isa plan swap)) bounds "_i" start stop stride (body kind c s seed) u ↔
      rangeSafe (engine (candidateAlgebra isa plan swap)) bounds "_i" start stop stride
        (candidateParts kind c s seed).body u := by
  intro start
  induction start using (measure (fun i => stop-i)).wf.induction with
  | h cur ih =>
    intro u
    rw [rangeSafe_unfold,rangeSafe_unfold]
    by_cases hz : stride=0
    · simp [hz]
    by_cases hc : cur<stop
    · simp only [hz,hc,if_false,if_true,body_safe,body_run]
      apply and_congr Iff.rfl
      cases he : run (engine (candidateAlgebra isa plan swap))
        (candidateParts kind c s seed).body (u.setReg "_i" .nat [] (fun _ => cur)) with
      | none => rfl
      | some v => exact ih (cur+stride) (by simp_wf; omega) v
    · simp [hz,hc]

theorem complete_safe (isa : Instructions) (plan : Reduction.ProductPlan 1024) (swap : Bool)
    (kind : Kind) (c : Shape) (s : SeedStrides) (seed : Bool) (bounds : RegionName → Nat) (u : State Node) :
    TraceSafe (engine (algebra isa plan swap)) bounds (source kind c s seed) u ↔
      TraceSafe (engine (candidateAlgebra isa plan swap)) bounds (candidateSource kind c s seed) u := by
  let loopR := ComputeStmt.forRangeDyn "_i" (.constNat 0)
    (.sub .nat .nil (.constNat c.N) (.constNat 1)) (.constNat 1) (candidateParts kind c s seed).body
  have hp := funext (before_run isa plan swap kind c s seed)
  have he : ∀ v, FP.Structural.step (engine (algebra isa plan swap)) (loop kind c s seed) v =
      FP.Structural.step (engine (candidateAlgebra isa plan swap)) loopR v := by
    intro v
    simpa only [KernelProgram.loop,loopR,FP.Structural.step,FP.Structural.evalOp,Option.bind_some,bind]
      using range_run isa plan swap kind c s seed (c.N-1) 1 0 v
  have hs : ∀ v, stepSafe (engine (algebra isa plan swap)) bounds (loop kind c s seed) v ↔
      stepSafe (engine (candidateAlgebra isa plan swap)) bounds loopR v := by
    intro v
    simpa only [KernelProgram.loop,loopR,stepSafe_unfold,safeOp_unfold,FP.Structural.evalOp,
      and_self,Option.bind_some,bind,forall_eq',and_true,true_and] using
      range_safe isa plan swap kind c s seed bounds (c.N-1) 1 0 v
  unfold TraceSafe
  rw [source_body,candidate_source_body]
  change traceSafeList _ bounds (before kind c s seed ++ [loop kind c s seed] ++ after kind c s seed) u ↔
    traceSafeList _ bounds (before kind c s seed ++ [loopR] ++ (candidateParts kind c s seed).after) u
  simp only [List.append_assoc,List.cons_append,List.nil_append,traceSafeList_append,traceSafeList_cons,
    prefix_safe,hp,hs,he,after_safe]

end
end VeriTile.Bench.Repairs.ChunkGateRecurrence.ProgramPair
