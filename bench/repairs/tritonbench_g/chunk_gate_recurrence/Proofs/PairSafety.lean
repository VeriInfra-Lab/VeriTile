import bench.repairs.tritonbench_g.chunk_gate_recurrence.Proofs.ProgramPair

namespace VeriTile.Bench.Repairs.ChunkGateRecurrence.ProgramPair
open VeriTile Triton FP.Structural FP.ProfiledRewrite KernelProgram KernelCarrier
open FP.Structural.Safety
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

private theorem body_before_safe (isa : Instructions) (plan : Reduction.ProductPlan 1024) (swap : Bool)
    (kind : Kind) (c : Shape) (bounds : RegionName → Nat) (u : State Node) :
    traceSafeList (engine (algebra isa plan swap)) bounds (bodyBefore kind c) u ↔
      traceSafeList (engine (candidateAlgebra isa plan swap)) bounds (bodyBefore kind c) u := by
  cases kind <;> simp [bodyBefore,carry,Source.prefixBefore,KernelOps.forwardBody,KernelOps.backwardBody,
    KernelOps.loadScalar,KernelOps.loadMatrix,KernelOps.assign32,KernelOps.matrixPtr,KernelOps.scalarPtr,
    traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,safeOp_unfold,
    safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
    run,FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,
    engine,Algebra.withDefaultPrecision,resolvePrecision,numeric,algebra,candidateAlgebra]

private theorem body_after_safe (isa : Instructions) (plan : Reduction.ProductPlan 1024) (swap : Bool)
    (kind : Kind) (c : Shape) (bounds : RegionName → Nat) (u : State Node) :
    traceSafeList (engine (algebra isa plan swap)) bounds (bodyAfter kind c) u ↔
      traceSafeList (engine (candidateAlgebra isa plan swap)) bounds (bodyAfter kind c) u := by
  cases kind <;> simp [bodyBefore,bodyAfter,carry,Source.prefixBefore,KernelOps.forwardBody,KernelOps.backwardBody,
    KernelOps.loadScalar,KernelOps.loadMatrix,KernelOps.assign32,KernelOps.storeMatrix,KernelOps.storeScalar,
    KernelOps.advance,KernelOps.retreat,KernelOps.gradient,KernelOps.matrixPtr,KernelOps.scalarPtr,
    KernelOps.matrix,KernelOps.scalar,KernelOps.stride,KernelOps.tileStride,
    traceSafeList_unfold,stepSafe_unfold,safeExpr_unfold,safeCompute_unfold,safeOp_unfold,
    safeAddress_unfold,safeMask_unfold,addressSafe_unfold,maskActive_unfold,
    run,FP.Structural.step,FP.Structural.store,evalExpr,evalComputeOp,evalOp_unfold,
    engine,Algebra.withDefaultPrecision,resolvePrecision,numeric,algebra,candidateAlgebra]

private theorem assign_step (isa : Instructions) (plan : Reduction.ProductPlan 1024) (swap : Bool)
    (out left right : RegName) (u : State Node) :
    FP.Structural.step (engine (algebra isa plan swap)) (KernelOps.assign32 out (KernelOps.update left right)) u =
      FP.Structural.step (engine (candidateAlgebra isa plan swap)) (KernelOps.assign32 out (rightUpdate left right)) u := by
  simp only [FP.Structural.step,KernelOps.assign32,evalExpr,evalComputeOp]
  exact congrArg (fun (e : Option (Values Node .real [64,16])) =>
    e.bind (fun values => some (u.setReg out .real [64,16] values)))
    (ProgramPair.eval_update isa plan swap left right u)

private theorem assign_safe (isa : Instructions) (plan : Reduction.ProductPlan 1024) (swap : Bool)
    (out left right : RegName) (bounds : RegionName → Nat) (u : State Node) :
    stepSafe (engine (algebra isa plan swap)) bounds (KernelOps.assign32 out (KernelOps.update left right)) u ↔
      stepSafe (engine (candidateAlgebra isa plan swap)) bounds (KernelOps.assign32 out (rightUpdate left right)) u := by
  simp [stepSafe_unfold,KernelOps.assign32,KernelOps.update,rightUpdate,KernelOps.matrix,KernelOps.scalar,
    safeExpr_unfold,safeCompute_unfold,safeOp_unfold]

theorem body_safe (isa : Instructions) (plan : Reduction.ProductPlan 1024) (swap : Bool)
    (kind : Kind) (c : Shape) (s : SeedStrides) (seed : Bool) (bounds : RegionName → Nat) (u : State Node) :
    traceSafeList (engine (algebra isa plan swap)) bounds (body kind c s seed) u ↔
      traceSafeList (engine (candidateAlgebra isa plan swap)) bounds (candidateParts kind c s seed).body u := by
  have hp := funext (body_before_run isa plan swap kind c)
  simp only [body_original,body_candidate,List.append_assoc,List.cons_append,List.nil_append,
    traceSafeList_append,traceSafeList_cons,body_before_safe,hp,assign_safe,assign_step,body_after_safe]

end
end VeriTile.Bench.Repairs.ChunkGateRecurrence.ProgramPair
