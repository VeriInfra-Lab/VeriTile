import bench.tritonbench_g.bmm_chunk_fwd.NativeIO

namespace VeriTile.Bench.Repairs.BmmChunkFwd.BmmIO.Native
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.BmmChunkFwd
open KernelInterface NativeGraph
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

/-- The only changed instruction is the fourth, a pure natural-number multiply
inside the head index. Its evaluated prefix is identical; every later memory
access is therefore checked at the same actual state. -/
theorem candidate_safe {α : Type} [Inhabited α] (algebra : Algebra α)
    (h : Host.Call) (c : Host.Config) (bounds : RegionName → Nat) (s : State α)
    (safe : TraceSafe (engine algebra) bounds (Kernels.fp32Source h c) s) :
    TraceSafe (engine algebra) bounds (Kernels.indexCommuted h c) s := by
  have prefixSafe : Safety.traceSafeList (engine algebra) bounds
      ((Kernels.indexCommuted h c).surfaceBody.take 4) s := by
    simp (config := {maxSteps := 300000}) [Kernels.indexCommuted,Kernels.bindMain,Kernels.mainSource,Kernels.mainIndexCommuted,
      ComputeKernel.surfaceBody,Safety.traceSafeList,Safety.stepSafe,Safety.safeExpr,
      Safety.safeCompute,Safety.safeAlg,Safety.safeOp.eq_def,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,engine,
      Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
      numeric,bop,State.setReg,integralFloorDiv,integralMod]
  have prefixEq : run (engine algebra) ((Kernels.fp32Source h c).surfaceBody.take 4) s=
      run (engine algebra) ((Kernels.indexCommuted h c).surfaceBody.take 4) s := by
    simp (config := {maxSteps := 300000}) [Kernels.fp32Source,Kernels.indexCommuted,Kernels.bindMain,Kernels.mainSource,Kernels.mainIndexCommuted,
      ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,engine,
      Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
      numeric,bop,State.setReg,integralFloorDiv,integralMod,Nat.mul_comm]
    have comm (a b : Nat) : bop (fun x y : Nat=>x*y) Broadcast.nil (fun _ : TileIndex []=>a) (fun _=>b)=
        bop (fun x y : Nat=>x*y) Broadcast.nil (fun _ : TileIndex []=>b) (fun _=>a) := by
      funext i
      exact Nat.mul_comm a b
    simp only [show bop (fun x y : Nat=>x/y) Broadcast.nil (fun _ : TileIndex []=>s.pids 2) (fun _=>h.shape.groups)=
      (fun _ : TileIndex []=>s.pids 2/h.shape.groups) from rfl,comm (s.pids 2/h.shape.groups) h.shape.groups]
  have tailEq : (Kernels.fp32Source h c).surfaceBody.drop 4=
      (Kernels.indexCommuted h c).surfaceBody.drop 4 := rfl
  have old : Safety.traceSafeList (engine algebra) bounds
      ((Kernels.fp32Source h c).surfaceBody.take 4 ++ (Kernels.fp32Source h c).surfaceBody.drop 4) s := by
    simpa only [List.take_append_drop] using safe
  obtain ⟨_,suffix⟩ := (Safety.traceSafeList_append _ _ _ _ _).mp old
  change Safety.traceSafeList (engine algebra) bounds (Kernels.indexCommuted h c).surfaceBody s
  rw [←List.take_append_drop 4 (Kernels.indexCommuted h c).surfaceBody,Safety.traceSafeList_append]
  refine ⟨prefixSafe,?_⟩
  intro next executed
  rw [←tailEq]
  exact suffix next (prefixEq.trans executed)

end
end VeriTile.Bench.Repairs.BmmChunkFwd.BmmIO.Native
