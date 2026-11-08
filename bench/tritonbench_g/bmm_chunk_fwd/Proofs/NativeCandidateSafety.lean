import bench.tritonbench_g.bmm_chunk_fwd.Proofs.Support.NativeIO

namespace VeriTile.Bench.Repairs.BmmChunkFwd.BmmIO.Native
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.BmmChunkFwd
open KernelInterface NativeGraph
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

/-- The only changed instruction is the twenty-third, a pure signed-integer multiply
inside the head index. Its evaluated prefix is identical; every later memory
access is therefore checked at the same actual state. -/
theorem candidate_safe {α : Type} [Inhabited α] (algebra : Algebra α)
    (h : Host.Call) (c : Host.Config) (bounds : RegionName → Nat) (s : State α)
    (safe : TraceSafe (engine algebra) bounds (Kernels.fp32Source h c) s) :
    TraceSafe (engine algebra) bounds (Kernels.indexCommutedKernel h c) s := by
  have beforeSafe : Safety.traceSafeList (engine algebra) bounds
      ((Kernels.fp32Source h c).surfaceBody.take 22) s := by
    have old : Safety.traceSafeList (engine algebra) bounds
        ((Kernels.fp32Source h c).surfaceBody.take 22 ++ (Kernels.fp32Source h c).surfaceBody.drop 22) s := by
      simpa only [List.take_append_drop] using safe
    exact ((Safety.traceSafeList_append _ _ _ _ _).mp old).1
  have prefixSafe : Safety.traceSafeList (engine algebra) bounds
      ((Kernels.indexCommutedKernel h c).surfaceBody.take 23) s := by
    change Safety.traceSafeList (engine algebra) bounds
      ((Kernels.fp32Source h c).surfaceBody.take 22 ++
        [.assign .int [] "pid_h" (.alg (.sub .int .nil (.ref .int [] "pid_ch")
          (.mul .int .nil (.castNatToInt (.constNat h.shape.groups)) (.ref .int [] "pid_c"))))]) s
    rw [Safety.traceSafeList_append]
    refine ⟨beforeSafe,?_⟩
    intro t _
    simp [Safety.traceSafeList,Safety.stepSafe,Safety.safeExpr,Safety.safeAlg,Safety.safeOp.eq_def]
    split <;> trivial
  have prefixEq : run (engine algebra) ((Kernels.fp32Source h c).surfaceBody.take 23) s=
      run (engine algebra) ((Kernels.indexCommutedKernel h c).surfaceBody.take 23) s := by
    change Contextual algebra s ((Kernels.fp32Source h c).surfaceBody.take 22) []
      (.assign .int [] "pid_h" (.alg (.sub .int .nil (.ref .int [] "pid_ch")
        (.mul .int .nil (.ref .int [] "pid_c") (.castNatToInt (.constNat h.shape.groups))))))
      (.assign .int [] "pid_h" (.alg (.sub .int .nil (.ref .int [] "pid_ch")
        (.mul .int .nil (.castNatToInt (.constNat h.shape.groups)) (.ref .int [] "pid_c")))))
    solve_fp
  have tailEq : (Kernels.fp32Source h c).surfaceBody.drop 23=
      (Kernels.indexCommutedKernel h c).surfaceBody.drop 23 := rfl
  have old : Safety.traceSafeList (engine algebra) bounds
      ((Kernels.fp32Source h c).surfaceBody.take 23 ++ (Kernels.fp32Source h c).surfaceBody.drop 23) s := by
    simpa only [List.take_append_drop] using safe
  obtain ⟨_,suffix⟩ := (Safety.traceSafeList_append _ _ _ _ _).mp old
  change Safety.traceSafeList (engine algebra) bounds (Kernels.indexCommutedKernel h c).surfaceBody s
  rw [←List.take_append_drop 23 (Kernels.indexCommutedKernel h c).surfaceBody,Safety.traceSafeList_append]
  refine ⟨prefixSafe,?_⟩
  intro next executed
  rw [←tailEq]
  exact suffix next (prefixEq.trans executed)

end
end VeriTile.Bench.Repairs.BmmChunkFwd.BmmIO.Native
