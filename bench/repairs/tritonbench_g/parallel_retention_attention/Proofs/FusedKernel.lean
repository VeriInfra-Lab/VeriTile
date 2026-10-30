import bench.repairs.tritonbench_g.parallel_retention_attention.Proofs.FusedLocal

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention.Fused
open VeriTile Triton FP.Structural Kernel
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (ForwardConfig BackwardConfig DotPrimitive withDot)
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 35000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem split_code (xs : List ComputeStmt) (n k : Nat) :
    xs = xs.take n ++ (xs.drop n).take k ++ xs.drop (n+k) := by
  simp only [← List.drop_drop]
  rw [List.append_assoc,List.take_append_drop,List.take_append_drop]

theorem forward_body_equal (M : Algebra α) (D : DotPrimitive α) (c : ForwardConfig) :
    CodeEq (withDot M D) (loopBody (FPStages.forwardKernel c) 16) (loopBody (forward c) 16) := by
  rw [split_code (loopBody (FPStages.forwardKernel c) 16) 3 2,split_code (loopBody (forward c) 16) 3 1]
  apply CodeEq.append
  · apply CodeEq.append
    · have h : (loopBody (FPStages.forwardKernel c) 16).take 3=(loopBody (forward c) 16).take 3 := rfl
      rw [h]; exact CodeEq.refl _ _
    · exact ⟨forward_local_run M D c,forward_local_safe M D c⟩
  · have h : (loopBody (FPStages.forwardKernel c) 16).drop 5=(loopBody (forward c) 16).drop 4 := rfl
    change CodeEq _ ((loopBody (FPStages.forwardKernel c) 16).drop 5) ((loopBody (forward c) 16).drop 4)
    rw [h]; exact CodeEq.refl _ _

theorem forward_equal (M : Algebra α) (D : DotPrimitive α) (c : ForwardConfig) :
    CodeEq (withDot M D) (FPStages.forwardKernel c).surfaceBody (forward c).surfaceBody := by
  rw [split_code (FPStages.forwardKernel c).surfaceBody 16 1,split_code (forward c).surfaceBody 16 1]
  apply CodeEq.append
  · apply CodeEq.append
    · have h : (FPStages.forwardKernel c).surfaceBody.take 16=(forward c).surfaceBody.take 16 := rfl
      rw [h]; exact CodeEq.refl _ _
    · change CodeEq _ [.forRangeDyn "_i" _ _ _ (loopBody (FPStages.forwardKernel c) 16)]
        [.forRangeDyn "_i" _ _ _ (loopBody (forward c) 16)]
      exact (dynamic_context (forward_body_equal M D c) _ _ _ _).singleton
  · have h : (FPStages.forwardKernel c).surfaceBody.drop 17=(forward c).surfaceBody.drop 17 := rfl
    change CodeEq _ ((FPStages.forwardKernel c).surfaceBody.drop 17) ((forward c).surfaceBody.drop 17)
    rw [h]; exact CodeEq.refl _ _

theorem query_body_equal (M : Algebra α) (D : DotPrimitive α) (c : BackwardConfig) (dq : RegionName) :
    CodeEq (withDot M D) (loopBody (FPQuery.source c dq) 8) (loopBody (query c dq) 8) := by
  rw [split_code (loopBody (FPQuery.source c dq) 8) 3 2,split_code (loopBody (query c dq) 8) 3 1]
  apply CodeEq.append
  · apply CodeEq.append
    · have h : (loopBody (FPQuery.source c dq) 8).take 3=(loopBody (query c dq) 8).take 3 := rfl
      rw [h]; exact CodeEq.refl _ _
    · exact ⟨query_local_run M D c dq,query_local_safe M D c dq⟩
  · have h : (loopBody (FPQuery.source c dq) 8).drop 5=(loopBody (query c dq) 8).drop 4 := rfl
    change CodeEq _ ((loopBody (FPQuery.source c dq) 8).drop 5) ((loopBody (query c dq) 8).drop 4)
    rw [h]; exact CodeEq.refl _ _

theorem query_equal (M : Algebra α) (D : DotPrimitive α) (c : BackwardConfig) (dq : RegionName) :
    CodeEq (withDot M D) (FPQuery.source c dq).surfaceBody (query c dq).surfaceBody := by
  rw [split_code (FPQuery.source c dq).surfaceBody 8 1,split_code (query c dq).surfaceBody 8 1]
  apply CodeEq.append
  · apply CodeEq.append
    · have h : (FPQuery.source c dq).surfaceBody.take 8=(query c dq).surfaceBody.take 8 := rfl
      rw [h]; exact CodeEq.refl _ _
    · change CodeEq _ [.forRange "_i" _ _ _ (loopBody (FPQuery.source c dq) 8)]
        [.forRange "_i" _ _ _ (loopBody (query c dq) 8)]
      exact (static_context (query_body_equal M D c dq) _ _ _ _).singleton
  · have h : (FPQuery.source c dq).surfaceBody.drop 9=(query c dq).surfaceBody.drop 9 := rfl
    change CodeEq _ ((FPQuery.source c dq).surfaceBody.drop 9) ((query c dq).surfaceBody.drop 9)
    rw [h]; exact CodeEq.refl _ _

theorem shell_body (c : BackwardConfig) (dq : RegionName) :
    (FPBackward.shellSource c dq).surfaceBody=(FPQuery.source c dq).surfaceBody ++
      [.effectMarker "tl.debug_barrier"] ++ (FPStages.backwardKernel c).surfaceBody := rfl

theorem fusedShell_body (c : BackwardConfig) (dq : RegionName) :
    (FPBackward.fusedShellSource c dq).surfaceBody=(query c dq).surfaceBody ++
      [.effectMarker "tl.debug_barrier"] ++ (FPStages.backwardKernel c).surfaceBody := rfl

theorem shell_original (c : BackwardConfig) (dq : RegionName) :
    (FPBackward.shellSource c dq).surfaceBody=(FPBackward.source c dq).surfaceBody := rfl

theorem shell_equal (M : Algebra α) (D : DotPrimitive α) (c : BackwardConfig) (dq : RegionName) :
    CodeEq (withDot M D) (FPBackward.source c dq).surfaceBody (FPBackward.fusedShellSource c dq).surfaceBody := by
  rw [← shell_original,shell_body,fusedShell_body]
  exact ((query_equal M D c dq).append (CodeEq.refl _ _)).append (CodeEq.refl _ _)

theorem forward_run (M : Algebra α) (D : DotPrimitive α) (c : ForwardConfig) (s : State α) :
    FP.Structural.exec (withDot M D) (FPStages.forwardKernel c) s=FP.Structural.exec (withDot M D) (forward c) s := by
  cases h : FPStages.forwardKernel c; cases hc : forward c
  simpa only [h,hc,ComputeKernel.surfaceBody,FP.Structural.exec] using (forward_equal M D c).run s

theorem forward_safe (M : Algebra α) (D : DotPrimitive α) (c : ForwardConfig) (cap : RegionName → Nat) (s : State α) :
    TraceSafe (withDot M D) cap (FPStages.forwardKernel c) s ↔ TraceSafe (withDot M D) cap (forward c) s := (forward_equal M D c).safe cap s

theorem shell_run (M : Algebra α) (D : DotPrimitive α) (c : BackwardConfig) (dq : RegionName) (s : State α) :
    FP.Structural.exec (withDot M D) (FPBackward.source c dq) s=FP.Structural.exec (withDot M D) (FPBackward.fusedShellSource c dq) s := by
  cases h : FPBackward.source c dq; cases hc : FPBackward.fusedShellSource c dq
  simpa only [h,hc,ComputeKernel.surfaceBody,FP.Structural.exec] using (shell_equal M D c dq).run s

theorem shell_safe (M : Algebra α) (D : DotPrimitive α) (c : BackwardConfig) (dq : RegionName) (cap : RegionName → Nat) (s : State α) :
    TraceSafe (withDot M D) cap (FPBackward.source c dq) s ↔ TraceSafe (withDot M D) cap (FPBackward.fusedShellSource c dq) s := (shell_equal M D c dq).safe cap s

#axiomsClean forward_run
#axiomsClean shell_run
end
end VeriTile.Bench.Repairs.ParallelRetentionAttention.Fused
