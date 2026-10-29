import bench.repairs.tritonbench_g.parallel_attention.Proofs.FusedLocal

namespace VeriTile.Bench.Repairs.ParallelAttentionKernel.Fused
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.ParallelAttention
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 35000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem split_code (xs : List ComputeStmt) (n k : Nat) :
    xs = xs.take n ++ (xs.drop n).take k ++ xs.drop (n+k) := by
  simp only [← List.drop_drop]
  rw [List.append_assoc,List.take_append_drop,List.take_append_drop]

theorem forward_body_equal (M : Algebra α) (c : ForwardConfig) :
    CodeEq M (loopBody (forwardKernel c) 19) (loopBody (forward c) 19) := by
  rw [split_code (loopBody (forwardKernel c) 19) 4 2,
    split_code (loopBody (forward c) 19) 4 1]
  apply CodeEq.append
  · apply CodeEq.append
    · have h : (loopBody (forwardKernel c) 19).take 4=(loopBody (forward c) 19).take 4 := rfl
      rw [h]; exact CodeEq.refl M _
    · exact ⟨forward_local_run M c,forward_local_safe M c⟩
  · have h : (loopBody (forwardKernel c) 19).drop 6=(loopBody (forward c) 19).drop 5 := rfl
    change CodeEq M ((loopBody (forwardKernel c) 19).drop 6) ((loopBody (forward c) 19).drop 5)
    rw [h]; exact CodeEq.refl M _

theorem forward_equal (M : Algebra α) (c : ForwardConfig) :
    CodeEq M (forwardKernel c).surfaceBody (forward c).surfaceBody := by
  rw [split_code (forwardKernel c).surfaceBody 19 1,split_code (forward c).surfaceBody 19 1]
  apply CodeEq.append
  · apply CodeEq.append
    · have h : (forwardKernel c).surfaceBody.take 19=(forward c).surfaceBody.take 19 := rfl
      rw [h]; exact CodeEq.refl M _
    · change CodeEq M [.forRangeDyn "_i" _ _ _ (loopBody (forwardKernel c) 19)]
        [.forRangeDyn "_i" _ _ _ (loopBody (forward c) 19)]
      exact (dynamic_context (forward_body_equal M c) _ _ _ _).singleton
  · have h : (forwardKernel c).surfaceBody.drop 20=(forward c).surfaceBody.drop 20 := rfl
    change CodeEq M ((forwardKernel c).surfaceBody.drop 20) ((forward c).surfaceBody.drop 20)
    rw [h]; exact CodeEq.refl M _

theorem forward_run (M : Algebra α) (c : ForwardConfig) (s : State α) :
    FP.Structural.exec M (forwardKernel c) s=FP.Structural.exec M (forward c) s := by
  cases h : forwardKernel c; cases hc : forward c
  simpa only [h,hc,ComputeKernel.surfaceBody,FP.Structural.exec] using (forward_equal M c).run s

theorem forward_safe (M : Algebra α) (c : ForwardConfig) (cap : RegionName → Nat) (s : State α) :
    TraceSafe M cap (forwardKernel c) s ↔ TraceSafe M cap (forward c) s := (forward_equal M c).safe cap s

theorem query_body_equal (M : Algebra α) (D : DotPrimitive α) (c : QueryConfig) :
    CodeEq (withDot M D) (loopBody (queryKernel c) 16) (loopBody (query c) 16) := by
  rw [split_code (loopBody (queryKernel c) 16) 6 2,
    split_code (loopBody (query c) 16) 6 1]
  apply CodeEq.append
  · apply CodeEq.append
    · have h : (loopBody (queryKernel c) 16).take 6=(loopBody (query c) 16).take 6 := rfl
      rw [h]; exact CodeEq.refl (withDot M D) _
    · exact ⟨query_local_run M D c,query_local_safe M D c⟩
  · have h : (loopBody (queryKernel c) 16).drop 8=(loopBody (query c) 16).drop 7 := rfl
    change CodeEq (withDot M D) ((loopBody (queryKernel c) 16).drop 8) ((loopBody (query c) 16).drop 7)
    rw [h]; exact CodeEq.refl (withDot M D) _

theorem query_equal (M : Algebra α) (D : DotPrimitive α) (c : QueryConfig) :
    CodeEq (withDot M D) (queryKernel c).surfaceBody (query c).surfaceBody := by
  rw [split_code (queryKernel c).surfaceBody 16 1,split_code (query c).surfaceBody 16 1]
  apply CodeEq.append
  · apply CodeEq.append
    · have h : (queryKernel c).surfaceBody.take 16=(query c).surfaceBody.take 16 := rfl
      rw [h]; exact CodeEq.refl (withDot M D) _
    · change CodeEq (withDot M D) [.forRangeDyn "_i" _ _ _ (loopBody (queryKernel c) 16)]
        [.forRangeDyn "_i" _ _ _ (loopBody (query c) 16)]
      exact (dynamic_context (query_body_equal M D c) _ _ _ _).singleton
  · have h : (queryKernel c).surfaceBody.drop 17=(query c).surfaceBody.drop 17 := rfl
    change CodeEq (withDot M D) ((queryKernel c).surfaceBody.drop 17) ((query c).surfaceBody.drop 17)
    rw [h]; exact CodeEq.refl (withDot M D) _

theorem query_run (M : Algebra α) (D : DotPrimitive α) (c : QueryConfig) (s : State α) :
    FP.Structural.exec (withDot M D) (queryKernel c) s=FP.Structural.exec (withDot M D) (query c) s := by
  cases h : queryKernel c; cases hc : query c
  simpa only [h,hc,ComputeKernel.surfaceBody,FP.Structural.exec] using (query_equal M D c).run s

theorem query_safe (M : Algebra α) (D : DotPrimitive α) (c : QueryConfig) (cap : RegionName → Nat) (s : State α) :
    TraceSafe (withDot M D) cap (queryKernel c) s ↔ TraceSafe (withDot M D) cap (query c) s := (query_equal M D c).safe cap s

theorem shell_body (c : QueryConfig) :
    (shell c).surfaceBody=(queryKernel c).surfaceBody ++ [.effectMarker "tl.debug_barrier"] ++
      (backwardKernel c.toBackwardConfig).surfaceBody := rfl

theorem fusedShell_body (c : QueryConfig) :
    (fusedShell c).surfaceBody=(query c).surfaceBody ++ [.effectMarker "tl.debug_barrier"] ++
      (backwardKernel c.toBackwardConfig).surfaceBody := rfl

theorem shell_equal (M : Algebra α) (D : DotPrimitive α) (c : QueryConfig) :
    CodeEq (withDot M D) (shell c).surfaceBody (fusedShell c).surfaceBody := by
  rw [shell_body,fusedShell_body]
  exact ((query_equal M D c).append (CodeEq.refl (withDot M D) _)).append (CodeEq.refl (withDot M D) _)

theorem shell_original (shape : Shape) (scale : ℝ) (tile : Tile) :
    (shell (dqConfig shape scale tile)).surfaceBody=(combinedKernel shape scale tile).surfaceBody := rfl

#axiomsClean forward_run
#axiomsClean query_run
#axiomsClean shell_equal
end
end VeriTile.Bench.Repairs.ParallelAttentionKernel.Fused
