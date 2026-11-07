import bench.tritonbench_g.parallel_attention.Proofs.FusedLocal

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
    CodeEq M (loopBody (forwardKernel c) 17) (loopBody (forward c) 17) := by
  rw [split_code (loopBody (forwardKernel c) 17) 4 2,
    split_code (loopBody (forward c) 17) 4 1]
  apply CodeEq.append
  · apply CodeEq.append
    · have h : (loopBody (forwardKernel c) 17).take 4=(loopBody (forward c) 17).take 4 := rfl
      rw [h]; exact CodeEq.refl M _
    · exact ⟨forward_local_run M c,forward_local_safe M c⟩
  · have h : (loopBody (forwardKernel c) 17).drop 6=(loopBody (forward c) 17).drop 5 := rfl
    change CodeEq M ((loopBody (forwardKernel c) 17).drop 6) ((loopBody (forward c) 17).drop 5)
    rw [h]; exact CodeEq.refl M _

theorem forward_equal (M : Algebra α) (c : ForwardConfig) :
    CodeEq M (forwardKernel c).surfaceBody (forward c).surfaceBody := by
  rw [split_code (forwardKernel c).surfaceBody 17 1,split_code (forward c).surfaceBody 17 1]
  apply CodeEq.append
  · apply CodeEq.append
    · have h : (forwardKernel c).surfaceBody.take 17=(forward c).surfaceBody.take 17 := rfl
      rw [h]; exact CodeEq.refl M _
    · change CodeEq M [.forRangeDyn "_" _ _ _ (loopBody (forwardKernel c) 17)]
        [.forRangeDyn "_" _ _ _ (loopBody (forward c) 17)]
      exact (dynamic_context (forward_body_equal M c) _ _ _ _).singleton
  · have h : (forwardKernel c).surfaceBody.drop 18=(forward c).surfaceBody.drop 18 := rfl
    change CodeEq M ((forwardKernel c).surfaceBody.drop 18) ((forward c).surfaceBody.drop 18)
    rw [h]; exact CodeEq.refl M _

theorem forward_run (M : Algebra α) (c : ForwardConfig) (s : State α) :
    FP.Structural.exec M (forwardKernel c) s=FP.Structural.exec M (forward c) s := by
  cases h : forwardKernel c; cases hc : forward c
  simpa only [h,hc,ComputeKernel.surfaceBody,FP.Structural.exec] using (forward_equal M c).run s

theorem forward_safe (M : Algebra α) (c : ForwardConfig) (cap : RegionName → Nat) (s : State α) :
    TraceSafe M cap (forwardKernel c) s ↔ TraceSafe M cap (forward c) s := (forward_equal M c).safe cap s

theorem query_body_equal (M : Algebra α) (D : DotPrimitive α) (c : QueryConfig) :
    CodeEq (withDot M D) (loopBody (queryKernel c) 17) (loopBody (query c) 17) := by
  rw [split_code (loopBody (queryKernel c) 17) 6 2,
    split_code (loopBody (query c) 17) 6 1]
  apply CodeEq.append
  · apply CodeEq.append
    · have h : (loopBody (queryKernel c) 17).take 6=(loopBody (query c) 17).take 6 := rfl
      rw [h]; exact CodeEq.refl (withDot M D) _
    · exact ⟨query_local_run M D c,query_local_safe M D c⟩
  · have h : (loopBody (queryKernel c) 17).drop 8=(loopBody (query c) 17).drop 7 := rfl
    change CodeEq (withDot M D) ((loopBody (queryKernel c) 17).drop 8) ((loopBody (query c) 17).drop 7)
    rw [h]; exact CodeEq.refl (withDot M D) _

theorem query_equal (M : Algebra α) (D : DotPrimitive α) (c : QueryConfig) :
    CodeEq (withDot M D) (queryKernel c).surfaceBody (query c).surfaceBody := by
  rw [split_code (queryKernel c).surfaceBody 17 1,split_code (query c).surfaceBody 17 1]
  apply CodeEq.append
  · apply CodeEq.append
    · have h : (queryKernel c).surfaceBody.take 17=(query c).surfaceBody.take 17 := rfl
      rw [h]; exact CodeEq.refl (withDot M D) _
    · change CodeEq (withDot M D) [.forRangeDyn "_" _ _ _ (loopBody (queryKernel c) 17)]
        [.forRangeDyn "_" _ _ _ (loopBody (query c) 17)]
      exact (dynamic_context (query_body_equal M D c) _ _ _ _).singleton
  · have h : (queryKernel c).surfaceBody.drop 18=(query c).surfaceBody.drop 18 := rfl
    change CodeEq (withDot M D) ((queryKernel c).surfaceBody.drop 18) ((query c).surfaceBody.drop 18)
    rw [h]; exact CodeEq.refl (withDot M D) _

theorem query_run (M : Algebra α) (D : DotPrimitive α) (c : QueryConfig) (s : State α) :
    FP.Structural.exec (withDot M D) (queryKernel c) s=FP.Structural.exec (withDot M D) (query c) s := by
  cases h : queryKernel c; cases hc : query c
  simpa only [h,hc,ComputeKernel.surfaceBody,FP.Structural.exec] using (query_equal M D c).run s

theorem query_safe (M : Algebra α) (D : DotPrimitive α) (c : QueryConfig) (cap : RegionName → Nat) (s : State α) :
    TraceSafe (withDot M D) cap (queryKernel c) s ↔ TraceSafe (withDot M D) cap (query c) s := (query_equal M D c).safe cap s

/-- The changed score assignments are compared inside the actual caller,
whose PID decoding, complete helper calls, barrier and DKV suffix remain. -/
theorem shell_body_equal (M : Algebra α) (D : DotPrimitive α) (c : QueryConfig) :
    CodeEq (withDot M D) (loopBody (shell c) 22) (loopBody (fusedShell c) 22) := by
  rw [split_code (loopBody (shell c) 22) 6 2,
    split_code (loopBody (fusedShell c) 22) 6 1]
  apply CodeEq.append
  · apply CodeEq.append
    · have same : (loopBody (shell c) 22).take 6=(loopBody (fusedShell c) 22).take 6:=rfl
      rw [same]
      exact CodeEq.refl (withDot M D) _
    · have original : ((loopBody (shell c) 22).drop 6).take 2=
          ((loopBody (queryKernel c) 17).drop 6).take 2:=rfl
      have candidate : ((loopBody (fusedShell c) 22).drop 6).take 1=
          ((loopBody (query c) 17).drop 6).take 1:=rfl
      rw [original,candidate]
      exact ⟨query_local_run M D c,query_local_safe M D c⟩
  · have same : (loopBody (shell c) 22).drop 8=(loopBody (fusedShell c) 22).drop 7:=rfl
    change CodeEq (withDot M D) ((loopBody (shell c) 22).drop 8) ((loopBody (fusedShell c) 22).drop 7)
    rw [same]
    exact CodeEq.refl (withDot M D) _

theorem shell_equal (M : Algebra α) (D : DotPrimitive α) (c : QueryConfig) :
    CodeEq (withDot M D) (shell c).surfaceBody (fusedShell c).surfaceBody := by
  rw [split_code (shell c).surfaceBody 22 1,split_code (fusedShell c).surfaceBody 22 1]
  apply CodeEq.append
  · apply CodeEq.append
    · have same : (shell c).surfaceBody.take 22=(fusedShell c).surfaceBody.take 22:=rfl
      rw [same]
      exact CodeEq.refl (withDot M D) _
    · change CodeEq (withDot M D) [.forRangeDyn "_" _ _ _ (loopBody (shell c) 22)]
        [.forRangeDyn "_" _ _ _ (loopBody (fusedShell c) 22)]
      exact (dynamic_context (shell_body_equal M D c) _ _ _ _).singleton
  · have same : (shell c).surfaceBody.drop 23=(fusedShell c).surfaceBody.drop 23:=rfl
    change CodeEq (withDot M D) ((shell c).surfaceBody.drop 23) ((fusedShell c).surfaceBody.drop 23)
    rw [same]
    exact CodeEq.refl (withDot M D) _

theorem shell_original (c : QueryConfig) :
    shell c=backwardPastedKernel c.q c.k c.v c.do_ c.dz c.dq c.dk c.dv
      c.s_qk_h c.s_qk_t c.s_qk_d c.s_vo_h c.s_vo_t c.s_vo_d c.scale
      c.B c.H c.T c.K c.V c.BTL c.BTS c.BK c.BV:=rfl

#axiomsClean forward_run
#axiomsClean query_run
#axiomsClean shell_equal
end
end VeriTile.Bench.Repairs.ParallelAttentionKernel.Fused
