import bench.tritonbench_g.softmax_optimize.Proofs.Input
import bench.tritonbench_g.softmax_optimize.Proofs.Mathematical_SafetyOutput
import bench.tritonbench_g.softmax_optimize.Proofs.Mathematical_SafetyState

namespace VeriTile.Bench.Repairs.SoftmaxOptimize.IOSafetyBodies
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.SoftmaxOptimize
open SignedTransport
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000

theorem prefix_safe (bounds : RegionBounds) (X : RegionName) (B : Nat) (s : BlockState) :
    Stmt.TraceSafeList bounds (Source.indexPrefix X B) s := by
  apply TraceProofs.all_safe
  intro st hst t
  simp only [Source.indexPrefix, List.mem_cons, List.not_mem_nil, or_false] at hst
  rcases hst with rfl | rfl | rfl <;> simp [Stmt.TraceSafe, Op.SafeAt.eq_def]

theorem update_safe (bounds : RegionBounds) (B : Nat) (s : BlockState) :
    Stmt.TraceSafeList bounds (Source.update B) s := by
  apply TraceProofs.all_safe
  intro st hst t
  simp only [Source.update, List.mem_cons, List.not_mem_nil, or_false] at hst
  rcases hst with rfl | rfl | rfl | rfl | rfl | rfl <;>
    simp [sourceMaximum, Stmt.TraceSafe, Op.SafeAt.eq_def]

theorem input_full (bounds : RegionBounds) (X : RegionName) (N B start row : Nat) (s : BlockState)
    (hpid : s.regs .int [] "pid_m" = some (Tile.scalar (row : Int)))
    (hcols : s.regs .int [] "cols" = some (Tile.scalar (N : Int)))
    (hx : ∀ i : Fin B, row*N+(start+i.val) < bounds X) :
    Stmt.TraceSafeList bounds (Source.inputFull X N B)
      (s.setReg "start_n" .int [] (Tile.scalar (start : Int))) := by
  unfold Source.inputFull
  apply TraceProofs.append_safe _ _ (prefix_safe bounds X B _)
  intro t executed
  have same := Option.some.inj ((Primitive.prefix_run X N B start row s hpid hcols).symm.trans executed)
  subst t
  change Stmt.TraceSafeList bounds
    (.assign .real [B] "inp" (.load .real (.ptr (.ref .ptr [B] "input_ptrs")) .none) :: Source.update B) _
  apply TraceProofs.assign_cons
  · exact Safety.full_load_safe bounds X N B start row _ (by simp [prefixIntState]) hx
  · intro value
    exact update_safe bounds B _

theorem input_tail (bounds : RegionBounds) (X : RegionName) (N B start row : Nat) (s : BlockState)
    (hpid : s.regs .int [] "pid_m" = some (Tile.scalar (row : Int)))
    (hcols : s.regs .int [] "cols" = some (Tile.scalar (N : Int)))
    (hx : ∀ i : Fin B, start+i.val < N → row*N+(start+i.val) < bounds X) :
    Stmt.TraceSafeList bounds (Source.inputTail X N B)
      (s.setReg "start_n" .int [] (Tile.scalar (start : Int))) := by
  unfold Source.inputTail
  rw [List.append_assoc]
  apply TraceProofs.append_safe _ _ (prefix_safe bounds X B _)
  intro t executed
  have same := Option.some.inj ((Primitive.prefix_run X N B start row s hpid hcols).symm.trans executed)
  subst t
  change Stmt.TraceSafeList bounds (Source.maskStatement B :: _) _
  rw [TraceProofs.cons_of_run_iff (Input.mask_run s X N B start row hcols)]
  refine ⟨by simp [Source.maskStatement, Stmt.TraceSafe, Op.SafeAt.eq_def], ?_⟩
  change Stmt.TraceSafeList bounds
    (.assign .real [B] "inp" (.load .real (.ptr (.ref .ptr [B] "input_ptrs"))
      (.maskOther (.ref .bool [B] "mask") (Op.negInf.broadcast [B]))) :: Source.update B) _
  apply TraceProofs.assign_cons
  · exact Safety.tail_load_safe bounds X N B start row _
      (by simp [Input.maskedPrefix,prefixIntState]) (by simp [Input.maskedPrefix,Input.tailMask,Tile.vec]) hx
  · intro value
    exact update_safe bounds B _

theorem output_full (bounds : RegionBounds) (Y X : RegionName) (N B start row : Nat) (s : BlockState)
    (hpid : s.regs .int [] "pid_m" = some (Tile.scalar (row : Int)))
    (hcols : s.regs .int [] "cols" = some (Tile.scalar (N : Int)))
    (hx : ∀ i : Fin B, row*N+(start+i.val) < bounds X)
    (hy : ∀ i : Fin B, row*N+(start+i.val) < bounds Y) :
    Stmt.TraceSafeList bounds (Source.outputFull Y X N B)
      (s.setReg "start_n" .int [] (Tile.scalar (start : Int))) := by
  unfold Source.outputFull
  rw [List.append_assoc]
  apply TraceProofs.append_safe _ _ (prefix_safe bounds X B _)
  intro t executed
  have same := Option.some.inj ((Primitive.prefix_run X N B start row s hpid hcols).symm.trans executed)
  subst t
  simp only [loopApass2body, List.drop, List.take, List.cons_append, List.nil_append]
  apply TraceProofs.assign_cons
  · exact Safety.full_load_safe bounds X N B start row _ (by simp [prefixIntState]) hx
  · intro inp
    apply TraceProofs.assign_cons
    · simp [Op.SafeAt.eq_def]
    · intro e
      apply TraceProofs.assign_cons
      · simp [Op.SafeAt.eq_def]
      · intro out
        exact Safety.full_output_end_safe bounds Y N B start row _
          (by simp [prefixIntState]) hy

theorem output_tail (bounds : RegionBounds) (Y X : RegionName) (N B start row : Nat) (s : BlockState)
    (hpid : s.regs .int [] "pid_m" = some (Tile.scalar (row : Int)))
    (hcols : s.regs .int [] "cols" = some (Tile.scalar (N : Int)))
    (hx : ∀ i : Fin B, start+i.val < N → row*N+(start+i.val) < bounds X)
    (hy : ∀ i : Fin B, start+i.val < N → row*N+(start+i.val) < bounds Y) :
    Stmt.TraceSafeList bounds (Source.outputTail Y X N B)
      (s.setReg "start_n" .int [] (Tile.scalar (start : Int))) := by
  unfold Source.outputTail
  simp only [List.append_assoc]
  apply TraceProofs.append_safe _ _ (prefix_safe bounds X B _)
  intro t executed
  have same := Option.some.inj ((Primitive.prefix_run X N B start row s hpid hcols).symm.trans executed)
  subst t
  change Stmt.TraceSafeList bounds (Source.maskStatement B :: _) _
  rw [TraceProofs.cons_of_run_iff (Input.mask_run s X N B start row hcols)]
  refine ⟨by simp [Source.maskStatement, Stmt.TraceSafe, Op.SafeAt.eq_def], ?_⟩
  simp only [loopBpass2body, List.drop, List.take, List.cons_append, List.nil_append]
  apply TraceProofs.assign_cons
  · exact Safety.tail_load_safe bounds X N B start row _
      (by simp [Input.maskedPrefix,prefixIntState]) (by simp [Input.maskedPrefix,Input.tailMask,Tile.vec]) hx
  · intro inp
    apply TraceProofs.assign_cons
    · simp [Op.SafeAt.eq_def]
    · intro e
      apply TraceProofs.assign_cons
      · simp [Op.SafeAt.eq_def]
      · intro out
        exact Safety.tail_output_end_safe bounds Y N B start row _
          (by simp [Input.maskedPrefix,prefixIntState])
          (by simp [Input.maskedPrefix,Input.tailMask,Tile.vec]) hy

#axiomsClean input_full
#axiomsClean input_tail
#axiomsClean output_full
#axiomsClean output_tail
end
end VeriTile.Bench.Repairs.SoftmaxOptimize.IOSafetyBodies
