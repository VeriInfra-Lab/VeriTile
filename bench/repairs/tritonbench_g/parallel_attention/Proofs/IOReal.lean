import bench.repairs.tritonbench_g.parallel_attention.Proofs.IOGeometry
import bench.repairs.tritonbench_g.parallel_attention.Proofs.IORealInputs
import bench.repairs.tritonbench_g.parallel_attention.Proofs.IOScheduleSafety

namespace VeriTile.Bench.Repairs.ParallelAttentionKernel.IOData
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.ParallelAttention
noncomputable section
set_option maxHeartbeats 800000
set_option linter.unusedSimpArgs false

theorem real_forward (shape : Shape) (accepted : ShapeContract shape) (scale : ℝ) :
    (dataIO ℝ .forward shape (realJobs .forward shape scale)).Implements (realExpected shape scale) := by
  apply ProgramIO.Implements.intro
  · intro xs bounds hb s _
    exact real_safe .forward shape scale xs bounds hb s
  · intro xs s hin
    have agree:=inputs_agree .forward shape _ xs s hin
    obtain ⟨final,run,_,outputs,norms,frame⟩:=forward_grid_real shape accepted scale s
    refine ⟨final,?_,?_,?_⟩
    · exact (real_forward_schedule shape scale _ s).trans run
    · intro o j active
      fin_cases o
      · let p:=program shape 128 shape.BV j
        have ha : (outputRectangle shape p.tile 128).active ⟨j.2.2.2.1,j.2.2.2.2.1,()⟩ := active.2
        have read:=outputs p j.2.2.2.1 j.2.2.2.2.1 ha
        rw [output_address] at read
        change mapCell id (final.mem "o" (outputWindow shape (⟨0,by decide⟩ : Fin 5) j))=Cell.mk .real _
        rw [read]
        apply congrArg (Cell.mk .real)
        exact output_congr (tensor_agree .forward shape xs s p.head.val p.head.isLt agree)
          scale _ _ ha.1 ha.2
      · let p:=normProgram shape j
        have read:=norms p j.2.2.2.1 active.2.1 active.2.2
        rw [normalizer_address] at read
        change mapCell id (final.mem "z" (outputWindow shape (⟨1,by decide⟩ : Fin 5) j))=Cell.mk .real _
        rw [read]
        apply congrArg (Cell.mk .real)
        exact normalizer_congr (tensor_agree .forward shape xs s p.head.val p.head.isLt agree)
          scale _ active.2.2
      all_goals cases active.1
    · intro R off outside
      exact frame R off (fun p _=>forward_outside shape _ xs R off outside p)

theorem real_backward (shape : Shape) (accepted : ShapeContract shape) (scale : ℝ) :
    (dataIO ℝ .backward shape (realJobs .backward shape scale)).Implements (realExpected shape scale) := by
  apply ProgramIO.Implements.intro
  · intro xs bounds hb s _
    exact real_safe .backward shape scale xs bounds hb s
  · intro xs s hin
    have agree:=inputs_agree .backward shape _ xs s hin
    obtain ⟨final,run,_,reads,frame⟩:=backward_grid_real shape accepted scale s
    refine ⟨final,?_,?_,?_⟩
    · exact (real_backward_schedule shape scale _ s).trans run
    · intro o j active
      fin_cases o
      · cases active.1
      · cases active.1
      · let p:=program shape 64 shape.BK j
        have ha : (queryRectangle shape p.tile).active ⟨j.2.2.2.1,j.2.2.2.2.1,()⟩ := by
          simpa [queryRectangle,Rectangle.active,Program.tile,program,row] using active.2
        have read:=reads p (program_member shape 64 p) "dq" _ (Or.inl ⟨rfl,_,ha,rfl⟩)
        simp only [backwardMathTarget,ite_true] at read
        erw [rectangleTarget_at _ _ _ _ ha] at read
        rw [query_address] at read
        change mapCell id (final.mem "dq" (outputWindow shape (⟨2,by decide⟩ : Fin 5) j))=Cell.mk .real _
        rw [read]
        apply congrArg (Cell.mk .real)
        exact query_congr (tensor_agree .backward shape xs s p.head.val p.head.isLt agree)
          scale _ _ _ active.2.1 active.2.2
      · let p:=program shape 64 shape.BK j
        have ha : (keyRectangle shape p.tile).active ⟨j.2.2.2.1,j.2.2.2.2.1,()⟩ := by
          simpa [keyRectangle,queryRectangle,Rectangle.active,Program.tile,program,row] using active.2
        have read:=reads p (program_member shape 64 p) "dk" _ (Or.inr (Or.inl ⟨rfl,_,ha,rfl⟩))
        simp only [backwardMathTarget,show ¬(("dk" : RegionName)="dq") by decide,ite_false,ite_true] at read
        erw [rectangleTarget_at _ _ _ _ ha] at read
        rw [key_address] at read
        change mapCell id (final.mem "dk" (outputWindow shape (⟨3,by decide⟩ : Fin 5) j))=Cell.mk .real _
        rw [read]
        apply congrArg (Cell.mk .real)
        exact key_congr (tensor_agree .backward shape xs s p.head.val p.head.isLt agree)
          scale _ _ _ active.2.1 active.2.2
      · let p:=program shape 64 shape.BV j
        have ha : (valueRectangle shape p.tile).active ⟨j.2.2.2.1,j.2.2.2.2.1,()⟩ := active.2
        have read:=reads p (program_member shape 64 p) "dv" _ (Or.inr (Or.inr ⟨rfl,_,ha,rfl⟩))
        simp only [backwardMathTarget,show ¬(("dv" : RegionName)="dq") by decide,
          show ¬(("dv" : RegionName)="dk") by decide,ite_false] at read
        erw [rectangleTarget_at _ _ _ _ ha] at read
        rw [value_address] at read
        change mapCell id (final.mem "dv" (outputWindow shape (⟨4,by decide⟩ : Fin 5) j))=Cell.mk .real _
        rw [read]
        apply congrArg (Cell.mk .real)
        exact value_congr (tensor_agree .backward shape xs s p.head.val p.head.isLt agree)
          scale _ _ active.2.1 active.2.2
    · intro R off outside
      exact frame R off (fun p _=>backward_outside shape _ xs R off outside p)

#axiomsClean real_forward
#axiomsClean real_backward
end
end VeriTile.Bench.Repairs.ParallelAttentionKernel.IOData
