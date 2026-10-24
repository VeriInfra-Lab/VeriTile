import bench.repairs.tritonbench_g.parallel_attention.Proofs.IOGeometry
import bench.repairs.tritonbench_g.parallel_attention.Proofs.IOBoundValues
import bench.repairs.tritonbench_g.parallel_attention.Proofs.IOGraphGrid
import bench.repairs.tritonbench_g.parallel_attention.Proofs.IOScheduleSafety

namespace VeriTile.Bench.Repairs.ParallelAttentionKernel.IOData
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.ParallelAttention
noncomputable section
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

theorem native_forward_result (a : Scalars NativeStage.Word) (d : NativePlans.Descriptor) (scale : ℝ)
    (xs : ChannelValues NativeStage.Expr (inputPorts d.shape)) (s : State NativeStage.Expr)
    (hin : (dataIO NativeStage.Expr .forward d.shape (nativeJobs .forward d scale)).InputBinding xs s) :
    ∃ final,runSchedule (nativeJobs .forward d scale) s=some final ∧
      (dataIO NativeStage.Expr .forward d.shape (nativeJobs .forward d scale)).ObservedOutputBinding
        (nativeObserver a d) xs (nativeExpected a d scale xs) final ∧
      (dataIO NativeStage.Expr .forward d.shape (nativeJobs .forward d scale)).Frame xs s final := by
  have agree:=inputs_agree .forward d.shape _ xs s hin
  obtain ⟨final,run,_,reads,frame⟩:=IOGraph.forward_grid
    (NativeStage.algebra .forward d) (NativeStage.dots .forward d.shape) d.shape scale (programs d.shape 128) s
  refine ⟨final,?_,?_,?_⟩
  · exact (IOGraph.forward_schedule _ _ d.shape scale _ s).trans run
  · intro o j active
    fin_cases o
    · let p:=program d.shape 128 d.shape.BV j
      have ha : (outputRectangle d.shape p.tile 128).active ⟨j.2.2.2.1,j.2.2.2.2.1,()⟩ := active.2
      have read:=reads p (program_member d.shape 128 p) "o" _ (Or.inl ⟨rfl,_,ha,rfl⟩)
      simp only [IOGraph.forwardTarget,ite_true] at read
      erw [rectangleTarget_at _ _ _ _ ha] at read
      rw [output_address] at read
      change mapCell (NativeStage.evaluate a .forward d p.tile)
        (mapCell id (final.mem "o" (outputWindow d.shape (⟨0,by decide⟩ : Fin 5) j)))=Cell.mk .real _
      rw [read,forward_bound _ _ .forward d.shape scale xs s p agree]
      rfl
    · let p:=normProgram d.shape j
      have read:=reads p (program_member d.shape 128 p) "z" _
        (Or.inr ⟨active.2.1,rfl,_,active.2.2,rfl⟩)
      rw [IOGraph.forward_normalizer_at _ _ d.shape scale p s j.2.2.2.1 active.2.2] at read
      rw [normalizer_address] at read
      change mapCell (NativeStage.evaluate a .forward d p.tile)
        (mapCell id (final.mem "z" (outputWindow d.shape (⟨1,by decide⟩ : Fin 5) j)))=Cell.mk .real _
      rw [read,forward_bound _ _ .forward d.shape scale xs s p agree]
      rfl
    all_goals cases active.1
  · intro R off outside
    exact frame R off (fun p _=>forward_outside d.shape _ xs R off outside p)

theorem native_backward_result (a : Scalars NativeStage.Word) (d : NativePlans.Descriptor) (scale : ℝ)
    (xs : ChannelValues NativeStage.Expr (inputPorts d.shape)) (s : State NativeStage.Expr)
    (hin : (dataIO NativeStage.Expr .backward d.shape (nativeJobs .backward d scale)).InputBinding xs s) :
    ∃ final,runSchedule (nativeJobs .backward d scale) s=some final ∧
      (dataIO NativeStage.Expr .backward d.shape (nativeJobs .backward d scale)).ObservedOutputBinding
        (nativeObserver a d) xs (nativeExpected a d scale xs) final ∧
      (dataIO NativeStage.Expr .backward d.shape (nativeJobs .backward d scale)).Frame xs s final := by
  have agree:=inputs_agree .backward d.shape _ xs s hin
  obtain ⟨final,run,_,reads,frame⟩:=IOGraph.backward_grid
    (NativeStage.algebra .query d) (NativeStage.dots .query d.shape)
    (NativeStage.algebra .keyValue d) (NativeStage.dots .keyValue d.shape)
    d.shape scale (programs d.shape 64) s
  refine ⟨final,?_,?_,?_⟩
  · exact (IOGraph.backward_schedule _ _ _ _ d.shape scale _ s).trans run
  · intro o j active
    fin_cases o
    · cases active.1
    · cases active.1
    · let p:=program d.shape 64 d.shape.BK j
      have ha : (queryRectangle d.shape p.tile).active ⟨j.2.2.2.1,j.2.2.2.2.1,()⟩ := by
        simpa [queryRectangle,Rectangle.active,Program.tile,program,row] using active.2
      have read:=reads p (program_member d.shape 64 p) "dq" _ (Or.inl ⟨rfl,_,ha,rfl⟩)
      simp only [IOGraph.backwardTarget,IOGraph.queryTarget,ite_true] at read
      erw [rectangleTarget_at _ _ _ _ ha] at read
      rw [query_address] at read
      change mapCell (NativeStage.evaluate a .query d p.tile)
        (mapCell id (final.mem "dq" (outputWindow d.shape (⟨2,by decide⟩ : Fin 5) j)))=Cell.mk .real _
      rw [read,query_bound _ _ d.shape scale xs s p agree]
      rfl
    · let p:=program d.shape 64 d.shape.BK j
      have ha : (keyRectangle d.shape p.tile).active ⟨j.2.2.2.1,j.2.2.2.2.1,()⟩ := by
        simpa [keyRectangle,queryRectangle,Rectangle.active,Program.tile,program,row] using active.2
      have read:=reads p (program_member d.shape 64 p) "dk" _ (Or.inr (Or.inl ⟨rfl,_,ha,rfl⟩))
      simp only [IOGraph.backwardTarget,IOGraph.keyValueTarget,
        show ¬(("dk" : RegionName)="dq") by decide,ite_false,ite_true] at read
      erw [rectangleTarget_at _ _ _ _ ha] at read
      rw [key_address] at read
      change mapCell (NativeStage.evaluate a .keyValue d p.tile)
        (mapCell id (final.mem "dk" (outputWindow d.shape (⟨3,by decide⟩ : Fin 5) j)))=Cell.mk .real _
      rw [read,keyValue_bound _ _ d.shape scale xs s p agree]
      rfl
    · let p:=program d.shape 64 d.shape.BV j
      have ha : (valueRectangle d.shape p.tile).active ⟨j.2.2.2.1,j.2.2.2.2.1,()⟩ := active.2
      have read:=reads p (program_member d.shape 64 p) "dv" _ (Or.inr (Or.inr ⟨rfl,_,ha,rfl⟩))
      simp only [IOGraph.backwardTarget,IOGraph.keyValueTarget,
        show ¬(("dv" : RegionName)="dq") by decide,
        show ¬(("dv" : RegionName)="dk") by decide,ite_false] at read
      erw [rectangleTarget_at _ _ _ _ ha] at read
      rw [value_address] at read
      change mapCell (NativeStage.evaluate a .keyValue d p.tile)
        (mapCell id (final.mem "dv" (outputWindow d.shape (⟨4,by decide⟩ : Fin 5) j)))=Cell.mk .real _
      rw [read,keyValue_bound _ _ d.shape scale xs s p agree]
      rfl
  · intro R off outside
    exact frame R off (fun p _=>backward_outside d.shape _ xs R off outside p)

theorem native_forward (a : Scalars NativeStage.Word) (d : NativePlans.Descriptor) (scale : ℝ) :
    (dataIO NativeStage.Expr .forward d.shape (nativeJobs .forward d scale)).Observes
      (nativeObserver a d) (nativeExpected a d scale) := by
  intro xs _ bounds hb s hin
  exact ⟨native_safe .forward d scale xs bounds hb s,native_forward_result a d scale xs s hin⟩

theorem native_backward (a : Scalars NativeStage.Word) (d : NativePlans.Descriptor) (scale : ℝ) :
    (dataIO NativeStage.Expr .backward d.shape (nativeJobs .backward d scale)).Observes
      (nativeObserver a d) (nativeExpected a d scale) := by
  intro xs _ bounds hb s hin
  exact ⟨native_safe .backward d scale xs bounds hb s,native_backward_result a d scale xs s hin⟩

#axiomsClean native_forward
#axiomsClean native_backward
end
end VeriTile.Bench.Repairs.ParallelAttentionKernel.IOData
