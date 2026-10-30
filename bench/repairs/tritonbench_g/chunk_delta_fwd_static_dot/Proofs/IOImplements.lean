import bench.repairs.tritonbench_g.chunk_delta_fwd_static_dot.Proofs.IOSafety
import bench.repairs.tritonbench_g.chunk_delta_fwd_static_dot.Proofs.GridSchedule
import bench.repairs.tritonbench_g.chunk_delta_fwd_static_dot.Proofs.RealBinding
import bench.repairs.tritonbench_g.chunk_delta_fwd_static_dot.Proofs.NativeIOBinding

namespace VeriTile.Bench.Repairs.ChunkDeltaFwdStaticDot.IOImplements
open VeriTile Triton FP.Structural ChunkDeltaFwd Interface Reference
open VeriTile.Bench.Repairs.ChunkGlaSimple.ScalarGraph (WordPrimitives)
open VeriTile.Bench.Repairs.ChunkGlaFwd.NativeStaging (Expr)
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 12000
set_option linter.unusedSimpArgs false

theorem real (p : Parameters) (shape : Geometry.ShapeValid p)
    (schedule : List PID) (order : schedule.Perm (grid p)) :
    (io Real.algebra Real.dot p schedule id id).Implements (realFormula p) := by
  apply ProgramIO.Implements.intro
  · intro xs bounds hb initial _
    exact IOSafety.schedule Real.algebra Real.dot p shape schedule order id id xs bounds hb initial
  · intro xs initial hin
    obtain ⟨final,ran,mem,_⟩ := GridSchedule.executes Real.algebra Real.dot p schedule
      initial.mem initial (SameInputs.refl initial.mem)
    have result := Real.grid_to_real p shape initial.mem initial.mem final.mem
      (by rw [mem]; exact grid_spec Real.algebra Real.dot p shape schedule order initial.mem initial.mem)
    have hb := IOBinding.reads Real.algebra Real.dot p schedule id id xs initial hin
    refine ⟨final,ran,?_,?_⟩
    · intro port j active
      fin_cases port
      · change mapCell id (final.mem "history" (((j.1.val*p.NT+j.2.1.val)*p.K+j.2.2.1.val)*p.V+j.2.2.2.1.val)) = Cell.mk .real _
        rw [result.history j.1.val j.1.isLt j.2.1.val j.2.1.isLt j.2.2.1.val j.2.2.1.isLt
          j.2.2.2.1.val j.2.2.2.1.isLt]
        simp only [mapCell,mapValue,id_eq]
        congr 1
        exact RealBinding.state p xs initial.mem hb j.1.val j.1.isLt j.2.1.val
          j.2.2.1.val j.2.2.2.1.val j.2.2.1.isLt j.2.2.2.1.isLt
      · change mapCell id (final.mem "corrected" ((j.1.val*p.T+j.2.1.val)*p.V+j.2.2.1.val)) = Cell.mk .real _
        rw [result.corrected j.1.val j.1.isLt j.2.1.val j.2.1.isLt j.2.2.1.val j.2.2.1.isLt]
        simp only [mapCell,mapValue,id_eq]
        congr 1
        exact RealBinding.corrected p xs initial.mem hb j.1.val j.2.1.val j.2.2.1.val
          j.1.isLt j.2.1.isLt j.2.2.1.isLt
      · change mapCell id (final.mem "final" ((j.1.val*p.K+j.2.1.val)*p.V+j.2.2.1.val)) = Cell.mk .real _
        rw [result.final active j.1.val j.1.isLt j.2.1.val j.2.1.isLt j.2.2.1.val j.2.2.1.isLt]
        simp only [mapCell,mapValue,id_eq]
        congr 1
        exact RealBinding.state p xs initial.mem hb j.1.val j.1.isLt p.NT
          j.2.1.val j.2.2.1.val j.2.1.isLt j.2.2.1.isLt
    · intro r o outside
      exact result.frame r o (IOBinding.outside Real.algebra Real.dot p schedule id id xs r o outside)

theorem native (I : WordPrimitives) (p : Parameters) (shape : Geometry.ShapeValid p)
    (schedule : List PID) (order : schedule.Perm (grid p)) :
    (io Native.algebra Native.dot p schedule Expr.bits (nativeObserve I)).Implements (nativeFormula I p shape) := by
  apply ProgramIO.Implements.intro
  · intro xs bounds hb initial _
    exact IOSafety.schedule Native.algebra Native.dot p shape schedule order Expr.bits (nativeObserve I)
      xs bounds hb initial
  · intro xs initial hin
    obtain ⟨final,ran,mem,_⟩ := GridSchedule.executes Native.algebra Native.dot p schedule
      initial.mem initial (SameInputs.refl initial.mem)
    have result : GridSpec Native.algebra Native.dot p shape initial.mem initial.mem final.mem := by
      rw [mem]
      exact grid_spec Native.algebra Native.dot p shape schedule order initial.mem initial.mem
    have hb := IOBinding.reads Native.algebra Native.dot p schedule Expr.bits (nativeObserve I) xs initial hin
    refine ⟨final,ran,?_,?_⟩
    · intro port j active
      fin_cases port
      · change mapCell (nativeObserve I) (final.mem "history" (((j.1.val*p.NT+j.2.1.val)*p.K+j.2.2.1.val)*p.V+j.2.2.2.1.val)) = Cell.mk .real _
        rw [result.history j.1.val j.1.isLt j.2.1.val j.2.1.isLt j.2.2.1.val j.2.2.1.isLt
          j.2.2.2.1.val j.2.2.2.1.isLt]
        simp only [mapCell,mapValue,nativeObserve]
        rw [NativeIOBinding.history p shape xs initial.mem hb j.1.val j.2.1.val
          j.2.2.1.val j.2.2.2.1.val j.1.isLt j.2.2.1.isLt j.2.2.2.1.isLt]
        rfl
      · change mapCell (nativeObserve I) (final.mem "corrected" ((j.1.val*p.T+j.2.1.val)*p.V+j.2.2.1.val)) = Cell.mk .real _
        rw [result.corrected j.1.val j.1.isLt j.2.1.val j.2.1.isLt j.2.2.1.val j.2.2.1.isLt]
        simp only [mapCell,mapValue,nativeObserve]
        rw [NativeIOBinding.corrected p shape xs initial.mem hb j.1.val j.2.1.val j.2.2.1.val
          j.1.isLt j.2.2.1.isLt]
        rfl
      · change mapCell (nativeObserve I) (final.mem "final" ((j.1.val*p.K+j.2.1.val)*p.V+j.2.2.1.val)) = Cell.mk .real _
        rw [result.final active j.1.val j.1.isLt j.2.1.val j.2.1.isLt j.2.2.1.val j.2.2.1.isLt]
        simp only [mapCell,mapValue,nativeObserve,finalValue]
        rw [NativeIOBinding.history p shape xs initial.mem hb j.1.val p.NT j.2.1.val j.2.2.1.val
          j.1.isLt j.2.1.isLt j.2.2.1.isLt]
        rfl
    · intro r o outside
      exact result.frame r o (IOBinding.outside Native.algebra Native.dot p schedule Expr.bits
        (nativeObserve I) xs r o outside)

#axiomsClean real
#axiomsClean native
end
end VeriTile.Bench.Repairs.ChunkDeltaFwdStaticDot.IOImplements
