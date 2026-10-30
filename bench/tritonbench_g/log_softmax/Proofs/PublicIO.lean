import bench.tritonbench_g.log_softmax.IO
import bench.tritonbench_g.log_softmax.Proofs.IOBinding
import bench.tritonbench_g.log_softmax.Proofs.RawCorrect
import Mathlib.Tactic.FinCases

namespace VeriTile.Bench.Repairs.LogSoftmax.PublicIO
open VeriTile.Triton FP.Structural Address
noncomputable section

theorem forward (M N K BM : Nat) (retained : Grid.RetainedBlock BM)
    (plan : Plan N (Grid.blockColumns N)) :
    (IO.realForward M N K BM plan).Implements (Reference.forwardFormula M N K BM plan) := by
  apply ProgramIO.Implements.intro
  · intro vals bounds bounded initial _
    exact IOBinding.schedule_safe .forward some (fun v => v.getD 0)
      (KernelMath.forwardEngine RealProof.ops plan).algebra Bool.false M N K BM _
      (block_positive N) bounds vals bounded _ initial
  · intro vals initial inputs
    have bound : KernelMath.InputValues initial "input" M N K (IO.vector 0 (vals (0 : Fin 1))) := by
      intro i hi
      simpa [ProgramIO.InputBinding,IO.realForward,IO.forward,IO.vector,hi,mapValue] using
        inputs (0 : Fin 1) (⟨i,hi⟩,PUnit.unit) trivial
    obtain ⟨final,ran,values,preserved⟩ :=
      RawCorrect.forward_grid_correctness M N K BM retained plan initial _ bound
    refine ⟨final,?_,?_,?_⟩
    · simpa only [IO.realForward,IO.forward,IO.schedule,IOBinding.engine_run] using ran
    · intro o j _
      fin_cases o
      have value := values (Coordinates.decode j.1)
      rw [Coordinates.decode_linear] at value
      change mapCell (fun v : WithBot ℝ => v.getD 0) (final.mem "output" j.1.val)=
        Cell.mk .real (Reference.forward M N K (vals (0 : Fin 1)) j)
      rw [value]
      rfl
    · exact IOBinding.frame .forward some (fun v => v.getD 0)
        (KernelMath.forwardEngine RealProof.ops plan).algebra Bool.false M N K BM _
        vals initial final preserved

theorem backward (M N K BM : Nat) (retained : Grid.RetainedBlock BM)
    (plan : Plan N (Grid.blockColumns N)) :
    (IO.realBackward M N K BM plan).Implements (Reference.backwardFormula M N K BM plan) := by
  apply ProgramIO.Implements.intro
  · intro vals bounds bounded initial _
    exact IOBinding.schedule_safe .backward some (fun v => v.getD 0)
      (KernelMath.backwardEngine RealProof.ops plan).algebra Bool.false M N K BM _
      (block_positive N) bounds vals bounded _ initial
  · intro vals initial inputs
    have saved : KernelMath.InputValues initial "input" M N K (IO.vector 0 (vals (0 : Fin 2))) := by
      intro i hi
      simpa [ProgramIO.InputBinding,IO.realBackward,IO.backward,IO.vector,hi,mapValue] using
        inputs (0 : Fin 2) (⟨i,hi⟩,PUnit.unit) trivial
    have upstream : KernelMath.InputValues initial "gradient" M N K (IO.vector 0 (vals (1 : Fin 2))) := by
      intro i hi
      simpa [ProgramIO.InputBinding,IO.realBackward,IO.backward,IO.vector,hi,mapValue] using
        inputs (1 : Fin 2) (⟨i,hi⟩,PUnit.unit) trivial
    obtain ⟨final,ran,values,preserved⟩ :=
      RawCorrect.backward_grid_correctness M N K BM retained plan initial _ _ saved upstream
    refine ⟨final,?_,?_,?_⟩
    · simpa only [IO.realBackward,IO.backward,IO.schedule,IOBinding.engine_run] using ran
    · intro o j _
      fin_cases o
      have value := values (Coordinates.decode j.1)
      rw [Coordinates.decode_linear] at value
      change mapCell (fun v : WithBot ℝ => v.getD 0) (final.mem "output" j.1.val)=
        Cell.mk .real (Reference.backward M N K (vals (0 : Fin 2)) (vals (1 : Fin 2)) j)
      rw [value]
      rfl
    · exact IOBinding.frame .backward some (fun v => v.getD 0)
        (KernelMath.backwardEngine RealProof.ops plan).algebra Bool.false M N K BM _
        vals initial final preserved

end
end VeriTile.Bench.Repairs.LogSoftmax.PublicIO
