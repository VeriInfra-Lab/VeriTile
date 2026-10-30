import bench.tritonbench_g.triton_attention.Proofs.IOSafety
import bench.tritonbench_g.triton_attention.Proofs.FiniteExecution

namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.PublicIO
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.TritonAttention.Whole001
noncomputable section
set_option maxHeartbeats 3000000

theorem real_results (c : Shape) (scale : ℝ) (fw pw bw : List Grid.Program)
    (hf : Grid.FullSchedule (Grid.forwardGrid c) fw)
    (hp : Grid.FullSchedule (Grid.preprocessGrid c) pw)
    (hb : Grid.FullSchedule (Grid.backwardGrid c) bw)
    (xs : Inputs ℝ c) (admitted : ZeroDQ 0 c xs) (memory : Grid.Memory Erased.R)
    (bound : (realIO c scale fw pw bw).InputCells xs memory) :
    ∃ final,runMaterializedSchedule ((realIO c scale fw pw bw).schedule xs) memory=some final ∧
      (realIO c scale fw pw bw).OutputBinding xs (realFormula c scale xs) final ∧
      (realIO c scale fw pw bw).Frame xs memory final := by
  have inputs : ∀ r∈FiniteExecution.inputRegions,∀ o,o<c.elements →
      (memory r o).read .fp32=some (inputAt 0 c xs r o) := by
    intro r hr o ho
    have cell := input_cells 0 c some (WithBot.unbotD 0) _ xs memory bound r hr o ho
    rw [cell]
    rfl
  have zero := input_zero 0 c some (WithBot.unbotD 0) _ xs admitted memory bound
  obtain ⟨final,ran,values,frame⟩ := FiniteExecution.real c scale fw pw bw hf hp hb memory (inputAt 0 c xs) inputs zero
  refine ⟨final,(real_schedule c scale fw pw bw xs memory).trans ran,?_,?_⟩
  · intro port
    fin_cases port
    · intro i _
      exact congrArg (FP.Structural.mapCell (WithBot.unbotD 0)) (values.output (coordinate c i))
    · intro i _
      exact congrArg (FP.Structural.mapCell (WithBot.unbotD 0)) (values.dq (coordinate c i))
    · intro i _
      exact congrArg (FP.Structural.mapCell (WithBot.unbotD 0)) (values.dk (coordinate c i))
    · intro i _
      exact congrArg (FP.Structural.mapCell (WithBot.unbotD 0)) (values.dv (coordinate c i))
  · intro r o exterior
    exact frame r o (outside c some (WithBot.unbotD 0) _ xs r o exterior)

theorem native_results (scalar : Checked.ScalarABI) (c : Shape)
    (owners : TritonAttention.Whole001.Native.Owners) (I : Native.Prior.Instructions)
    (literal : ℝ→Native.Word) (fw pw bw : List Grid.Program)
    (hf : Grid.FullSchedule (Grid.forwardGrid c) fw)
    (hp : Grid.FullSchedule (Grid.preprocessGrid c) pw)
    (hb : Grid.FullSchedule (Grid.backwardGrid c) bw)
    (xs : Inputs Native.Word c) (admitted : ZeroDQ 0 c xs) (memory : Grid.Memory Native.Graph)
    (bound : (nativeIO scalar c owners I literal fw pw bw).InputCells xs memory) :
    ∃ final,runMaterializedSchedule ((nativeIO scalar c owners I literal fw pw bw).schedule xs) memory=some final ∧
      (nativeIO scalar c owners I literal fw pw bw).OutputBinding xs (nativeFormula scalar c owners I literal xs) final ∧
      (nativeIO scalar c owners I literal fw pw bw).Frame xs memory final := by
  have inputs := input_cells 0 c Native.Graph.word (Native.Graph.eval I literal) _ xs memory bound
  have zero := input_zero 0 c Native.Graph.word (Native.Graph.eval I literal) _ xs admitted memory bound
  obtain ⟨final,ran,values,frame⟩ := FiniteExecution.native (Native.constexprScale scalar) c (scalar.project scalar.raw)
    owners fw pw bw hf hp hb memory (fun r o=>.word (inputAt 0 c xs r o)) inputs zero
  refine ⟨final,(native_schedule scalar c owners I literal fw pw bw xs memory).trans ran,?_,?_⟩
  · intro port
    fin_cases port
    · intro i _
      exact congrArg (FP.Structural.mapCell (Native.Graph.eval I literal)) (values.1 (coordinate c i))
    · intro i _
      exact congrArg (FP.Structural.mapCell (Native.Graph.eval I literal)) (values.2.1 (coordinate c i))
    · intro i _
      exact congrArg (FP.Structural.mapCell (Native.Graph.eval I literal)) (values.2.2.1 (coordinate c i))
    · intro i _
      exact congrArg (FP.Structural.mapCell (Native.Graph.eval I literal)) (values.2.2.2 (coordinate c i))
  · intro r o exterior
    exact frame r o (outside c Native.Graph.word (Native.Graph.eval I literal) _ xs r o exterior)

theorem real_implements (c : Shape) (scale : ℝ) (fw pw bw : List Grid.Program)
    (hf : Grid.FullSchedule (Grid.forwardGrid c) fw)
    (hp : Grid.FullSchedule (Grid.preprocessGrid c) pw)
    (hb : Grid.FullSchedule (Grid.backwardGrid c) bw) :
    (realIO c scale fw pw bw).ImplementsOnCells (ZeroDQ 0 c) (realFormula c scale) := by
  apply MaterializedProgramIO.ImplementsOnCells.intro
  · intro xs _ bounds backed memory _
    exact real_safe c scale fw pw bw hf hp hb xs bounds backed memory
  · exact real_results c scale fw pw bw hf hp hb

theorem native_implements (scalar : Checked.ScalarABI) (c : Shape)
    (owners : TritonAttention.Whole001.Native.Owners) (I : Native.Prior.Instructions)
    (literal : ℝ→Native.Word) (fw pw bw : List Grid.Program)
    (hf : Grid.FullSchedule (Grid.forwardGrid c) fw)
    (hp : Grid.FullSchedule (Grid.preprocessGrid c) pw)
    (hb : Grid.FullSchedule (Grid.backwardGrid c) bw) :
    (nativeIO scalar c owners I literal fw pw bw).ImplementsOnCells (ZeroDQ 0 c)
      (nativeFormula scalar c owners I literal) := by
  apply MaterializedProgramIO.ImplementsOnCells.intro
  · intro xs _ bounds backed memory _
    exact native_safe scalar c owners I literal fw pw bw hf hp hb xs bounds backed memory
  · exact native_results scalar c owners I literal fw pw bw hf hp hb

#axiomsClean real_implements
#axiomsClean native_implements
end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.PublicIO
