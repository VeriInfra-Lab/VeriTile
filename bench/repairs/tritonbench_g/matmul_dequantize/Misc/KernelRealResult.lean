import bench.repairs.tritonbench_g.matmul_dequantize.Whole001.Proofs.GPUCorrect

namespace VeriTile.Bench.Repairs.MatmulDequantize.KernelRealResult
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.MatmulDequantizeWhole001
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
noncomputable section
set_option maxHeartbeats 5000000

/-- Full raw source grid with independent mathematical outputs and exterior frame. -/
theorem gptq_kernel_correct (p : VeriTile.Bench.TritonBenchG.MatmulDequantize.FPTypedFirst.Parameters) (valid : Geometry.Valid (FirstGrid.geometry p))
    (ha : p.input≠p.output) (hb : p.weight≠p.output)
    (hs : p.scale≠p.output) (hz : p.zero≠p.output)
    (hBK : p.blockInner=32) (G : Nat) (hG : p.groupSize=32*G)
    (hmode : p.noGroups=Bool.true → p.groupSize=p.inner) (hK : p.inner=VeriTile.Bench.TritonBenchG.MatmulDequantize.FPTypedFirst.steps p*32)
    (initial : State ℝ) (jobs : List Nat)
    (schedule : jobs.Perm (List.range (Geometry.count (FirstGrid.geometry p)))) :
    ∃ final,Grid.run (FirstGrid.program Real.scalar Real.firstDot p valid ha hb hs hz) jobs initial=some final ∧
      (∀ pid,pid<Geometry.count (FirstGrid.geometry p) → ∀ i,Geometry.Active (FirstGrid.geometry p) pid i →
        final.mem p.output (Geometry.offset (FirstGrid.geometry p) pid i)=.mk .fp16 (Real.rawMatmul (RealFirst.inputs p (GridExecution.atProgram initial pid)) p.inner p.groupSize
          (RealFirst.outputRow p (GridExecution.atProgram initial pid) i) (RealFirst.outputColumn p (GridExecution.atProgram initial pid) i))) ∧
      (∀ r o,(r≠p.output ∨ ∀ pid,pid<Geometry.count (FirstGrid.geometry p) → ∀ i,
        Geometry.Active (FirstGrid.geometry p) pid i → o≠Geometry.offset (FirstGrid.geometry p) pid i) →
        final.mem r o=initial.mem r o) := by
  obtain ⟨final,ran,values,frame⟩ := FirstGrid.whole Real.scalar Real.firstDot p valid ha hb hs hz initial jobs schedule
  refine ⟨final,ran,?_,frame⟩
  intro pid hp i hi
  rw [values pid hp i hi]
  apply congrArg (fun v : ℝ => (Cell.mk .fp16 v : Cell ℝ))
  exact RealFirst.output_formula p (GridExecution.atProgram initial pid) hBK G hG hmode hK i hi
#axiomsClean gptq_kernel_correct

/-- Full raw source grid with independent mathematical outputs and exterior frame. -/
theorem s2_kernel_correct (p : FPTypedSecond.Parameters) (valid : Geometry.Valid (SecondGrid.geometry p))
    (ha : p.input≠p.output) (hb : p.weight≠p.output)
    (hs : p.scale≠p.output) (hz : p.zero≠p.output)
    (hBK : p.blockInner=32) (hsteps : p.steps=(p.inner+31)/32)
    (initial : State ℝ) (jobs : List Nat)
    (schedule : jobs.Perm (List.range (Geometry.count (SecondGrid.geometry p)))) :
    ∃ final,Grid.run (SecondGrid.program Real.scalar Real.secondDot p valid ha hb hs hz) jobs initial=some final ∧
      (∀ pid,pid<Geometry.count (SecondGrid.geometry p) → ∀ i,Geometry.Active (SecondGrid.geometry p) pid i →
        final.mem p.output (Geometry.offset (SecondGrid.geometry p) pid i)=.mk .fp16 (Real.rawMatmul (RealSecond.inputs p (GridExecution.atProgram initial pid)) p.inner p.groupSize
          (RealSecond.outputRow p (GridExecution.atProgram initial pid) i) (RealSecond.outputColumn p (GridExecution.atProgram initial pid) i))) ∧
      (∀ r o,(r≠p.output ∨ ∀ pid,pid<Geometry.count (SecondGrid.geometry p) → ∀ i,
        Geometry.Active (SecondGrid.geometry p) pid i → o≠Geometry.offset (SecondGrid.geometry p) pid i) →
        final.mem r o=initial.mem r o) := by
  obtain ⟨final,ran,values,frame⟩ := SecondGrid.whole Real.scalar Real.secondDot p valid ha hb hs hz initial jobs schedule
  refine ⟨final,ran,?_,frame⟩
  intro pid hp i hi
  rw [values pid hp i hi]
  apply congrArg (fun v : ℝ => (Cell.mk .fp16 v : Cell ℝ))
  exact RealSecond.output_formula p (GridExecution.atProgram initial pid) hBK (by simp [GridExecution.atProgram]) hsteps i hi
#axiomsClean s2_kernel_correct

/-- Full raw source grid with independent mathematical outputs and exterior frame. -/
theorem dequant32_kernel_correct (p : Dequant32.Parameters) (valid : Geometry.Valid (DequantGrid32.geometry p))
    (hb : p.weight≠p.output) (hs : p.scale≠p.output) (hz : p.zero≠p.output)
    (initial : State ℝ) (jobs : List Nat)
    (schedule : jobs.Perm (List.range (Geometry.count (DequantGrid32.geometry p)))) :
    ∃ final,Grid.run (DequantGrid32.program Real.scalar p valid hb hs hz) jobs initial=some final ∧
      (∀ pid,pid<Geometry.count (DequantGrid32.geometry p) → ∀ i,Geometry.Active (DequantGrid32.geometry p) pid i →
        final.mem p.output (Geometry.offset (DequantGrid32.geometry p) pid i)=.mk .fp32 (Real.rawDequant (RealDequant.D32.inputs p (DequantGrid32.atProgram p initial pid)) p.groupSize
          (Dequant32.row p (DequantGrid32.atProgram p initial pid) i) (Dequant32.column p (DequantGrid32.atProgram p initial pid) i))) ∧
      (∀ r o,(r≠p.output ∨ ∀ pid,pid<Geometry.count (DequantGrid32.geometry p) → ∀ i,
        Geometry.Active (DequantGrid32.geometry p) pid i → o≠Geometry.offset (DequantGrid32.geometry p) pid i) →
        final.mem r o=initial.mem r o) := by
  obtain ⟨final,ran,values,frame⟩ := DequantGrid32.whole Real.scalar p valid hb hs hz initial jobs schedule
  refine ⟨final,ran,?_,frame⟩
  intro pid hp i hi
  rw [values pid hp i hi]
  apply congrArg (fun v : ℝ => (Cell.mk .fp32 v : Cell ℝ))
  exact RealDequant.D32.formula p (DequantGrid32.atProgram p initial pid) i
#axiomsClean dequant32_kernel_correct

/-- Full raw source grid with independent mathematical outputs and exterior frame. -/
theorem dequant16_kernel_correct (p : VeriTile.Bench.TritonBenchG.MatmulDequantInt4.FPTyped.Parameters) (valid : Geometry.Valid (DequantGrid16.geometry p))
    (hb : p.weight≠p.output) (hs : p.scale≠p.output) (hz : p.zero≠p.output)
    (initial : State ℝ) (jobs : List Nat)
    (schedule : jobs.Perm (List.range (Geometry.count (DequantGrid16.geometry p)))) :
    ∃ final,Grid.run (DequantGrid16.program Real.scalar p valid hb hs hz) jobs initial=some final ∧
      (∀ pid,pid<Geometry.count (DequantGrid16.geometry p) → ∀ i,Geometry.Active (DequantGrid16.geometry p) pid i →
        final.mem p.output (Geometry.offset (DequantGrid16.geometry p) pid i)=.mk .fp16 (Real.rawDequant (RealDequant.D16.inputs p (DequantGrid16.atProgram p initial pid)) p.groupSize
          (VeriTile.Bench.TritonBenchG.MatmulDequantInt4.FPTyped.row p (DequantGrid16.atProgram p initial pid) i) (VeriTile.Bench.TritonBenchG.MatmulDequantInt4.FPTyped.column p (DequantGrid16.atProgram p initial pid) i))) ∧
      (∀ r o,(r≠p.output ∨ ∀ pid,pid<Geometry.count (DequantGrid16.geometry p) → ∀ i,
        Geometry.Active (DequantGrid16.geometry p) pid i → o≠Geometry.offset (DequantGrid16.geometry p) pid i) →
        final.mem r o=initial.mem r o) := by
  obtain ⟨final,ran,values,frame⟩ := DequantGrid16.whole Real.scalar p valid hb hs hz initial jobs schedule
  refine ⟨final,ran,?_,frame⟩
  intro pid hp i hi
  rw [values pid hp i hi]
  apply congrArg (fun v : ℝ => (Cell.mk .fp16 v : Cell ℝ))
  exact RealDequant.D16.formula p (DequantGrid16.atProgram p initial pid) i
#axiomsClean dequant16_kernel_correct
end
end VeriTile.Bench.Repairs.MatmulDequantize.KernelRealResult
