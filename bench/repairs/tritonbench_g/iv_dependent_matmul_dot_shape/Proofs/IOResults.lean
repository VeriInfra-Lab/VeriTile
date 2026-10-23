import bench.repairs.tritonbench_g.iv_dependent_matmul_dot_shape.Proofs.FormulaCongruence
import bench.repairs.tritonbench_g.iv_dependent_matmul_dot_shape.Proofs.KernelRealResult
import bench.repairs.tritonbench_g.iv_dependent_matmul_dot_shape.Proofs.KernelNativeResult

namespace VeriTile.Bench.Repairs.IvDependentMatmulDotShape.IOResults
open VeriTile.Triton FP.Structural
open TritonBenchG.IvDependentMatmul FPValue Whole
open IvDependentMatmul IOBinding
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 40000
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

/-- The historical layout premise supplies exactly the new source-free public
geometry. Its independent Real snapshot remains unrestricted. -/
def ofAllocation (g : Kernel.Geometry) (s : State Word) (a : Allocation g.request s)
    (d : Index64.Addressable g.request s a) : IO.Layout g :=
  ⟨a.freshA,a.freshB,a.baseA,a.baseB,a.baseC,a.physicalFreshA,a.physicalFreshB,
    a.alignA,a.alignB,a.alignC,d.input,d.weight,d.output⟩

theorem real (rest : Algebra ℝ) (g : Kernel.Geometry) (L : IO.Layout g)
    (l : Index64.Launchable g.request) :
    (IO.matmulIO (RealExecution.backend rest) .source g).Implements
      (fun vals => IO.result g (Target.mathematical g.inner (IO.matrixA g vals) (IO.matrixB g vals))) := by
  apply ProgramIO.Implements.intro
  · intro vals bounds hb initial _
    exact real_schedule_safe rest g .source bounds (windows _ _ _ _ _ hb) _
      (fun _ h => List.mem_range.mp h) initial
  · intro vals initial hin
    let layout := SourceSafety.geometryState
    have ha : ∀ o<g.rows*g.inner, ∃ v, layout.mem g.A o=Cell.mk .fp32 v :=
      fun _ _ => ⟨_,rfl⟩
    have hb : ∀ o<g.inner*g.columns, ∃ v, layout.mem g.B o=Cell.mk .fp32 v :=
      fun _ _ => ⟨_,rfl⟩
    let a := allocation g L layout ha hb
    obtain ⟨final,ran,values,frame⟩ := KernelRealResult.kernel_real rest g layout a
      (addressable g L layout ha hb) l initial
    refine ⟨final,(schedule_run _ _ _ _ _).trans ran,?_,?_⟩
    · intro o j _
      fin_cases o
      have value := values j.1.val j.1.isLt j.2.1.val j.2.1.isLt
      change final.mem g.C (max g.columns 1*j.1.val+j.2.1.val)=Cell.mk .fp32
        (RealTarget.mathematical g.inner (Ideal.inputA g.request initial)
          (Ideal.inputB g.request initial) j.1.val j.2.1.val) at value
      rw [Nat.mul_comm (max g.columns 1) j.1.val] at value
      rw [real_formula rest g vals initial hin _ _ j.1.isLt j.2.1.isLt] at value
      simpa [IO.matmulIO,IO.outputs,IO.result,mapCell,mapValue] using congrArg (mapCell id) value
    · intro r o ho
      exact frame r o (outside _ _ _ _ r o ho)

theorem native (rest : Algebra Word) (profile : Target.NativeProfile) (g : Kernel.Geometry)
    (L : IO.Layout g) (l : Index64.Launchable g.request) :
    (IO.matmulIO (KernelNativeResult.backend rest profile) .accumulator g).ImplementsCells
      (fun vals => IO.result g (Target.native profile.reference g.inner g.blockInner
        (IO.matrixA g vals) (IO.matrixB g vals))) := by
  apply ProgramIO.ImplementsCells.intro
  · intro vals bounds hb initial _
    exact native_schedule_safe rest _ _ _ g .accumulator bounds (windows _ _ _ _ _ hb) _
      (fun _ h => List.mem_range.mp h) initial
  · intro vals initial hin
    have tags := input_cells _ _ _ _ _ hin
    let a := allocation g L initial tags.1 tags.2
    obtain ⟨final,ran,values,frame⟩ := KernelNativeResult.kernel_native rest profile g initial a
      (addressable g L initial tags.1 tags.2) l
    refine ⟨final,(schedule_run _ _ _ _ _).trans ran,?_,?_⟩
    · intro o j _
      fin_cases o
      have value := values j.1.val j.1.isLt j.2.1.val j.2.1.isLt
      change final.mem g.C (max g.columns 1*j.1.val+j.2.1.val)=Cell.mk .fp32
        (Target.native profile.reference g.inner g.blockInner (Target.inputA g.request initial)
          (Target.inputB g.request initial) j.1.val j.2.1.val) at value
      rw [Nat.mul_comm (max g.columns 1) j.1.val] at value
      rw [native_formula _ g vals initial hin.toInputBinding _ _ _ j.1.isLt j.2.1.isLt] at value
      simpa [IO.matmulIO,IO.outputs,IO.result,mapCell,mapValue] using congrArg (mapCell id) value
    · intro r o ho
      exact frame r o (outside _ _ _ _ r o ho)

#axiomsClean real
#axiomsClean native
end
end VeriTile.Bench.Repairs.IvDependentMatmulDotShape.IOResults
