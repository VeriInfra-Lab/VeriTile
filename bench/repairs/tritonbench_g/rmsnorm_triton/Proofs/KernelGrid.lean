import bench.repairs.tritonbench_g.rmsnorm_triton.Geometry
import bench.repairs.tritonbench_g.rmsnorm_triton.Proofs.TensorSource
import bench.repairs.tritonbench_g.rmsnorm_triton.Proofs.FPObserved

namespace VeriTile.Bench.Repairs.RmsnormTriton.KernelGrid
open VeriTile Triton
open VeriTile.Bench.TritonBenchG.RmsnormTriton.Whole
noncomputable section


def realTarget (p : Parameters) (before : Memory ℝ) (b m k : Nat) : ℝ :=
  mathematical p.width p.eps
    (fun j => before p.input.allocation (inputOffset p b m j))
    (fun j => before p.weights.allocation (weightOffset p j)) k

def nativeTarget (Q : Numerics Torch.FP32.Word) (profile : Symbolic.Profile)
    (p : Parameters) (before : Memory Torch.FP32.Word) (b m k : Nat) : Torch.FP32.Word :=
  Symbolic.observe Q profile (Symbolic.rowLogical p (Symbolic.entryGraph before) b m k)

/-- Shared raw physical-memory grid theorem. Only address projections of p are used. -/
theorem real (red : Reduction) (p : Parameters) (sep : Separation p) (unique : OutputUnique p)
    (u : Nat → ℝ) (schedule : Schedule p) (before : Memory ℝ) :
    ∃ final, TensorSource.grid p schedule.order before=some final ∧
      TensorSource.candidateGrid p schedule.order before=some final ∧
      kernelRun realNumerics red p u schedule before=some final ∧
      (∀ b m k,b<p.batches → m<p.rows → k<p.width →
        final p.output.allocation (outputOffset p b m k)=realTarget p before b m k) ∧
      final=referenceMemory p (realTarget p before) before ∧
      (∀ a o,Outside p a o → final a o=before a o) := by
  let final := executeGrid realNumerics red p u schedule.order before
  obtain ⟨values,frame⟩ := grid_result realNumerics red p sep unique u schedule.order schedule.complete before
  have outputs : ∀ b m k,b<p.batches → m<p.rows → k<p.width →
      final p.output.allocation (outputOffset p b m k)=realTarget p before b m k := by
    intro b m k hb hm hk
    exact (values b m k hb hm hk).trans (logicalValue_real red p.width p.eps _ _ k)
  have source : TensorSource.grid p schedule.order before=some final :=
    TensorSource.scheduled_run red p sep unique u schedule before
  exact ⟨final,source,(TensorSource.candidate_grid p schedule.order before).trans source,
    rfl,outputs,memory_ext p _ before final outputs frame,frame⟩

theorem native (Q : Numerics Torch.FP32.Word) (profile : Symbolic.Profile)
    (p : Parameters) (sep : Separation p) (unique : OutputUnique p)
    (u : Nat → Torch.FP32.Word) (schedule : Schedule p) (before : Memory Torch.FP32.Word) :
    ∃ symbolicFinal final,
      Symbolic.grid p (fun k=>.input (u k)) schedule.order (Symbolic.entryGraph before)=some symbolicFinal ∧
      Symbolic.candidateGrid p (fun k=>.input (u k)) schedule.order (Symbolic.entryGraph before)=some symbolicFinal ∧
      Symbolic.observedGrid Q profile p u schedule.order before=some final ∧
      Symbolic.observedCandidateGrid Q profile p u schedule.order before=some final ∧
      final=Symbolic.projectMemory Q profile symbolicFinal ∧
      (∀ b m k,b<p.batches → m<p.rows → k<p.width →
        final p.output.allocation (outputOffset p b m k)=nativeTarget Q profile p before b m k) ∧
      final=referenceMemory p (nativeTarget Q profile p before) before ∧
      (∀ a o,Outside p a o → final a o=before a o) := by
  let final := Symbolic.gridModel p (fun k=>Symbolic.Term.input (u k)) schedule.order (Symbolic.entryGraph before)
  obtain ⟨values,frame⟩ := Symbolic.grid_result p sep unique (fun k=>Symbolic.Term.input (u k)) schedule (Symbolic.entryGraph before)
  have projected : ∀ b m k,b<p.batches → m<p.rows → k<p.width →
      Symbolic.projectMemory Q profile final p.output.allocation (outputOffset p b m k)=nativeTarget Q profile p before b m k := by
    intro b m k hb hm hk
    exact congrArg (Symbolic.observe Q profile) (values b m k hb hm hk)
  have outside : ∀ a o,Outside p a o → Symbolic.projectMemory Q profile final a o=before a o := by
    intro a o h
    unfold Symbolic.projectMemory
    rw [show final a o=Symbolic.entryGraph before a o from frame a o h]
    rfl
  refine ⟨final,Symbolic.projectMemory Q profile final,Symbolic.grid_run _ _ _ _,?_,?_,?_,rfl,
    projected,memory_ext p _ before _ projected outside,outside⟩
  · rw [Symbolic.candidate_grid]
    exact Symbolic.grid_run _ _ _ _
  · simp only [Symbolic.observedGrid,Symbolic.grid_run,Option.map_some]
    rfl
  · simp only [Symbolic.observedCandidateGrid,Symbolic.candidate_grid,Symbolic.grid_run,Option.map_some]
    rfl

#axiomsClean real
#axiomsClean native
end
end VeriTile.Bench.Repairs.RmsnormTriton.KernelGrid
