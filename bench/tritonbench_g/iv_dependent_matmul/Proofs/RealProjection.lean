import bench.tritonbench_g.iv_dependent_matmul.Proofs.RealSchedules
import bench.tritonbench_g.iv_dependent_matmul.Proofs.SourceBridge
import bench.tritonbench_g.iv_dependent_matmul.Proofs.Ideal

namespace VeriTile.Bench.Repairs.IvDependentMatmulDotShape.RealExecution
open VeriTile.Triton FP.Structural
open TritonBenchG.IvDependentMatmul FPValue Whole
open IvDependentMatmul
open TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 40000

def instanceState (pid : Nat) (s : State ℝ) : State ℝ :=
  {s with pids := fun ax => if ax=0 then pid else 0}

theorem a_operand (r : Request) (initial : State Word) (current : State ℝ) (a : Allocation r initial)
    (d : Index64.Addressable r initial a) (l : Index64.Launchable r)
    (pid k : Nat) (hp : pid<programs r) (hk : k<count r.parameters) :
    IvDependentMatmul.Execution.aLoad (IvDependentMatmul.Execution.ideal r) a.baseA a.baseB pid k current=
      aValues r.parameters (instanceState pid current) k := by
  have h := Native.correspondence r initial a d l
  funext i
  simp only [IvDependentMatmul.Execution.aLoad,(h.stage pid k).1,h.kMask pid k hp hk i.2.1,decide_eq_true_eq]
  by_cases hm : i.2.1.val<r.inner-k*r.blockInner
  · rw [if_pos hm,h.activeA pid k hp i hm,IvDependentMatmul.Execution.decode_address]
    simp only [aValues,Request.parameters,if_pos hm]
    rfl
  · simp only [if_neg hm,aValues,Request.parameters,IvDependentMatmul.Execution.ideal]

theorem b_operand (r : Request) (initial : State Word) (current : State ℝ) (a : Allocation r initial)
    (d : Index64.Addressable r initial a) (l : Index64.Launchable r)
    (pid k : Nat) (hp : pid<programs r) (hk : k<count r.parameters) :
    IvDependentMatmul.Execution.bLoad (IvDependentMatmul.Execution.ideal r) a.baseA a.baseB pid k current=
      bValues r.parameters (instanceState pid current) k := by
  have h := Native.correspondence r initial a d l
  funext i
  simp only [IvDependentMatmul.Execution.bLoad,(h.stage pid k).2,h.kMask pid k hp hk i.1,decide_eq_true_eq]
  by_cases hm : i.1.val<r.inner-k*r.blockInner
  · rw [if_pos hm,h.activeB pid k hp i hm,IvDependentMatmul.Execution.decode_address]
    simp only [bValues,Request.parameters,if_pos hm]
    rfl
  · simp only [if_neg hm,bValues,Request.parameters,IvDependentMatmul.Execution.ideal]

theorem step_projection (g : Graph)
    (r : Request) (initial : State Word) (current : State ℝ) (a : Allocation r initial)
    (d : Index64.Addressable r initial a) (l : Index64.Launchable r)
    (pid k : Nat) (hp : pid<programs r) (hk : k<count r.parameters) (acc : IvDependentMatmul.Execution.Acc ℝ r) :
    IvDependentMatmul.Execution.step (IvDependentMatmul.Execution.ideal r) g a.baseA a.baseB pid current k acc=
      update g r.parameters (instanceState pid current) k acc := by
  cases g <;> simp only [IvDependentMatmul.Execution.step]
  all_goals rw [a_operand r initial current a d l pid k hp hk,
    b_operand r initial current a d l pid k hp hk]
  all_goals simp only [IvDependentMatmul.Execution.ideal,update,sumDot]
  all_goals funext q
  all_goals congr 1
  all_goals apply Finset.sum_congr rfl
  all_goals intro j _
  all_goals rfl

private theorem range_projection (g : Graph)
    (r : Request) (initial : State Word) (current : State ℝ) (a : Allocation r initial)
    (d : Index64.Addressable r initial a) (l : Index64.Launchable r)
    (pid cur : Nat) (hp : pid<programs r) (acc : IvDependentMatmul.Execution.Acc ℝ r) :
    foldRange (IvDependentMatmul.Execution.step (IvDependentMatmul.Execution.ideal r) g a.baseA a.baseB pid current) cur (count r.parameters) 1 acc=
      foldRange (update g r.parameters (instanceState pid current)) cur (count r.parameters) 1 acc := by
  by_cases hk : cur<count r.parameters
  · conv_lhs => rw [foldRange]
    conv_rhs => rw [foldRange]
    simp only [Nat.one_ne_zero,hk,↓reduceDIte]
    rw [step_projection g r initial current a d l pid cur hp hk acc]
    exact range_projection g r initial current a d l pid (cur+1) hp _
  · conv_lhs => rw [foldRange]
    conv_rhs => rw [foldRange]
    simp only [Nat.one_ne_zero,hk,↓reduceDIte]
termination_by count r.parameters-cur
decreasing_by omega

theorem accumulated_projection (g : Graph)
    (r : Request) (initial : State Word) (current : State ℝ) (a : Allocation r initial)
    (d : Index64.Addressable r initial a) (l : Index64.Launchable r)
    (pid : Nat) (hp : pid<programs r) :
    IvDependentMatmul.Execution.accumulated (IvDependentMatmul.Execution.ideal r) g a.baseA a.baseB pid current=
      accumulated g r.parameters (instanceState pid current) := by
  have h := Native.correspondence r initial a d l
  have hc := (Index64.counts_control r initial a d l pid hp).2.2
  unfold IvDependentMatmul.Execution.accumulated accumulated
  rw [IvDependentMatmul.Execution.sourceRange_eq]
  rw [h.counter pid hp,Native.unsigned_word _ (by omega)]
  exact range_projection g r initial current a d l pid 0 hp _

theorem program_projection (g : Graph)
    (r : Request) (initial : State Word) (current : State ℝ) (a : Allocation r initial)
    (d : Index64.Addressable r initial a) (l : Index64.Launchable r)
    (pid : Nat) (hp : pid<programs r) :
    IvDependentMatmul.Execution.program (IvDependentMatmul.Execution.ideal r) g a.baseA a.baseB a.baseC pid current=
      written r.parameters (instanceState pid current)
        (accumulated g r.parameters (instanceState pid current)) := by
  have h := Native.correspondence r initial a d l
  unfold IvDependentMatmul.Execution.program
  rw [accumulated_projection g r initial current a d l pid hp]
  unfold written
  dsimp only
  apply congrArg (fun f : State ℝ → CIndex r → State ℝ =>
    (TileShape.allIndices [r.blockRows,r.blockColumns]).foldl f (instanceState pid current))
  funext t q
  rw [h.cMask pid hp q]
  have mask : active r.parameters (instanceState pid current) q=FPValue.active r.parameters (Whole.instanceState pid initial) q := rfl
  rw [mask]
  by_cases hm : FPValue.active r.parameters (Whole.instanceState pid initial) q=Bool.true
  · simp [hm]
    rw [h.activeC pid hp q hm,IvDependentMatmul.Execution.decode_address]
    rfl
  · simp [hm]


theorem quotation_execution (rest : Algebra ℝ) (r : Request) (layout : State Word)
    (a : Allocation r layout) (d : Index64.Addressable r layout a) (l : Index64.Launchable r)
    (pid : Nat) (hp : pid<programs r) (initial : State ℝ) :
    ∃ final, exec (backend rest) (Kernel.source r) (instanceState pid initial)=some final ∧
      final.mem=(IvDependentMatmul.Execution.program (IvDependentMatmul.Execution.ideal r)
        .source a.baseA a.baseB a.baseC pid initial).mem := by
  rw [Kernel.source, Kernel.quotation_eq]
  obtain ⟨final, he, hm⟩ := execution_memory rest r.schedule .source r.parameters (instanceState pid initial)
  exact ⟨final, he, hm.trans (congrArg State.mem
    (program_projection .source r layout initial a d l pid hp)).symm⟩

#axiomsClean a_operand
#axiomsClean accumulated_projection
#axiomsClean program_projection
#axiomsClean quotation_execution
end
end VeriTile.Bench.Repairs.IvDependentMatmulDotShape.RealExecution
