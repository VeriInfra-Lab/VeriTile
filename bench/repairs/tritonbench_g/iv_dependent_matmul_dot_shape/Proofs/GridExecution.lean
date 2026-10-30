import bench.repairs.tritonbench_g.iv_dependent_matmul_dot_shape.Proofs.RealProjection
import bench.repairs.tritonbench_g.iv_dependent_matmul.Proofs.ByteHost
import bench.repairs.tritonbench_g.iv_dependent_matmul.Proofs.Instructions

namespace VeriTile.Bench.Repairs.IvDependentMatmulDotShape.Kernel
open VeriTile.Triton FP.Structural
open TritonBenchG.IvDependentMatmul FPValue Whole
open IvDependentMatmul
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 40000

def runPrograms {α : Type} [Inhabited α] (M : Algebra α) (g : Graph) (r : Request) :
    List Nat → State α → Option (State α)
  | [], initial => some initial
  | pid::ids, initial => do
      let next ← exec M (quotation r.schedule g r.parameters)
        {initial with pids := fun ax => if ax=0 then pid else 0}
      runPrograms M g r ids next

def grid {α : Type} [Inhabited α] (M : Algebra α) (g : Graph) (r : Request) (initial : State α) :=
  runPrograms M g r (List.range (programs r)) initial

private theorem scatter_congr {α ι : Type} (items : List ι) (region : RegionName)
    (offset : ι → Nat) (active : ι → Prop) [DecidablePred active]
    (values : ι → Cell α) (s t : State α) (h : s.mem=t.mem) :
    (items.foldl (fun u i => if active i then u.write region (offset i) (values i) else u) s).mem =
    (items.foldl (fun u i => if active i then u.write region (offset i) (values i) else u) t).mem := by
  induction items generalizing s t with
  | nil => exact h
  | cons i tail ih =>
    simp only [List.foldl_cons]
    split <;> apply ih
    · simp only [State.write,h]
    · exact h

theorem finite_memory_congr {α : Type} [Inhabited α] (m : Execution.Arithmetic α r)
    (g : Graph) (ba bb bc pid : Nat) (s t : State α) (hm : s.mem=t.mem) :
    (Execution.program m g ba bb bc pid s).mem=(Execution.program m g ba bb bc pid t).mem := by
  have step : Execution.step m g ba bb pid s=Execution.step m g ba bb pid t := by
    funext k acc
    have ha : Execution.aLoad m ba bb pid k s=Execution.aLoad m ba bb pid k t := by
      funext q; simp only [Execution.aLoad,hm]
    have hb : Execution.bLoad m ba bb pid k s=Execution.bLoad m ba bb pid k t := by
      funext q; simp only [Execution.bLoad,hm]
    cases g <;> simp only [Execution.step,ha,hb]
  have values : Execution.accumulated m g ba bb pid s=Execution.accumulated m g ba bb pid t := by
    simp only [Execution.accumulated,step]
  unfold Execution.program
  rw [values]
  exact scatter_congr _ _ _ _ _ _ _ hm

/-- This induction composes the already proved one-program execution theorem.
The premise is discharged below by expression/loop execution, not supplied by
the public user as a result or output-memory assumption. -/
theorem grid_projection {α : Type} [Inhabited α] (M : Algebra α)
    (m : Execution.Arithmetic α r) (g : Graph) (ba bb bc : Nat)
    (one : ∀ pid, pid<programs r → ∀ initial : State α,
      ∃ final, exec M (quotation r.schedule g r.parameters)
          {initial with pids := fun ax => if ax=0 then pid else 0}=some final ∧
        final.mem=(Execution.program m g ba bb bc pid initial).mem)
    (ids : List Nat) (valid : ∀ pid∈ids,pid<programs r)
    (s t : State α) (hm : s.mem=t.mem) :
    ∃ final, runPrograms M g r ids s=some final ∧
      final.mem=(Execution.runPrograms m g ba bb bc ids t).mem := by
  induction ids generalizing s t with
  | nil => exact ⟨s,rfl,hm⟩
  | cons pid ids ih =>
    obtain ⟨next,he,hn⟩ := one pid (valid pid (by simp)) s
    have memory := hn.trans (finite_memory_congr m g ba bb bc pid s t hm)
    obtain ⟨final,hf,hfm⟩ := ih (fun p hp => valid p (by simp [hp])) next _ memory
    exact ⟨final,by simp only [runPrograms,he,Option.bind,bind,hf],hfm⟩

theorem real_grid (rest : Algebra ℝ) (r : Request) (layout : State Word)
    (a : Allocation r layout) (d : Index64.Addressable r layout a) (l : Index64.Launchable r)
    (initial : State ℝ) :
    ∃ final, grid (RealExecution.backend rest) .source r initial=some final ∧
      final.mem=(Execution.host (Execution.ideal r) .source a.baseA a.baseB a.baseC initial).2.mem := by
  exact grid_projection _ _ _ _ _ _
    (fun pid hp s => RealExecution.quotation_execution rest r layout a d l pid hp s)
    _ (fun _ h => List.mem_range.mp h) initial initial rfl

theorem native_grid [Inhabited Word] (rest : Algebra Word) (cfg : FP.Config) (D : Dot) (F : DotAcc)
    (r : Request) (layout : State Word) (a : Allocation r layout)
    (d : Index64.Addressable r layout a) (l : Index64.Launchable r) (initial : State Word) :
    ∃ final, grid (FPValue.backend rest cfg D F) .accumulator r initial=some final ∧
      final.mem=(Execution.host (Execution.fp cfg D F r) .accumulator a.baseA a.baseB a.baseC initial).2.mem := by
  apply grid_projection _ _ _ _ _ _ _ _ (fun _ h => List.mem_range.mp h) initial initial rfl
  intro pid hp s
  obtain ⟨final,he,hm⟩ := native_execution rest cfg D F r (instanceState pid s)
  exact ⟨final,he,hm.trans (congrArg State.mem
    (Execution.program_projection cfg D F .accumulator r layout s a d l pid hp)).symm⟩

#axiomsClean finite_memory_congr
#axiomsClean grid_projection
#axiomsClean real_grid
#axiomsClean native_grid
end
end VeriTile.Bench.Repairs.IvDependentMatmulDotShape.Kernel
