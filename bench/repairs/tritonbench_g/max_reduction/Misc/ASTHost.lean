import bench.repairs.tritonbench_g.max_reduction.Misc.ASTCorrect

namespace VeriTile.Bench.Repairs.MaxReduction.ASTHost
open VeriTile.Triton FP.Structural SourceGrid ASTCorrect
noncomputable section
open Classical

def runTrials [Inhabited A] (algebra : Compute.Algebra A) (a : DimArguments) :
    List (Host.DimProfile a.N) → State A → Option (State A)
  | [],s => some s
  | p::rest,s => do
      let t ← runDimension (engine algebra p.reduction.tree) a.M a.N a.K p.blockM (nextPower a.N)
        (Grid.programs a.M a.K p.blockM) s
      runTrials algebra a rest t

theorem trial_frame [Inhabited A] (algebra : Compute.Algebra A) (a : DimArguments)
    (profiles : List (Host.DimProfile a.N)) (s : State A) :
    ∃ t,runTrials algebra a profiles s=some t ∧
      (∀ (r : RegionName) (i : Nat), (r≠"values" ∧ r≠"indices" ∨ a.M*a.K ≤ i) → t.mem r i=s.mem r i) := by
  induction profiles generalizing s with
  | nil => exact ⟨s,rfl,by intros; rfl⟩
  | cons p rest ih =>
    obtain ⟨mid,hm,_,hf⟩ := dimension_whole (engine algebra p.reduction.tree)
      a.M a.N a.K p.blockM (nextPower a.N) (block_choice _ p.choice).1 (nextPower_positive _) s
    obtain ⟨t,ht,tf⟩ := ih mid
    exact ⟨t,by simp only [runTrials,hm]; exact ht,
      fun r i miss => (tf r i miss).trans (hf r i miss)⟩

def runAutotuned [Inhabited A] (algebra : Compute.Algebra A) (a : DimArguments)
    (profiles : List (Host.DimProfile a.N)) (selected : Host.DimProfile a.N) (s : State A) : Option (State A) := do
  let reached ← runTrials algebra a profiles s
  runDimension (engine algebra selected.reduction.tree) a.M a.N a.K selected.blockM (nextPower a.N)
    (Grid.programs a.M a.K selected.blockM) reached

/-- Whole source call including all autotuning probes, the selected kernel,
all output cells and frames. Metadata return is Host.maxDim_success. -/
theorem max_dim_native (ops : NativeOps) (a : DimArguments) (h : DimContract a)
    (profiles : List (Host.DimProfile a.N)) (selected : Host.DimProfile a.N) (s : State Word) :
    ∃ t,runAutotuned (Compute.native ops) a profiles selected s=some t ∧
      (∀ j,j<a.M*a.K →
        t.mem "values" j=.mk .real (nativeDimension ops selected.reduction.tree
          (fun i => (s.mem "input" i).read .real) a.N a.K (j/a.K) (j%a.K)).1 ∧
        t.mem "indices" j=.mk .int (nativeDimension ops selected.reduction.tree
          (fun i => (s.mem "input" i).read .real) a.N a.K (j/a.K) (j%a.K)).2.toInt) ∧
      (∀ (r : RegionName) (i : Nat), (r≠"values" ∧ r≠"indices" ∨ a.M*a.K ≤ i) → t.mem r i=s.mem r i) := by
  obtain ⟨mid,hm,hf⟩ := trial_frame (Compute.native ops) a profiles s
  obtain ⟨t,ht,hv,tf⟩ := max_dim_native_ast ops a h selected mid
  have input : (fun i => (mid.mem "input" i).read .real)=(fun i => (s.mem "input" i).read .real) := by
    funext i
    rw [hf "input" i (Or.inl ⟨by decide,by decide⟩)]
  rw [input] at hv
  exact ⟨t,by simp only [runAutotuned,hm]; exact ht,hv,
    fun r i miss => (tf r i miss).trans (hf r i miss)⟩

theorem max_dim_real (a : DimArguments) (h : DimContract a)
    (profiles : List (Host.DimProfile a.N)) (selected : Host.DimProfile a.N)
    (input : Nat → ℝ) (s : State (WithBot ℝ))
    (binding : ∀ i,i<a.input.numel → (s.mem "input" i).read .real=(input i : WithBot ℝ)) :
    ∃ t,runAutotuned Compute.real a profiles selected s=some t ∧
      (∀ j,j<a.M*a.K → ∃ (answer : ℝ) (index : Nat),
        t.mem "values" j=.mk .real (answer : WithBot ℝ) ∧ t.mem "indices" j=.mk .int (index : Int) ∧
        RealModel.IsLeastArgmax (fun n => input (j/a.K*a.N*a.K+n*a.K+j%a.K)) a.N answer index) ∧
      (∀ (r : RegionName) (i : Nat), (r≠"values" ∧ r≠"indices" ∨ a.M*a.K ≤ i) → t.mem r i=s.mem r i) := by
  obtain ⟨mid,hm,hf⟩ := trial_frame Compute.real a profiles s
  have pins : ∀ i,i<a.input.numel → (mid.mem "input" i).read .real=(input i : WithBot ℝ) := by
    intro i hi
    rw [hf "input" i (Or.inl ⟨by decide,by decide⟩)]
    exact binding i hi
  obtain ⟨t,ht,hv,tf⟩ := max_dim_real_ast a h selected input mid pins
  exact ⟨t,by simp only [runAutotuned,hm]; exact ht,hv,
    fun r i miss => (tf r i miss).trans (hf r i miss)⟩

theorem max_native (ops : NativeOps) (x : VeriTile.Torch.TensorMetadata) (h : GlobalContract x)
    (first : ReductionPlan (globalBlock x.numel)) (second : ReductionPlan (globalWidth x.numel)) (s : State Word) :
    ∃ t,runGlobal (engine (Compute.native ops) first.tree) (engine (Compute.native ops) second.tree)
        x.numel (globalBlock x.numel) (globalWidth x.numel) s=some t ∧
      t.mem "output" 0=.mk .real (Execution.globalReference (Compute.native ops)
        (fun i => (s.mem "input" i).read .real) x.numel (globalBlock x.numel) (fun _ => first.tree) second.tree) ∧
      (∀ (r : RegionName) (i : Nat), (r≠"middle" ∨ globalMid x.numel ≤ i) →
        (r≠"output" ∨ i≠0) → t.mem r i=s.mem r i) := by
  obtain ⟨t,he,hv,hf⟩ := global_whole (engine (Compute.native ops) first.tree) (engine (Compute.native ops) second.tree)
    x.numel (globalBlock x.numel) (globalWidth x.numel) (global_block_bound x h).1 (nextPower_positive _) s
  rw [global_target (Compute.native ops) _ _ _ first second] at hv
  exact ⟨t,he,hv,hf⟩

#axiomsClean max_dim_native
#axiomsClean max_dim_real
#axiomsClean max_native
end
end VeriTile.Bench.Repairs.MaxReduction.ASTHost
