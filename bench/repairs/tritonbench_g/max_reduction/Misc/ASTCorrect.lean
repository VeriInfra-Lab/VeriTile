import bench.repairs.tritonbench_g.max_reduction.Proofs.Mathematical
import bench.repairs.tritonbench_g.max_reduction.Proofs.SourceGlobal
import bench.repairs.tritonbench_g.max_reduction.Proofs.KernelMath

namespace VeriTile.Bench.Repairs.MaxReduction.ASTCorrect
open VeriTile.Triton FP.Structural SourceGrid
noncomputable section
open Classical
set_option maxHeartbeats 3000000

export KernelMath (engine dimension_target target_local target_real)

theorem global_target (algebra : Compute.Algebra A) (N B W : Nat)
    (first : ReductionPlan B) (second : ReductionPlan W) (input : Nat → A) :
    globalTarget (engine algebra first.tree) (engine algebra second.tree) input N B W=
      Execution.globalReference algebra input N B (fun _ => first.tree) second.tree := by
  apply Compute.tree_congr
  intro p hp
  have width := (second.complete p).mp hp
  simp only [width,dif_pos,Nat.zero_add]
  split_ifs with active
  · exact Source.scalar_tree algebra (fun _ => first.tree) B
      (fun i hi => (first.complete i).mp hi) input N (p*B)
  · rfl

/-- Whole two-kernel AST theorem against the independent original-input maximum. -/
theorem max_real_ast (x : VeriTile.Torch.TensorMetadata) (h : GlobalContract x)
    (first : ReductionPlan (globalBlock x.numel)) (second : ReductionPlan (globalWidth x.numel))
    (input : Nat → ℝ) (s : State (WithBot ℝ))
    (binding : ∀ i,i<x.numel → (s.mem "input" i).read .real=(input i : WithBot ℝ)) :
    ∃ (t : State (WithBot ℝ)) (answer : ℝ),runGlobal (engine Compute.real first.tree) (engine Compute.real second.tree)
        x.numel (globalBlock x.numel) (globalWidth x.numel) s=some t ∧
      t.mem "output" 0=.mk .real (answer : WithBot ℝ) ∧ RealModel.IsMaximum input x.numel answer ∧
      (∀ (r : RegionName) (i : Nat), (r≠"middle" ∨ globalMid x.numel ≤ i) →
        (r≠"output" ∨ i≠0) → t.mem r i=s.mem r i) := by
  obtain ⟨t,he,hv,hf⟩ := global_whole (engine Compute.real first.tree) (engine Compute.real second.tree)
    x.numel (globalBlock x.numel) (globalWidth x.numel) (global_block_bound x h).1 (nextPower_positive _) s
  rw [global_target Compute.real _ _ _ first second] at hv
  rw [Mathematical.global_local Compute.real _ _ _ _ _ (fun i => (input i : WithBot ℝ)) binding] at hv
  obtain ⟨answer,eq,maximum⟩ := RealModel.global_correct input x.numel (globalBlock x.numel)
    (globalMid x.numel) (globalWidth x.numel) h.2 (global_block_bound x h).1 rfl
    (nextPower_covers _) (fun _ => first) second
  have eq' : Execution.globalReference Compute.real (fun i => (input i : WithBot ℝ)) x.numel
      (globalBlock x.numel) (fun _ => first.tree) second.tree=(answer : WithBot ℝ) := by
    simpa [Execution.globalReference,Compute.scalar,scalarReduction,Compute.real,RealModel.global,
      RealModel.firstStage,RealModel.valueReduction,globalMid,Address.count] using eq
  rw [eq'] at hv
  exact ⟨t,answer,he,hv,maximum,hf⟩

/-- Whole dimension AST with every value, least index and complete memory frame. -/
theorem max_dim_real_ast (a : DimArguments) (h : DimContract a) (profile : Host.DimProfile a.N)
    (input : Nat → ℝ) (s : State (WithBot ℝ))
    (binding : ∀ i,i<a.input.numel → (s.mem "input" i).read .real=(input i : WithBot ℝ)) :
    ∃ t,runDimension (engine Compute.real profile.reduction.tree) a.M a.N a.K profile.blockM (nextPower a.N)
        (Grid.programs a.M a.K profile.blockM) s=some t ∧
      (∀ j,j<a.M*a.K → ∃ (answer : ℝ) (index : Nat),
        t.mem "values" j=.mk .real (answer : WithBot ℝ) ∧ t.mem "indices" j=.mk .int (index : Int) ∧
        RealModel.IsLeastArgmax (fun n => input (j/a.K*a.N*a.K+n*a.K+j%a.K)) a.N answer index) ∧
      (∀ (r : RegionName) (i : Nat), (r≠"values" ∧ r≠"indices" ∨ a.M*a.K ≤ i) → t.mem r i=s.mem r i) := by
  obtain ⟨t,he,hv,hf⟩ := dimension_whole (engine Compute.real profile.reduction.tree)
    a.M a.N a.K profile.blockM (nextPower a.N) (block_choice _ profile.choice).1 (nextPower_positive _) s
  refine ⟨t,he,?_,hf⟩
  intro j hj
  have values := hv j hj
  rw [target_local _ a.M a.N a.K _ j hj _ (fun i => (input i : WithBot ℝ))
    (fun i hi => binding i (by simpa [←dimension_numel a h] using hi))] at values
  obtain ⟨answer,index,eq,maximum⟩ := target_real input a.N a.K (nextPower a.N) j h.2.2.2.1
    (nextPower_covers _) profile.reduction
  rw [eq] at values
  exact ⟨answer,index,values.1,values.2,maximum⟩

/-- Whole source AST native theorem; int64 storage preserves the reducer's
signed-int32 chosen index, even if unordered comparisons select padding. -/
theorem max_dim_native_ast (ops : NativeOps) (a : DimArguments) (h : DimContract a)
    (profile : Host.DimProfile a.N) (s : State Word) :
    ∃ t,runDimension (engine (Compute.native ops) profile.reduction.tree) a.M a.N a.K profile.blockM (nextPower a.N)
        (Grid.programs a.M a.K profile.blockM) s=some t ∧
      (∀ j,j<a.M*a.K →
        t.mem "values" j=.mk .real (nativeDimension ops profile.reduction.tree
          (fun i => (s.mem "input" i).read .real) a.N a.K (j/a.K) (j%a.K)).1 ∧
        t.mem "indices" j=.mk .int (nativeDimension ops profile.reduction.tree
          (fun i => (s.mem "input" i).read .real) a.N a.K (j/a.K) (j%a.K)).2.toInt) ∧
      (∀ (r : RegionName) (i : Nat), (r≠"values" ∧ r≠"indices" ∨ a.M*a.K ≤ i) → t.mem r i=s.mem r i) := by
  obtain ⟨t,he,hv,hf⟩ := dimension_whole (engine (Compute.native ops) profile.reduction.tree)
    a.M a.N a.K profile.blockM (nextPower a.N) (block_choice _ profile.choice).1 (nextPower_positive _) s
  refine ⟨t,he,?_,hf⟩
  intro j hj
  have values := hv j hj
  rw [dimension_target (Compute.native ops) a.N a.K _ j profile.reduction] at values
  have bound : (profile.reduction.tree.eval (nativePair ops)
      (fun n => (if n<a.N then (s.mem "input" (j/a.K*a.N*a.K+n*a.K+j%a.K)).read .real else negativeInfinity,n))).2<2^31 := by
    apply Machine.selected_index_bound
    intro n hn
    have nl := (profile.reduction.complete n).mp hn
    have cap : nextPower a.N≤32768 := nextPower_bound _ 15 h.2.2.2.2.1
    omega
  simpa only [nativeDimension,(Machine.stored_index_exact _ bound).2] using values

#axiomsClean max_real_ast
#axiomsClean max_dim_real_ast
#axiomsClean max_dim_native_ast
end
end VeriTile.Bench.Repairs.MaxReduction.ASTCorrect
