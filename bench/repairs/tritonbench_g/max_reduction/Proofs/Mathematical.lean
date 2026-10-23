import bench.repairs.tritonbench_g.max_reduction.Proofs.Host
import bench.repairs.tritonbench_g.max_reduction.Proofs.WholeReal
import bench.repairs.tritonbench_g.max_reduction.Proofs.Source

namespace VeriTile.Bench.Repairs.MaxReduction.Mathematical
open Execution
open VeriTile.Torch
noncomputable section
set_option maxHeartbeats 2000000

theorem global_local (algebra : Compute.Algebra A) (count block : Nat)
    (first : Nat → Tree) (second : Tree) (f g : Nat → A)
    (same : ∀ i,i<count → f i=g i) :
    globalReference algebra f count block first second=globalReference algebra g count block first second := by
  apply Compute.scalar_local
  intro p _
  exact Compute.scalar_local algebra (first p) f g count (p*block) same

/-- Complete nonempty global maximum, with actual first-stage writes consumed
by stage two, the returned scalar metadata, and every exterior/input cell. -/
theorem max_real (x : TensorMetadata) (h : GlobalContract x) (profile : Host.GlobalProfile x.numel)
    (input : Nat → ℝ) (s : Heap (WithBot ℝ))
    (binding : ∀ i,i<x.numel → Execution.read ⊥ s "input" i=(input i : WithBot ℝ)) :
    ∃ (answer : ℝ) (result : Host.ScalarResult (WithBot ℝ)),
      Host.max Compute.real x profile s=.ok result ∧
      result.shape=[] ∧ result.dtype=.floating .fp32 ∧ result.device=x.device ∧
      result.heap ("output",0)=.value (answer : WithBot ℝ) ∧
      RealModel.IsMaximum input x.numel answer ∧
      (∀ q,(q.1≠"middle" ∨ globalMid x.numel≤q.2) → q≠("output",0) → result.heap q=s q) := by
  obtain ⟨answer,formula,maximum⟩ := RealModel.global_correct input x.numel (globalBlock x.numel)
    (globalMid x.numel) (globalWidth x.numel) h.2 (global_block_bound x h).1 rfl
    (nextPower_covers _) profile.first profile.second
  let result : Host.ScalarResult (WithBot ℝ) := ⟨[],.floating .fp32,x.device,
    globalRun Compute.real x.numel (globalBlock x.numel) (globalWidth x.numel)
      (fun p => (profile.first p).tree) profile.second.tree s⟩
  refine ⟨answer,result,Host.max_success Compute.real x h profile s,rfl,rfl,rfl,?_,maximum,?_⟩
  · have value := global_result Compute.real x.numel (globalBlock x.numel) (globalWidth x.numel)
      (fun p => (profile.first p).tree) profile.second.tree s
    simp only [show Compute.real.padding=(⊥ : WithBot ℝ) from rfl] at value
    rw [global_local Compute.real _ _ _ _ _ (fun i => (input i : WithBot ℝ)) binding] at value
    have eq : globalReference Compute.real (fun i => (input i : WithBot ℝ)) x.numel (globalBlock x.numel)
        (fun p => (profile.first p).tree) profile.second.tree=
      RealModel.global input x.numel (globalBlock x.numel) (globalMid x.numel)
        (fun p => (profile.first p).tree) profile.second.tree := by
      simp [globalReference,Compute.scalar,scalarReduction,Compute.real,RealModel.global,
        RealModel.firstStage,RealModel.valueReduction,globalMid,Address.count]
    rw [eq,formula] at value
    exact value
  · intro q hm ho
    exact global_frame Compute.real x.numel (globalBlock x.numel) (globalWidth x.numel)
      (fun p => (profile.first p).tree) profile.second.tree s q hm ho

/-- Complete max_dim: every value and exact int64 least attaining index,
negative-dimension normalization, both keepdim branches, original contiguous
input semantics, all original autotuning probes, empty outputs and full frame. -/
theorem max_dim_real (a : DimArguments) (h : DimContract a)
    (trials : List (Host.DimProfile a.N)) (selected : Host.DimProfile a.N)
    (input : Nat → ℝ) (s : Heap (WithBot ℝ))
    (binding : ∀ i,i<a.input.numel → Execution.read ⊥ s "input" i=(input i : WithBot ℝ)) :
    ∃ result : Host.DimensionResult (WithBot ℝ),Host.maxDim Compute.real a trials selected s=.ok result ∧
      result.shape=a.outputShape ∧ result.valuesDtype=.floating .fp32 ∧
      result.indicesDtype=.other "int64" ∧ result.device=a.input.device ∧
      (∀ j,j<a.M*a.K → ∃ (answer : ℝ) (index : Nat),
        result.heap ("values",j)=.value (answer : WithBot ℝ) ∧
        result.heap ("indices",j)=.index (BitVec.ofNat 64 index) ∧
        RealModel.IsLeastArgmax (fun n => input (j/a.K*a.N*a.K+n*a.K+j%a.K)) a.N answer index) ∧
      (∀ q,(q.1≠"values" ∧ q.1≠"indices" ∨ a.M*a.K≤q.2) → result.heap q=s q) := by
  let result : Host.DimensionResult (WithBot ℝ) := ⟨a.outputShape,.floating .fp32,.other "int64",a.input.device,
    dimensionRun Compute.real selected.reduction.tree a.M a.N a.K selected.blockM
      (Host.probes Compute.real a.M a.N a.K trials s)⟩
  refine ⟨result,Host.maxDim_success _ a h trials selected s,rfl,rfl,rfl,rfl,?_,?_⟩
  · intro j hj
    have values := Host.maxDim_values Compute.real a trials selected s j hj
    simp only [show Compute.real.padding=(⊥ : WithBot ℝ) from rfl] at values
    have sameInput := Compute.dimension_local Compute.real selected.reduction.tree a.M a.N a.K
      (Execution.read ⊥ s "input") (fun i => (input i : WithBot ℝ))
      (fun i hi => binding i (by simpa [←dimension_numel a h] using hi)) j hj
    rw [sameInput] at values
    obtain ⟨answer,index,eq,argmax⟩ := Compute.dimension_real input a.N a.K j (nextPower a.N)
      h.2.2.2.1 (nextPower_covers _) selected.reduction
    rw [eq] at values
    have indexBound : index<2^31 := lt_of_lt_of_le argmax.1 (by have := h.2.2.2.2.1; omega)
    rw [(Machine.stored_index_exact index indexBound).1] at values
    exact ⟨answer,index,values.1,values.2,argmax⟩
  · intro q miss
    exact Host.maxDim_frame _ _ _ _ _ q miss

#axiomsClean max_real
#axiomsClean max_dim_real
end
end VeriTile.Bench.Repairs.MaxReduction.Mathematical
