import bench.repairs.tritonbench_g.max_reduction.Proofs.Execution
import bench.repairs.tritonbench_g.max_reduction.Proofs.Machine

namespace VeriTile.Bench.Repairs.MaxReduction.Host
open VeriTile.Torch
open Execution
noncomputable section

/-- Backend tree selection is internal to a launched reduction, not a user
input restriction. The axis width is exactly the original next-power heuristic. -/
structure DimProfile (N : Nat) where
  blockM : Nat
  choice : blockM∈blockChoices
  reduction : ReductionPlan (nextPower N)

structure GlobalProfile (count : Nat) where
  first : Nat → ReductionPlan (globalBlock count)
  second : ReductionPlan (globalWidth count)

/-- Autotuning may run any sequence of the original configurations, followed
by the selected launch. Every launch reads the same original contiguous input. -/
def probes (algebra : Compute.Algebra A) (M N K : Nat) : List (DimProfile N) → Heap A → Heap A
  | [],s => s
  | p::rest,s => probes algebra M N K rest
      (dimensionRun algebra p.reduction.tree M N K p.blockM s)

theorem probes_frame (algebra : Compute.Algebra A) (M N K : Nat)
    (runs : List (DimProfile N)) (s : Heap A) (q : String × Nat)
    (miss : q.1≠"values" ∧ q.1≠"indices" ∨ M*K≤q.2) :
    probes algebra M N K runs s q=s q := by
  induction runs generalizing s with
  | nil => rfl
  | cons p rest ih =>
    rw [probes,ih]
    exact dimension_frame _ _ _ _ _ _ (block_choice _ p.choice).1 _ _ miss

theorem probes_input (algebra : Compute.Algebra A) (M N K : Nat)
    (runs : List (DimProfile N)) (s : Heap A) :
    read algebra.padding (probes algebra M N K runs s) "input"=read algebra.padding s "input" := by
  funext i
  simp only [Execution.read,probes_frame algebra M N K runs s ("input",i) (Or.inl ⟨by simp,by simp⟩)]

structure ScalarResult (A : Type) where
  shape : List Nat
  dtype : DType
  device : Device
  heap : Heap A

structure DimensionResult (A : Type) where
  shape : List Nat
  valuesDtype : DType
  indicesDtype : DType
  device : Device
  heap : Heap A

/-- Successful allocation and owner-device launch are library boundaries.
Fresh region identities represent Torch empty allocations. No result is a
library premise, and the actual reached memory is returned. -/
def max (algebra : Compute.Algebra A) (x : TensorMetadata)
    (profile : GlobalProfile x.numel) (s : Heap A) : Except InputError (ScalarResult A) := do
  checkGlobal (.tensor x)
  return ⟨[],.floating .fp32,x.device,
    globalRun algebra x.numel (globalBlock x.numel) (globalWidth x.numel)
      (fun p => (profile.first p).tree) profile.second.tree s⟩

/-- The input region is the result of the original inp.contiguous() call.
The ordinary contiguous API law binds it to the caller's logical coordinates.
keepdim changes metadata only; both result allocations share the reached heap. -/
def maxDim (algebra : Compute.Algebra A) (a : DimArguments)
    (trials : List (DimProfile a.N)) (selected : DimProfile a.N) (s : Heap A) :
    Except InputError (DimensionResult A) := do
  checkDimension a
  return ⟨a.outputShape,.floating .fp32,.other "int64",a.input.device,
    dimensionRun algebra selected.reduction.tree a.M a.N a.K selected.blockM
      (probes algebra a.M a.N a.K trials s)⟩

theorem max_success (algebra : Compute.Algebra A) (x : TensorMetadata)
    (h : GlobalContract x) (profile : GlobalProfile x.numel) (s : Heap A) :
    max algebra x profile s=.ok ⟨[],.floating .fp32,x.device,
      globalRun algebra x.numel (globalBlock x.numel) (globalWidth x.numel)
        (fun p => (profile.first p).tree) profile.second.tree s⟩ := by
  simp [max,(check_global_iff x).mpr h]

theorem maxDim_success (algebra : Compute.Algebra A) (a : DimArguments)
    (h : DimContract a) (trials : List (DimProfile a.N)) (selected : DimProfile a.N) (s : Heap A) :
    maxDim algebra a trials selected s=.ok ⟨a.outputShape,.floating .fp32,.other "int64",a.input.device,
      dimensionRun algebra selected.reduction.tree a.M a.N a.K selected.blockM
        (probes algebra a.M a.N a.K trials s)⟩ := by
  simp [maxDim,(check_dimension_iff a).mpr h]

theorem maxDim_values (algebra : Compute.Algebra A) (a : DimArguments)
    (trials : List (DimProfile a.N)) (selected : DimProfile a.N) (s : Heap A)
    (j : Nat) (hj : j<a.M*a.K) :
    let final := dimensionRun algebra selected.reduction.tree a.M a.N a.K selected.blockM
      (probes algebra a.M a.N a.K trials s)
    final ("values",j)=.value (Compute.dimension algebra selected.reduction.tree
      (read algebra.padding s "input") a.N a.K j).1 ∧
    final ("indices",j)=.index (Compute.dimension algebra selected.reduction.tree
      (read algebra.padding s "input") a.N a.K j).2 := by
  have result := dimension_result algebra selected.reduction.tree a.M a.N a.K selected.blockM
    (block_choice _ selected.choice).1 (probes algebra a.M a.N a.K trials s) j hj
  simpa only [probes_input] using result

theorem maxDim_frame (algebra : Compute.Algebra A) (a : DimArguments)
    (trials : List (DimProfile a.N)) (selected : DimProfile a.N) (s : Heap A) (q : String × Nat)
    (miss : q.1≠"values" ∧ q.1≠"indices" ∨ a.M*a.K≤q.2) :
    dimensionRun algebra selected.reduction.tree a.M a.N a.K selected.blockM
      (probes algebra a.M a.N a.K trials s) q=s q := by
  rw [dimension_frame _ _ _ _ _ _ (block_choice _ selected.choice).1 _ q miss]
  exact probes_frame _ _ _ _ _ _ q miss

#axiomsClean max_success
#axiomsClean maxDim_success
#axiomsClean maxDim_values
#axiomsClean maxDim_frame
end
end VeriTile.Bench.Repairs.MaxReduction.Host
