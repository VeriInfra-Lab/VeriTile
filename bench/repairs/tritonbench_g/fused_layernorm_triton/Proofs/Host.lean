import bench.repairs.tritonbench_g.fused_layernorm_triton.Proofs.NativeIndex

namespace VeriTile.Bench.Repairs.FusedLayernormTriton.Host
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.FusedLayernormTriton FPStages Whole
open Binding
noncomputable section

def checked {α : Type} [Inhabited α] (M : Algebra α) (eq32 : α → α → Bool)
    (w b x : Contract.Metadata) (W B X : Torch.Tensor) (s : State α) (a : Allocations)
    (cs : List Configuration) (last : Configuration) : Option (Returned α) :=
  match Contract.checkInputs w b x with
  | .error _ => none
  | .ok _ => Source.host M eq32 s (descriptor X [Contract.width x,1]) (descriptor W [1])
      (descriptor B [1]) a (Contract.rows x) (Contract.width x) cs last

theorem checked_eq {α : Type} [Inhabited α] (M : Algebra α) (eq32 : α → α → Bool)
    (w b x : Contract.Metadata) (W B X : Torch.Tensor) (s : State α) (a : Allocations)
    (cs : List Configuration) (last : Configuration) (h : Contract.InputContract w b x) :
    checked M eq32 w b x W B X s a cs last =
      executeHost M eq32 s (descriptor X [Contract.width x,1]) (descriptor W [1]) (descriptor B [1])
        a (Contract.rows x) (Contract.width x) cs last := by
  simp only [checked, (Contract.checkInputs_iff w b x).mpr h, Source.host_eq]

/-- Historical host spelling of the kernel input binding. -/
abbrev Inputs := @Binding.Inputs

theorem domain {α : Type} (s : State α) (read : Nat → α) (w b x : Contract.Metadata)
    (W B X : Torch.Tensor) (api : Inputs s read w b x W B X) (h : Contract.InputContract w b x) :
    Whole.InputContract s (descriptor X [Contract.width x,1]) (descriptor W [1]) (descriptor B [1])
      (Contract.rows x) (Contract.width x) :=
  whole_domain s read w b x W B X h api.weightMetadata api.biasMetadata api.inputMetadata
    api.weightAPI api.biasAPI api.inputAPI api.weightMemory api.biasMemory api.inputMemory

/-- Bounds for every admitted invocation and both original configurations,
including padded lanes. The loop condition is the actual source condition. -/
theorem native_bounds {α : Type} (s : State α) (read : Nat → α) (w b x : Contract.Metadata)
    (W B X : Torch.Tensor) (api : Inputs s read w b x W B X) (h : Contract.InputContract w b x)
    (c : Configuration) (row k lane : Nat) (hr : row < Contract.rows x)
    (hk : k * c.block < Contract.width x) (hl : lane < c.block) :
    NativeIndex.Bounds (Contract.rows x) (Contract.width x) c.block row k lane := by
  have sizes := counts w b x W X h api.weightMetadata api.inputMetadata api.weightAPI api.inputAPI
  exact NativeIndex.bounds _ _ _ _ _ _ h.positive sizes.1 sizes.2 c.allowed hr hk hl

def NativeSafe (S D : Nat) : Prop :=
  ∀ (c : Configuration) row k lane, row < S → k * c.block < D → lane < c.block →
    NativeIndex.Bounds S D c.block row k lane

theorem native_safe {α : Type} (s : State α) (read : Nat → α) (w b x : Contract.Metadata)
    (W B X : Torch.Tensor) (api : Inputs s read w b x W B X) (h : Contract.InputContract w b x) :
    NativeSafe (Contract.rows x) (Contract.width x) :=
  native_bounds s read w b x W B X api h

/-- Result metadata records the original X object, BF16 Y, FP32 statistics,
and the exact inverse backing alias. -/
theorem metadata (a : Allocations) (X : Torch.Tensor) (S D : Nat) :
    (outputTensor a S D).dtype = .bf16 ∧
    (descriptor X [D,1]).objectId = X.object ∧
    (meanTensor a S).dtype = .fp32 ∧ (inverseTensor a S).dtype = .fp32 ∧
    (inverseTensor a S).region = (inverseBacking a S).region := by
  exact ⟨rfl,rfl,rfl,rfl,rfl⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FusedLayernormTriton.Host
