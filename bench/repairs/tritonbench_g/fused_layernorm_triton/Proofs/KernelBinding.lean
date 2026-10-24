import bench.repairs.tritonbench_g.fused_layernorm_triton.Proofs.Kernel
import bench.repairs.tritonbench_g.fused_layernorm_triton.Proofs.Native

namespace VeriTile.Bench.Repairs.FusedLayernormTriton.KernelBinding
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.FusedLayernormTriton FPStages Whole
open Binding
noncomputable section

abbrev Word := VeriTile.Torch.FP32.Word

def logical (read : Nat → Word) (t : Torch.Tensor) (laws : Torch.TensorLaws t) (i : Nat) : Word :=
  VeriTile.Torch.FP32.logicalValue read t laws i

def graphRead (read : Nat → Word) (a : Nat) : Native.Term :=
  .leaf "physical" a (read a).bits

def logicalGraph (read : Nat → Word) (t : Torch.Tensor) (laws : Torch.TensorLaws t) (i : Nat) : Native.Term :=
  .leaf "physical" (t.storage.address (t.physicalIndex i)) (logical read t laws i).bits

theorem decoded_logical (read : Nat → Word) (decode : Word → ℝ) (m : Contract.Metadata)
    (t : Torch.Tensor) (laws : Torch.TensorLaws t) (bound : MetadataBinding m t)
    (h : Contract.TensorContract m) (i : Nat) :
    logicalRead (fun a => decode (read a)) t i = decode (logical read t laws i) := by
  simp only [logicalRead, logical, VeriTile.Torch.FP32.logicalValue, bound.metadata, h.noNeg,
    Bool.false_eq_true, ↓reduceIte]

theorem graph_logical (read : Nat → Word) (m : Contract.Metadata)
    (t : Torch.Tensor) (laws : Torch.TensorLaws t) (bound : MetadataBinding m t)
    (h : Contract.TensorContract m) (i : Nat) :
    logicalRead (graphRead read) t i = logicalGraph read t laws i := by
  simp only [logicalRead, graphRead, logicalGraph, logical, VeriTile.Torch.FP32.logicalValue,
    bound.metadata, h.noNeg, Bool.false_eq_true, ↓reduceIte]

/-- Every entry read is the original logical Tensor word. One shared physical
reader supplies all inputs, including identical or overlapping read-only storage. -/
theorem graph_cells (read : Nat → Word) (s : State Native.Term)
    (w b x : Contract.Metadata) (W B X : Torch.Tensor) (h : Contract.InputContract w b x)
    (api : Inputs s (graphRead read) w b x W B X) :
    (∀ row, row < Contract.rows x → ∀ col, col < Contract.width x →
      (s.mem (region X) (col+Contract.width x*row)).read .real =
        logicalGraph read X api.inputAPI (col+Contract.width x*row)) ∧
    (∀ col, col < Contract.width x → (s.mem (region W) col).read .real = logicalGraph read W api.weightAPI col) ∧
    (∀ col, col < Contract.width x → (s.mem (region B) col).read .real = logicalGraph read B api.biasAPI col) := by
  refine ⟨?_,?_,?_⟩
  · intro row hr col hc
    have hi : col+Contract.width x*row < x.numel := by
      simpa [VeriTile.Torch.TensorMetadata.numel, Contract.input_shape x h.rank] using output_bounds _ _ _ _ hr hc
    rw [coordinate s (graphRead read) x X api.inputAPI api.inputMetadata api.inputMemory h.input _ hi]
    exact graph_logical read x X api.inputAPI api.inputMetadata h.input _
  · intro col hc
    have hi : col < w.numel := by simpa [VeriTile.Torch.TensorMetadata.numel, h.weightShape] using hc
    rw [coordinate s (graphRead read) w W api.weightAPI api.weightMetadata api.weightMemory h.weight col hi]
    exact graph_logical read w W api.weightAPI api.weightMetadata h.weight col
  · intro col hc
    have hi : col < b.numel := by simpa [VeriTile.Torch.TensorMetadata.numel, h.biasShape] using hc
    rw [coordinate s (graphRead read) b B api.biasAPI api.biasMetadata api.biasMemory h.bias col hi]
    exact graph_logical read b B api.biasAPI api.biasMetadata h.bias col

/-- Actual pointers and int64 loop arithmetic are derived from the same checked
metadata and physical storage used in both public value theorems. -/
theorem pointers_and_indices {α : Type} (s : State α) (read : Nat → α)
    (w b x : Contract.Metadata) (W B X : Torch.Tensor) (h : Contract.InputContract w b x)
    (api : Inputs s read w b x W B X) :
    (∀ m t, (m = w ∧ t = W) ∨ (m = b ∧ t = B) ∨ (m = x ∧ t = X) →
      0 < m.numel → m.dataPtr = BitVec.ofNat 64 t.storage.base) ∧
    (∀ (c : Contract.Configuration) row k lane, row < Contract.rows x →
      k*c.block < Contract.width x → lane < c.block →
      NativeIndex.Bounds (Contract.rows x) (Contract.width x) c.block row k lane) := by
  constructor
  · intro m t ht hn
    rcases ht with ⟨rfl,rfl⟩ | ⟨rfl,rfl⟩ | ⟨rfl,rfl⟩
    · exact actual_pointer _ _ api.weightAPI api.weightMetadata h.weight hn
    · exact actual_pointer _ _ api.biasAPI api.biasMetadata h.bias hn
    · exact actual_pointer _ _ api.inputAPI api.inputMetadata h.input hn
  · intro c row k lane hr hk hl
    have sizes := counts w b x W X h api.weightMetadata api.inputMetadata api.weightAPI api.inputAPI
    exact NativeIndex.bounds _ _ _ _ _ _ h.positive sizes.1 sizes.2 c.allowed hr hk hl

#axiomsClean graph_cells
#axiomsClean pointers_and_indices
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FusedLayernormTriton.KernelBinding
