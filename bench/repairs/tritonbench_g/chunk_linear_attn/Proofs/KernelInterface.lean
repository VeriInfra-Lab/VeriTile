import bench.repairs.tritonbench_g.chunk_linear_attn.Proofs.KernelReal
import bench.repairs.tritonbench_g.chunk_linear_attn.Proofs.KernelSchedule

namespace VeriTile.Bench.Repairs.ChunkLinearAttn.KernelInterface
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive)
noncomputable section
set_option maxHeartbeats 8000000

/-- Raw dimensions, scalar, optional-state flags and finite reduction trees.
No Tensor objects, runtime admission flags, device or profile labels occur. -/
structure Config where
  dims : Host.Dimensions
  scale : ℝ
  (seed final terminal : Bool)
  plans : Fin (valueCount (dims.parameters Bool.true)) → LibrarySum.Plan (keyBlocks (dims.parameters Bool.true))

def Config.call (c : Config) : Real.Call :=
  ⟨c.dims,c.scale,c.seed,c.final,c.terminal,⟨c.plans,"",0⟩⟩

/-- Discard provenance labels while retaining every numerical operation tree. -/
def Config.ofCall (c : Real.Call) : Config :=
  ⟨c.dims,c.scale,c.seed,c.final,c.terminal,c.selection.plans⟩

theorem run_correspondence {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : Real.Call) (before : Memory α) :
    KernelExecution.run M D (Config.ofCall c).call before=KernelExecution.run M D c before := rfl

/-- Exact complement of the four kernel grids and the finite final reduction. -/
def Outside (c : Real.Call) (r : RegionName) (o : Nat) : Prop :=
  (∀ pid∈grid .forwardState c.forward,¬Footprint .forwardState c.forward c.final pid r o) ∧
  (∀ pid∈grid .forwardOutput c.forward,¬Footprint .forwardOutput c.forward Bool.false pid r o) ∧
  (∀ pid∈grid .backwardState c.backward,¬Footprint .backwardState c.backward c.seed pid r o) ∧
  (∀ pid∈grid .backwardGradient c.backward,¬Footprint .backwardGradient c.backward Bool.false pid r o) ∧
  (r≠"valueGradient" ∨ valueCount c.backward≤o)

theorem complete_frame {α : Type} [Inhabited α] (M : Algebra α) (D : DotPrimitive α)
    (c : Real.Call) (before : Memory α) (r : RegionName) (o : Nat) (outside : Outside c r o) :
    completeMemory M D c.forward c.backward c.scale c.seed c.final c.terminal c.selection before r o=before r o := by
  rcases outside with ⟨hf,ho,hb,hg,hs⟩
  unfold completeMemory modelFour
  rw [host_sum_frame M c.backward c.selection _ _ r o hs,
    grid_frame M D .backwardGradient c.backward c.scale Bool.false Bool.false _ _ r o hg,
    grid_frame M D .backwardState c.backward c.scale c.terminal c.seed _ _ r o hb,
    grid_frame M D .forwardOutput c.forward c.scale Bool.false Bool.false _ _ r o ho,
    grid_frame M D .forwardState c.forward c.scale c.seed c.final _ _ r o hf]

theorem native_frame (profile : NativeTerms.FP32Profile) (c : Real.Call) (valid : Host.Valid c.dims)
    (scale : BitVec 32) (before : RegionName → Nat → BitVec 32) (r : RegionName) (o : Nat)
    (outside : Outside c r o) : KernelExecution.nativeTarget profile c valid scale before r o=before r o := by
  unfold KernelExecution.nativeTarget KernelExecution.project
  rw [←KernelReference.result_eq NativeTerms.algebra NativeTerms.dotPrimitive c.dims valid 1
    c.seed c.final c.terminal c.selection (NativeTerms.inputMemory before)]
  have frame:=complete_frame NativeTerms.algebra NativeTerms.dotPrimitive (KernelExecution.symbolicCall c)
    (NativeTerms.inputMemory before) r o outside
  dsimp only [KernelExecution.symbolicCall,Real.Call.forward,Real.Call.backward] at frame
  rw [frame]
  simp only [NativeTerms.inputMemory,Cell.read_same,NativeBound.bindScale,NativeTerms.lowerAt]
  split <;> rfl

#axiomsClean run_correspondence
#axiomsClean complete_frame
#axiomsClean native_frame
end
end VeriTile.Bench.Repairs.ChunkLinearAttn.KernelInterface
