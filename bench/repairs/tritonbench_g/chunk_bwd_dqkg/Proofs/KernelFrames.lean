import bench.repairs.tritonbench_g.chunk_bwd_dqkg.Proofs.KernelNative
import bench.repairs.tritonbench_g.chunk_bwd_dqkg.Proofs.KernelReal

namespace VeriTile.Bench.Repairs.ChunkBwdDqkg.KernelFrames
open VeriTile.Triton FP.Structural Whole KernelTensor KernelContract
open VeriTile.Bench.Repairs.LayernormFwdTriton
noncomputable section
def inputRoles : List Role := [.q,.k,.v,.h,.g,.dout,.dh]
theorem input_output_different (r s : Role) (hr : r∈inputRoles) (hs : s∈outputRoles) : s≠r := by
  cases r <;> cases s <;> simp_all [inputRoles,outputRoles]

theorem input_bytes {α β Store Byte : Type} [Inhabited α] (api : Memory.API β Store Byte)
    (b : Buffers c) (observe : RegionName → Nat → Cell α → β) (final : State α) (store : Store)
    (r : Role) (hr : r∈inputRoles) (i : Nat) (hi : i<(b.tensor (region r)).metadata.numel) (byte : Fin 4) :
    api.byte (Shared.commit api (placement b) observe final store) ((b.view (region r)).address i+byte.val)=
      api.byte store ((b.view (region r)).address i+byte.val) := by
  apply Shared.commit_frame
  intro out j active
  obtain ⟨s,hs,eq,_⟩ := output_bounds c out j active
  have apart := roles_apart b s r (writes_nonempty c out j active)
    (input_output_different r s hr hs) j i
    (by rw [eq]; exact output_bound b out j active) hi
  have bt := byte.isLt
  change _ < (b.view out).address j ∨ (b.view out).address j+4≤_
  rw [eq] at apart
  unfold Memory.Apart at apart
  omega

theorem no_programs (c : Config) (empty : ¬nonempty c) : allCTAs c=[] := by
  have h : c.B=0 ∨ c.H=0 ∨ c.T=0 := by unfold nonempty at empty; omega
  rcases h with h|h|h <;> simp [allCTAs,Config.heads,Config.NT,h]
theorem real_empty (c : Config) (empty : ¬nonempty c) (s : BlockState) :
    Whole.runGrid c (allCTAs c) s=some s := by rw [no_programs c empty]; rfl
theorem native_empty (quote : ComputeKernel) (isa : KernelNativeCarrier.Instructions)
    (route : KernelNativeCarrier.Routing) (scale : KernelNativeCarrier.Word) (c : Config)
    (empty : ¬nonempty c) (s : State KernelNativeCarrier.Node) :
    KernelNative.runQuoted quote isa route scale c (allCTAs c) s=some s := by
  rw [no_programs c empty]
  rfl

#axiomsClean input_bytes
#axiomsClean real_empty
#axiomsClean native_empty
end
end VeriTile.Bench.Repairs.ChunkBwdDqkg.KernelFrames
