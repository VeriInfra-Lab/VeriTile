import bench.repairs.tritonbench_g.bmm_chunk_bwd.Proofs.Layout
import bench.repairs.tritonbench_g.bmm_chunk_bwd.Proofs.NativeExecution

namespace VeriTile.Bench.Repairs.BmmChunkBwd.Direct
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.BmmChunkBwd
noncomputable section
open Classical

def placement (p : Metadata) : FPFlat.Placement where
  region := id
  base := fun _ => 0
  extent := extent p

def byteBase (p : Metadata) (a : Allocation p) (r : RegionName) : Nat :=
  if r="A" then base p .a else if r="Dout" then base p .dout
  else if r="Residual" then if p.hasResidual then base p .residual else 0
  else if r="Out" then a.pointer else 0

def nativeAllocation (p : Metadata) (h : InputContract p) (a : Allocation p) :
    NativeAddress.Allocation (placement p) where
  byteBase := byteBase p a
  addressEnd := by
    intro r
    simp only [placement,id_eq,Nat.zero_add]
    by_cases ha : r="A"
    · subst r; simpa [byteBase,extent,Nat.mul_comm] using a.inputFits .a (by simp [Used])
    by_cases hd : r="Dout"
    · subst r; simpa [byteBase,extent,Nat.mul_comm] using a.inputFits .dout (by simp [Used])
    by_cases hr : r="Residual"
    · subst r
      cases res : p.hasResidual
      · simp [byteBase,extent,res]
      · simpa [byteBase,extent,res,Nat.mul_comm] using a.inputFits .residual (by simp [Used,res])
    by_cases ho : r="Out"
    · subst r; simpa [byteBase,extent,Nat.mul_comm] using output_fits p h a
    simp [byteBase,extent,ha,hd,hr,ho]

def nativeMetadata (p : Metadata) (h : InputContract p) (nonempty : 0<(p.tensor .a).numel)
    (choice : Fin 9) : NativeControl.Metadata (call p) (tuning choice) where
  chunkPositive := h.chunk
  chunks := h.shapes.2
  scratchBytes := by
    have hh := h.extents .a (by simp [Used])
    rw [input_size p h] at hh
    change (batch p*seqlen p*(call p).groupCount*width p)*4<_
    rw [group_count]
    simpa [NativeControl.signedLimit,Nat.mul_comm] using hh
  doutNumel := by
    have hh := h.extents .dout (by simp [Used])
    rw [dout_size p h] at hh
    change batch p*chunks p*(call p).groupCount*chunk p*chunk p<_
    rw [group_count]
    unfold NativeControl.signedLimit
    omega
  gridX := by
    have hh := (h.grid (by omega)).2.2 choice
    change ((chunk p+(tuning choice).BM-1)/(tuning choice).BM)*((width p+(tuning choice).BN-1)/(tuning choice).BN)≤_
    omega
  gridY := by have hh := (h.grid (by omega)).1; change batch p≤_; omega
  gridZ := by
    have hh := (h.grid (by omega)).2.1
    change chunks p*(call p).groupCount≤_
    rw [group_count]
    omega

theorem precision (p : Metadata) : FPGrid.DefaultPrecision (call p) := ⟨rfl,rfl,rfl,rfl⟩
theorem residualTag (p : Metadata) : FPRoles.ResidualTag (call p) := by
  change ("Residual":RegionName)≠"A" ∧ ("Residual":RegionName)≠"Dout"
  decide
theorem freshStorage (p : Metadata) : FPPhysical.FreshStorage (call p) (placement p) := by
  intro r hr
  rcases hr with rfl|rfl|rfl <;> simp [placement,call]

#axiomsClean nativeAllocation
#axiomsClean nativeMetadata
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.BmmChunkBwd.Direct
