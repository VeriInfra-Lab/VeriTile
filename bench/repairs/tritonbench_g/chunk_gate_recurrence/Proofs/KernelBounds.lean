import bench.repairs.tritonbench_g.chunk_gate_recurrence.Proofs.KernelLaunch

namespace VeriTile.Bench.Repairs.ChunkGateRecurrence.KernelBounds
open VeriTile.Triton KernelProgram KernelTensor
noncomputable section

theorem storage_index (b : Buffers c s kind seed) (r : RegionName) (off : Nat) (hi : off<extent b r) :
    (b.tensor r).storageOffset+off<(b.tensor r).storage.elements := by
  by_cases hs : r="Seed"
  · subst r
    have capacity := b.seedBacking
    change off<KernelIndex.seedExtent c s at hi
    omega
  · have hi' : off<(b.tensor r).metadata.numel := by simpa [extent,hs] using hi
    have inside := (b.laws r).inBounds off hi'
    rw [(b.laws r).contiguous ((b.checked r).dense (by simp [dense,hs])) off hi'] at inside
    exact inside
theorem physical_word (b : Buffers c s kind seed) (r : RegionName) (off : Nat) (hi : off<extent b r) :
    (b.tensor r).storage.base≤KernelTensor.address b r off ∧
    KernelTensor.address b r off+4≤(b.tensor r).storage.base+4*(b.tensor r).storage.elements ∧
    KernelTensor.address b r off+4≤2^64 := by
  have inside := storage_index b r off hi
  have span := (b.tensor r).storage.spanFits
  simp only [KernelTensor.address,base,VeriTile.Torch.FP32.Storage.address]
  omega
theorem word_aligned (b : Buffers c s kind seed) (r : RegionName) (off : Nat)
    (hi : off<extent b r) (nonempty : 0<(b.tensor r).metadata.numel) :
    KernelTensor.address b r off%4=0 := by
  have span := (physical_word b r off hi).2.2
  have bounded : base b r<2^64 := by
    simp only [KernelTensor.address] at span
    omega
  have aligned := (b.checked r).alignment nonempty
  change (b.tensor r).dataPtr.toNat%4=0 at aligned
  rw [←pointer b r nonempty] at aligned
  simp only [KernelTensor.address,Nat.mul_zero,Nat.add_zero,BitVec.toNat_ofNat,
    Nat.mod_eq_of_lt bounded] at aligned
  simp [KernelTensor.address,Nat.add_mod,aligned]
theorem matrix (b : Buffers c s kind seed) (r : RegionName)
    (hd : r≠"D") (hs : r≠"Seed") (hl : r≠"DL") (hg : r≠"DG")
    (p : Program c) (t : Fin c.N) (e : Fin 64) (j : Fin 16) :
    KernelProgram.address c p t.val e.val j.val<extent b r := by
  simp only [extent,if_neg hs,KernelTensorInputs.matrix_count b r hd hs hl hg]
  exact KernelIndex.matrix_inside c p t e j
theorem gate (b : Buffers c s kind seed) (p : Program c) (t : Fin c.N) :
    gateAddress c p t.val<extent b "D" := by
  simp only [extent,show ("D" : RegionName)≠"Seed" by decide,if_false,KernelTensorInputs.gate_count]
  exact KernelIndex.gate_inside c p t
theorem seed_access (b : Buffers c s kind flag) (p : Program c) (e : Fin 64) (j : Fin 16) :
    KernelProgram.seedAddress c p s e.val j.val<extent b "Seed" := by
  simpa [extent] using KernelIndex.strided_seed_inside c p s e j
theorem dseed (b : Buffers c s kind seed) (p : Program c) (e : Fin 64) (j : Fin 16) :
    dseedAddress c p e.val j.val<extent b "DL" := by
  have count : (b.tensor "DL").metadata.numel=c.heads*c.K*c.V := by
    simp [VeriTile.Torch.TensorMetadata.numel,b.shape,tensorShape,Shape.heads,Nat.mul_assoc]
  simpa [extent,count] using KernelIndex.seed_inside c p e j
theorem gradient (b : Buffers c s kind seed) (p : Program c) (t : Fin c.N) :
    gradientAddress c p t.val<extent b "DG" := by
  have count : (b.tensor "DG").metadata.numel=c.heads*c.N*c.KT*c.VT := by
    simp [VeriTile.Torch.TensorMetadata.numel,b.shape,tensorShape,Nat.mul_assoc]
  simpa [extent,count] using KernelIndex.gradient_inside c p t

theorem writes (b : Buffers c s kind seed) (p : Program c) (r : RegionName) (off : Nat)
    (active : KernelMemory.footprint kind c p r off) : off<extent b r := by
  cases kind with
  | forward =>
    obtain ⟨rfl,t,e,j,rfl⟩ := active
    exact matrix b "O" (by decide) (by decide) (by decide) (by decide) p t e j
  | backward =>
    rcases active with ⟨rfl,t,e,j,rfl⟩ | ⟨rfl,t,rfl⟩ | ⟨rfl,e,j,rfl⟩
    · exact matrix b "DI" (by decide) (by decide) (by decide) (by decide) p ⟨t.val,by omega⟩ e j
    · exact gradient b p ⟨t.val,by omega⟩
    · exact dseed b p e j

#axiomsClean physical_word
#axiomsClean word_aligned
#axiomsClean matrix
#axiomsClean gate
#axiomsClean seed_access
#axiomsClean writes
end
end VeriTile.Bench.Repairs.ChunkGateRecurrence.KernelBounds
