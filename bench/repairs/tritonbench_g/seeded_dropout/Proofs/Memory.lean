import bench.repairs.tritonbench_g.seeded_dropout.Proofs.Host

namespace VeriTile.Bench.Repairs.SeededDropout
open VeriTile.Triton VeriTile.Torch
open TritonBenchG.SeededDropout TritonBenchG.SeededDropout.Whole
noncomputable section

/-- Ordinary physical allocation mapping. Only active accesses need an address
inside the allocation. Inactive lanes retain exact modular uint64 arithmetic. -/
structure PhysicalAPI (x : TensorMetadata) (a : Allocated α input entry preexisting)
    (api : AllocationAPI a) where
  byteBase : Nat → Nat
  inputPointer : x.dataPtr.toNat = byteBase input.allocation + 4*input.base
  outputAligned : 4 ∣ byteBase a.out.allocation
  inputByteEnd : byteBase input.allocation+4*api.capacity input.allocation ≤ 2^64
  outputByteEnd : byteBase a.out.allocation+4*api.capacity a.out.allocation ≤ 2^64
  separated : ∀ allocation, preexisting allocation →
    byteBase a.out.allocation+4*api.capacity a.out.allocation ≤ byteBase allocation ∨
    byteBase allocation+4*api.capacity allocation ≤ byteBase a.out.allocation

theorem pointer_modular (base elementBase offset : Nat) :
    (bytePointer base elementBase offset).toNat = (base+4*(elementBase+offset)) % 2^64 := by
  unfold bytePointer
  rw [BitVec.ofNat_add_ofNat,BitVec.toNat_ofNat]
  congr 1
  omega

def ActiveAccess (x : TensorMetadata) (a : Allocated α input entry preexisting)
    (api : AllocationAPI a) (physical : PhysicalAPI x a api) : Prop :=
  (∀ pid lane, pid < gridCount input.numel → lane < 1024 →
    SourceDomain.I32Rep (pid*1024+lane : Nat) ∧
    (SourceDomain.offsetBits pid 1024 lane).toNat = pid*1024+lane ∧
    ((SourceDomain.offsetBits pid 1024 lane).slt (SourceDomain.i32 input.numel) = Bool.true ↔
      pid*1024+lane < input.numel)) ∧
  (∀ i, i < input.numel →
    (bytePointer (physical.byteBase input.allocation) input.base i).toNat =
      physical.byteBase input.allocation+4*(input.base+i) ∧
    (bytePointer (physical.byteBase a.out.allocation) a.out.base i).toNat =
      physical.byteBase a.out.allocation+4*(a.out.base+i) ∧
    physical.byteBase input.allocation+4*(input.base+i)+4 ≤ 2^64 ∧
    physical.byteBase a.out.allocation+4*(a.out.base+i)+4 ≤ 2^64 ∧
    4 ∣ physical.byteBase input.allocation+4*(input.base+i) ∧
    4 ∣ physical.byteBase a.out.allocation+4*(a.out.base+i)) ∧
  (∀ allocation byte i lane, preexisting allocation → byte < 4*api.capacity allocation →
    i < input.numel → lane < 4 →
    physical.byteBase a.out.allocation+4*(a.out.base+i)+lane ≠ physical.byteBase allocation+byte)

theorem active_access (x : TensorMetadata) (p : ℝ) (seed : Int) (h : InputContract x p seed)
    (binding : MetadataBinding x input) (a : Allocated α input entry preexisting)
    (api : AllocationAPI a) (physical : PhysicalAPI x a api) : ActiveAccess x a api physical := by
  have size := (admittedDomain x p seed h binding a api).size
  refine ⟨?_,?_,?_⟩
  · intro pid lane hp hl
    have hp' : pid < SourceDomain.grid input.numel 1024 := by
      simpa [gridCount,SourceDomain.grid] using hp
    exact ⟨(SourceDomain.source_intermediates_i32 SourceDomain.wrapper_block_legal size hp' hl).2.2.2.2.2,
      (SourceDomain.source_operations_exact SourceDomain.wrapper_block_legal size hp' hl).2.2.2.2.2,
      SourceDomain.source_mask_exact SourceDomain.wrapper_block_legal size hp' hl⟩
  · intro i hi
    have ib := api.inputBounds i hi
    have ob := api.outputBounds i hi
    have ie := physical.inputByteEnd
    have oe := physical.outputByteEnd
    have positive : 0 < x.numel := by
      simpa only [VeriTile.Torch.TensorMetadata.numel,Tensor.numel,binding.shape] using (Nat.zero_lt_of_lt hi)
    have aligned : 4 ∣ x.dataPtr.toNat := Nat.dvd_of_mod_eq_zero (h.tensor.alignment positive).1
    rw [physical.inputPointer] at aligned
    refine ⟨bytePointer_exact _ _ _ (by omega),bytePointer_exact _ _ _ (by omega),
      by omega,by omega,?_,dvd_add physical.outputAligned (dvd_mul_right 4 _)⟩
    have sum := dvd_add aligned (dvd_mul_right 4 i)
    convert sum using 1 <;> omega
  · intro allocation byte i lane existed hbyte hi hlane
    have bound := api.outputBounds i hi
    rcases physical.separated allocation existed with before | after <;> omega

#axiomsClean pointer_modular
#axiomsClean active_access
end
end VeriTile.Bench.Repairs.SeededDropout
