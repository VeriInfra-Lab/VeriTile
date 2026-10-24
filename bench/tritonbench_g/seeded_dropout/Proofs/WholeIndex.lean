import bench.tritonbench_g.seeded_dropout.Proofs.WholeHost

namespace VeriTile.Bench.TritonBenchG.SeededDropout.Whole
open VeriTile.Triton TypedSource
noncomputable section
set_option maxHeartbeats 4000000

/-- Allocation byte ends and padded pointer arithmetic are independent of
active masks. The capacity is measured in FP32 cells, byteBase in bytes. -/
structure FiniteAdmission (a : Allocated α input entry preexisting) (d : Domain a) where
  byteBase : Nat → Nat
  inputAligned : 4 ∣ byteBase input.allocation
  outputAligned : 4 ∣ byteBase a.out.allocation
  inputByteEnd : byteBase input.allocation + 4*d.capacity input.allocation ≤ 2^64
  outputByteEnd : byteBase a.out.allocation + 4*d.capacity a.out.allocation ≤ 2^64
  inputPaddedEnd : byteBase input.allocation +
    4*(input.base+SourceDomain.grid input.numel 1024*1024) ≤ 2^64
  preexistingSeparated : ∀ allocation, preexisting allocation →
    byteBase a.out.allocation+4*d.capacity a.out.allocation ≤ byteBase allocation ∨
    byteBase allocation+4*d.capacity allocation ≤ byteBase a.out.allocation
  outputPaddedEnd : byteBase a.out.allocation +
    4*(a.out.base+SourceDomain.grid input.numel 1024*1024) ≤ 2^64

def bytePointer (base elementBase offset : Nat) : BitVec 64 :=
  BitVec.ofNat 64 (base+4*elementBase) + BitVec.ofNat 64 (4*offset)

theorem bytePointer_exact (base elementBase offset : Nat)
    (bound : base+4*(elementBase+offset) < 2^64) :
    (bytePointer base elementBase offset).toNat = base+4*(elementBase+offset) := by
  unfold bytePointer
  rw [BitVec.ofNat_add_ofNat,BitVec.toNat_ofNat]
  have equal : base+4*elementBase+4*offset = base+4*(elementBase+offset) := by omega
  rw [equal,Nat.mod_eq_of_lt bound]

/-- Full source-visible finite correspondence, before any numerical law. -/
def IndexCorrespondence (a : Allocated α input entry preexisting) (d : Domain a)
    (f : FiniteAdmission a d) : Prop :=
  (∀ (pid lane : Nat), pid < SourceDomain.grid input.numel 1024 → lane < 1024 →
    SourceDomain.I32Rep (input.numel : Int) ∧
    SourceDomain.I32Rep (1024 : Int) ∧ SourceDomain.I32Rep (pid : Int) ∧
    SourceDomain.I32Rep (lane : Int) ∧ SourceDomain.I32Rep (pid*1024 : Nat) ∧
    SourceDomain.I32Rep (pid*1024+lane : Nat) ∧
    (SourceDomain.offsetBits pid 1024 lane).toNat = pid*1024+lane ∧
    ((SourceDomain.offsetBits pid 1024 lane).slt (SourceDomain.i32 input.numel) = Bool.true ↔
      pid*1024+lane < input.numel) ∧
    (bytePointer (f.byteBase input.allocation) input.base (pid*1024+lane)).toNat =
      f.byteBase input.allocation+4*(input.base+(pid*1024+lane)) ∧
    (bytePointer (f.byteBase a.out.allocation) a.out.base (pid*1024+lane)).toNat =
      f.byteBase a.out.allocation+4*(a.out.base+(pid*1024+lane))) ∧
  (∀ i, i < input.numel →
    input.base+i < d.capacity input.allocation ∧ a.out.base+i < d.capacity a.out.allocation ∧
    f.byteBase input.allocation+4*(input.base+i)+4 ≤ 2^64 ∧
    f.byteBase a.out.allocation+4*(a.out.base+i)+4 ≤ 2^64 ∧
    4 ∣ f.byteBase input.allocation+4*(input.base+i) ∧
    4 ∣ f.byteBase a.out.allocation+4*(a.out.base+i)) ∧
  f.byteBase input.allocation+4*d.capacity input.allocation ≤ 2^64 ∧
  f.byteBase a.out.allocation+4*d.capacity a.out.allocation ≤ 2^64 ∧
  (∀ allocation byte i lane, preexisting allocation → byte < 4*d.capacity allocation →
    i < input.numel → lane < 4 →
    f.byteBase a.out.allocation+4*(a.out.base+i)+lane ≠ f.byteBase allocation+byte)

theorem index_correspondence (a : Allocated α input entry preexisting) (d : Domain a)
    (f : FiniteAdmission a d) : IndexCorrespondence a d f := by
  refine ⟨?_,?_,f.inputByteEnd,f.outputByteEnd,?_⟩
  · intro pid lane hp hl
    have rep := SourceDomain.source_intermediates_i32 SourceDomain.wrapper_block_legal d.size hp hl
    have ops := SourceDomain.source_operations_exact SourceDomain.wrapper_block_legal d.size hp hl
    have mask := SourceDomain.source_mask_exact SourceDomain.wrapper_block_legal d.size hp hl
    have padded : pid*1024+lane < SourceDomain.grid input.numel 1024*1024 := by omega
    refine ⟨rep.1,rep.2.1,rep.2.2.1,rep.2.2.2.1,rep.2.2.2.2.1,rep.2.2.2.2.2,
      ops.2.2.2.2.2,mask,?_,?_⟩
    · apply bytePointer_exact
      have h := f.inputPaddedEnd
      omega
    · apply bytePointer_exact
      have h := f.outputPaddedEnd
      omega
  · intro i hi
    have ib := d.inputBounds i hi
    have ob := d.outputBounds i hi
    refine ⟨ib,ob,?_,?_,?_,?_⟩
    · have h := f.inputByteEnd; omega
    · have h := f.outputByteEnd; omega
    · exact dvd_add f.inputAligned (dvd_mul_right 4 _)
    · exact dvd_add f.outputAligned (dvd_mul_right 4 _)

  · intro allocation byte i lane existed hbyte hi hlane
    have bound := d.outputBounds i hi
    rcases f.preexistingSeparated allocation existed with before | after
    · omega
    · omega

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.SeededDropout.Whole
