import bench.repairs.tritonbench_g.fp4_to_bf16.Proofs.RawKernel
import bench.repairs.tritonbench_g.fp4_to_bf16.Proofs.CheckedHost
import bench.repairs.tritonbench_g.fp4_to_bf16.Kernels
import bench.repairs.tritonbench_g.fp4_to_bf16.Proofs.Host

/-! Exact complete-memory equivalence for any two complete program orders and
arbitrary unused backend operations. Profile: raw packed uint8, signed-i32 bit
payload, FP32 bitcast, BF16 conversion. All 256 bytes and both signed zeros are
included. No scalar numerical obligations are deferred. This does not equate
the repaired finite source with the defective original finite-i32 source. -/
namespace VeriTile.Bench.Repairs.Fp4ToBf16.HistoricalFPEquiv
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.Fp4ToBf16 FPValue Host HostSchedule
open Bytes Connection
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

specification complete_fp_equivalence
    (left right : Algebra Word) (h : Input) (s : Finite.Storage h.shape.count)
    (heap : Heap) (initial : State Word) (a b : List Nat)
    (fresh : h.input≠h.output) (rep : StorageRep h s heap initial)
    (gridSafe : GridAdmitted h)
    (ha : a.Perm (List.range (grid h))) (hb : b.Perm (List.range (grid h))) :
    ∃ sa sb, Launch.run left h a initial=some sa ∧
      Launch.run right h b initial=some sb ∧ sa.mem=sb.mem ∧
      physical h s heap sa.mem=physical h s heap sb.mem ∧
      (∀ j, j<2*h.shape.count → read (physical h s heap sa.mem) (s.outputBase+2*j)=
        Bits.expectedBits (targetNibble s heap j)) ∧
      (∀ address, (address<s.outputBase ∨ s.outputBase+4*h.shape.count≤address) →
        physical h s heap sa.mem address=heap address) := by
  obtain ⟨sa,ra,ma,fa⟩ :=
    RawKernel.complete left h s a initial
      fresh (packed_of_rep h s heap initial rep.input) gridSafe ha
  obtain ⟨sb,rb,mb,_⟩ :=
    RawKernel.complete right h s b initial
      fresh (packed_of_rep h s heap initial rep.input) gridSafe hb
  have same : sa.mem=sb.mem := ma.trans mb.symm
  refine ⟨sa,sb,ra,rb,same,congrArg (physical h s heap) same,?_,physical_frame h s heap sa.mem⟩
  intro j hj
  rw [physical_read h s heap sa.mem j hj,ma]
  have bound : j<h.shape.count*2 := by omega
  simp only [expectedMemory, bound, and_self, if_true, cell, value]
  rw [read_embed,nibble_from_bytes h s heap initial rep.input j hj]

#axiomsClean complete_fp_equivalence

specification checked_complete_fp_equivalence
    (left right : Algebra Word) (x : VeriTile.Torch.TensorMetadata)
    (checked : Checked.checkInputs (.tensor x) = .ok ())
    (h : Input) (s : Finite.Storage h.shape.count) (identity device : Nat)
    (binding : Checked.MetadataBinding x h s device)
    (heap : Heap) (logical : Nat → BitVec 8) (storage : Checked.LogicalStorage x heap logical)
    (initial : State Word) (a b : List Nat)
    (fresh : h.input≠h.output) (rep : StorageRep h s heap initial)
    (gridSafe : GridAdmitted h)
    (ha : a.Perm (List.range (grid h))) (hb : b.Perm (List.range (grid h))) :
    Checked.InputContract x ∧ x.numel = h.shape.count ∧
    x.dataPtr.toNat = s.inputPointer ∧
    ∃ sa sb,
      Checked.runChecked left (.tensor x) h s heap initial identity device a Bool.true Bool.true Bool.true =
        some (.ok ⟨returned h s identity device,sa,physical h s heap sa.mem⟩) ∧
      Checked.runChecked right (.tensor x) h s heap initial identity device b Bool.true Bool.true Bool.true =
        some (.ok ⟨returned h s identity device,sb,physical h s heap sb.mem⟩) ∧
      (returned h s identity device).identity = identity ∧
      (returned h s identity device).shape = h.shape.output ∧
      (returned h s identity device).dtype = .bf16 ∧
      (returned h s identity device).device = x.device.index ∧
      sa.mem = sb.mem ∧ physical h s heap sa.mem = physical h s heap sb.mem ∧
      (∀ j, j<2*x.numel → read (physical h s heap sa.mem) (s.outputBase+2*j) =
        Bits.expectedBits (Checked.logicalNibble logical j)) ∧
      (∀ address, (address<s.outputBase ∨ s.outputBase+4*x.numel≤address) →
        physical h s heap sa.mem address=heap address) := by
  have count := Checked.numel_binding x h s device binding
  obtain ⟨sa,sb,ra,rb,mem,bytes,values,frame⟩ :=
    complete_fp_equivalence left right h s heap initial a b fresh rep gridSafe ha hb
  refine ⟨(Checked.checkInputs_iff (.tensor x)).mp checked,count,binding.pointer,
    sa,sb,?_,?_,rfl,rfl,rfl,binding.device.symm,mem,bytes,?_,?_⟩
  · rw [Checked.checked_source left x checked]
    simp [observe,protocol_success h.shape gridSafe,ra,Except.mapError]
  · rw [Checked.checked_source right x checked]
    simp [observe,protocol_success h.shape gridSafe,rb,Except.mapError]
  · intro j hj
    have raw := values j (by simpa only [←count] using hj)
    rw [Checked.logical_nibble x h s device binding heap logical storage checked j hj] at raw
    exact raw
  · simpa only [count] using frame

#axiomsClean checked_complete_fp_equivalence
/-- This exact structural theorem has no deferred FP assumption list. -/
def deferredNumericalObligations : List String := []
end
end VeriTile.Bench.Repairs.Fp4ToBf16.HistoricalFPEquiv
