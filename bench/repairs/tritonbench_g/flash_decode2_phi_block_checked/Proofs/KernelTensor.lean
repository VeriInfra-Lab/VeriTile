import bench.repairs.tritonbench_g.flash_decode2_phi_block_checked.Proofs.KernelModel
import bench.repairs.tritonbench_g.flash_decode2_phi.Proofs.NativeSafety

namespace VeriTile.Bench.Repairs.FlashDecode2PhiBlockChecked.KernelTensor
open VeriTile.Triton FP.Structural FlashDecode2
open TensorBinding
noncomputable section
abbrev Heap := Nat → Physical.Byte

/-- Actual argument handles and ordinary metadata observations. No execution,
allocation, return-value or output equation is assumed by this binding. -/
structure Arguments (c : Call) where
  mid : FP32Argument
  log : FP32Argument
  output : FP32Argument
  lengths : LengthArgument
  midObserved : mid.observation = c.val.mid
  logObserved : log.observation = c.val.log
  outputObserved : output.observation = c.val.output
  lengthsObserved : lengths.observation = c.val.lengths

def descriptor (c : Call) (r : RegionName) : Checked.Observation :=
  if r="L" then c.val.lengths else if r="M" then c.val.mid else if r="E" then c.val.log else c.val.output

def load (heap : Heap) (address : Nat) : Native.Word :=
  Physical.decode (fun byte => heap (address+byte.val))

def word (c : Call) (heap : Heap) (r : RegionName) (i : Nat) : Native.Word :=
  load heap ((descriptor c r).metadata.dataPtr.toNat+4*i)

def lengths (c : Call) (heap : Heap) (b : Nat) : Int := (word c heap "L" b).toInt

/-- Initial typed memory observes the actual input pointers. Unmentioned tags
retain the caller's arbitrary memory; stores are performed only by the source. -/
def lift {α : Type} (observe : RegionName → Nat → Native.Word → α)
    (c : Call) (heap : Heap) (other : State α) : State α :=
  {other with mem := fun r i =>
    if r="L" then .mk .nat (lengths c heap i).toNat
    else if r="M" ∨ r="E" ∨ r="O" then .mk .real (observe r i (word c heap r i))
    else other.mem r i}

theorem lift_length {α : Type} [Inhabited α] (observe : RegionName → Nat → Native.Word → α)
    (c : Call) (heap : Heap) (other : State α) (b : Nat) :
    ((lift observe c heap other).mem "L" b).read .nat = (lengths c heap b).toNat := by
  simp [lift,Cell.read_same]

def readWord (heap : Heap) (address : Nat) : VeriTile.Torch.FP32.Word := ⟨load heap address⟩

theorem direct_logical (x : FP32Argument) (heap : Heap)
    (contiguous : x.observation.metadata.isContiguous = Bool.true)
    (zero : x.observation.storageOffset = 0) (noNeg : x.observation.metadata.isNeg = Bool.false)
    (i : Nat) (inside : i < x.observation.metadata.numel) :
    (⟨load heap (x.tensor.metadata.dataPtr.toNat+4*i)⟩ : VeriTile.Torch.FP32.Word) =
      VeriTile.Torch.FP32.logicalValue (readWord heap) x.tensor x.laws i := by
  have facts := NativeSafety.fp_facts x contiguous zero
  have pointer := facts.pointer (by omega)
  have cap := facts.capacity
  have span := x.tensor.storage.spanFits
  change x.tensor.metadata.numel ≤ x.tensor.storage.elements at cap
  have inside' : i<x.tensor.metadata.numel := inside
  have bound : x.tensor.storage.base < 2^64 := by omega
  have index := NativeSafety.fp_coordinates x contiguous zero i inside
  have ptr : x.tensor.metadata.dataPtr.toNat=x.tensor.storage.base := by
    change x.tensor.metadata.dataPtr = BitVec.ofNat 64 x.tensor.storage.base at pointer
    simpa only [BitVec.toNat_ofNat,Nat.mod_eq_of_lt bound] using congrArg BitVec.toNat pointer
  change x.tensor.metadata.isNeg = Bool.false at noNeg
  simp only [VeriTile.Torch.FP32.logicalValue,noNeg,Bool.false_eq_true,↓reduceIte,
    readWord,index,VeriTile.Torch.FP32.Storage.address,ptr]

theorem mid_read (c : Call) (t : Arguments c) (heap : Heap)
    (h : InputContract c (lengths c heap)) (i : Nat) (inside : i<c.val.mid.metadata.numel) :
    (⟨word c heap "M" i⟩ : VeriTile.Torch.FP32.Word) =
      VeriTile.Torch.FP32.logicalValue (readWord heap) t.mid.tensor t.mid.laws i := by
  have checks := h.metadata.1 c.val.mid (by simp [Checked.tensors])
  rw [←t.midObserved] at checks inside
  simpa [word,descriptor,←t.midObserved,FP32Argument.observation] using
    direct_logical t.mid heap checks.2.2.2.1 checks.2.2.2.2.1 checks.2.2.2.2.2.1 i inside

theorem log_read (c : Call) (t : Arguments c) (heap : Heap)
    (h : InputContract c (lengths c heap)) (i : Nat) (inside : i<c.val.log.metadata.numel) :
    (⟨word c heap "E" i⟩ : VeriTile.Torch.FP32.Word) =
      VeriTile.Torch.FP32.logicalValue (readWord heap) t.log.tensor t.log.laws i := by
  have checks := h.metadata.1 c.val.log (by simp [Checked.tensors])
  rw [←t.logObserved] at checks inside
  simpa [word,descriptor,←t.logObserved,FP32Argument.observation] using
    direct_logical t.log heap checks.2.2.2.1 checks.2.2.2.2.1 checks.2.2.2.2.2.1 i inside

theorem output_index (c : Call) (t : Arguments c) (heap : Heap)
    (h : InputContract c (lengths c heap)) (i : Nat) (inside : i<KernelModel.size c) :
    t.output.tensor.physicalIndex i=i := by
  have checks := h.metadata.1 c.val.output (by simp [Checked.tensors])
  have shape := h.metadata.2.2.2.2.2.2.1
  have bound : i<c.val.output.metadata.numel := by
    simpa [VeriTile.Torch.TensorMetadata.numel,shape,KernelModel.size,KernelModel.rows,Nat.mul_assoc] using inside
  rw [←t.outputObserved] at checks bound
  exact NativeSafety.fp_coordinates t.output checks.2.2.2.1 checks.2.2.2.2.1 i bound

theorem signed_length (c : Call) (heap : Heap) (b : Nat) :
    ((word c heap "L" b).signExtend 64).toInt=lengths c heap b :=
  BitVec.toInt_signExtend_of_le (by decide)

#axiomsClean mid_read
#axiomsClean log_read
#axiomsClean output_index
#axiomsClean signed_length
#specNonCircular Arguments avoiding [KernelModel.run,Grid.source,FP.Structural.exec]
end
end VeriTile.Bench.Repairs.FlashDecode2PhiBlockChecked.KernelTensor
