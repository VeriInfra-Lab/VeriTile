import bench.repairs.tritonbench_g.chunk_gate_recurrence.Proofs.KernelTensor

namespace VeriTile.Bench.Repairs.ChunkGateRecurrence.KernelTensorInputs
open VeriTile.Triton FP.Structural KernelProgram KernelTensor
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 8000000

theorem matrix_count (b : Buffers c s kind seed) (r : RegionName)
    (hd : r≠"D") (hs : r≠"Seed") (hl : r≠"DL") (hg : r≠"DG") :
    (b.tensor r).metadata.numel=c.heads*c.N*c.K*c.V := by
  simp [VeriTile.Torch.TensorMetadata.numel,b.shape,tensorShape,hd,hs,hl,hg,Shape.heads,Nat.mul_assoc]
theorem gate_count (b : Buffers c s kind seed) : (b.tensor "D").metadata.numel=c.heads*c.N := by
  simp [VeriTile.Torch.TensorMetadata.numel,b.shape,tensorShape,Shape.heads,Nat.mul_assoc]

theorem native_dense (b : Buffers c s kind seed) (read : Nat → KernelCarrier.Word)
    (u : State KernelCarrier.Node) (r : RegionName) (notSeed : r≠"Seed")
    (i : Nat) (hi : i<(b.tensor r).metadata.numel) :
    (rawCell kind r ((physical b (fun a => KernelCarrier.atom (read a)) u).mem r i)).value=
      (VeriTile.Torch.FP32.logicalValue (fun a => ⟨read a⟩) (b.tensor r) (b.laws r) i).bits := by
  rw [physical_read b _ u r i (by simpa [extent,notSeed] using hi)]
  exact congrArg (fun x : VeriTile.Torch.FP32.Word => x.bits) (dense_read b (fun a => ⟨read a⟩) r notSeed i hi)
theorem native_seed (b : Buffers c s .forward seed) (read : Nat → KernelCarrier.Word)
    (u : State KernelCarrier.Node) (p : Program c) (e : Fin 64) (j : Fin 16) :
    (rawCell .forward "Seed" ((physical b (fun a => KernelCarrier.atom (read a)) u).mem "Seed"
      (KernelProgram.seedAddress c p s e.val j.val))).value=
      (VeriTile.Torch.FP32.logicalValue (fun a => ⟨read a⟩) (b.tensor "Seed") (b.laws "Seed")
        (dseedAddress c p e.val j.val)).bits := by
  rw [physical_read b _ u "Seed" _ (by simpa [extent] using KernelIndex.strided_seed_inside c p s e j)]
  exact congrArg (fun x : VeriTile.Torch.FP32.Word => x.bits) (seed_read b (fun a => ⟨read a⟩) p e j)

def logical (b : Buffers c s kind seed) (read : Nat → KernelCarrier.Word) (p : Program c) : KernelReference.Inputs KernelCarrier.Word where
  source := fun t i => (VeriTile.Torch.FP32.logicalValue (fun a => ⟨read a⟩) (b.tensor "S") (b.laws "S")
    (KernelProgram.address c p t i.1.val i.2.1.val)).bits
  gate := fun t => (VeriTile.Torch.FP32.logicalValue (fun a => ⟨read a⟩) (b.tensor "D") (b.laws "D")
    (gateAddress c p t)).bits
  seed := fun i => (VeriTile.Torch.FP32.logicalValue (fun a => ⟨read a⟩) (b.tensor "Seed") (b.laws "Seed")
    (dseedAddress c p i.1.val i.2.1.val)).bits
  incoming := fun t i => (VeriTile.Torch.FP32.logicalValue (fun a => ⟨read a⟩) (b.tensor "DS") (b.laws "DS")
    (KernelProgram.address c p t i.1.val i.2.1.val)).bits
  saved := fun t i => (VeriTile.Torch.FP32.logicalValue (fun a => ⟨read a⟩) (b.tensor "S") (b.laws "S")
    (KernelProgram.address c p t i.1.val i.2.1.val)).bits
def regional (b : Buffers c s kind seed) (read : Nat → KernelCarrier.Word) (p : Program c) (u : State KernelCarrier.Node) :=
  KernelWords.inputs (KernelReference.read c s p (physical b (fun a => KernelCarrier.atom (read a)) u))

theorem gate (b : Buffers c s kind seed) (read : Nat → KernelCarrier.Word)
    (p : Program c) (u : State KernelCarrier.Node) (t : Fin c.N) :
    (regional b read p u).gate t.val=(logical b read p).gate t.val := by
  have h := native_dense b read u "D" (by decide) (gateAddress c p t.val)
    (by rw [gate_count]; exact KernelIndex.gate_inside c p t)
  cases kind <;> simpa [regional,KernelWords.inputs,KernelReference.read,logical,rawCell,KernelExecution.dtype,toFloat] using h
theorem forward_source (b : Buffers c s .forward seed) (read : Nat → KernelCarrier.Word)
    (p : Program c) (u : State KernelCarrier.Node) (t : Fin c.N) (i : Lane) :
    (regional b read p u).source t.val i=(logical b read p).source t.val i := by
  have h := native_dense b read u "S" (by decide) (KernelProgram.address c p t.val i.1.val i.2.1.val)
    (by rw [matrix_count b "S" (by decide) (by decide) (by decide) (by decide)]; exact KernelIndex.matrix_inside c p t i.1 i.2.1)
  simpa [regional,KernelWords.inputs,KernelReference.read,logical,rawCell,KernelProgram.raw,KernelExecution.dtype,toFloat] using h
theorem forward_seed (b : Buffers c s .forward seed) (read : Nat → KernelCarrier.Word)
    (p : Program c) (u : State KernelCarrier.Node) (i : Lane) :
    (regional b read p u).seed i=(logical b read p).seed i := by
  simpa [regional,KernelWords.inputs,KernelReference.read,logical,rawCell,KernelProgram.raw,KernelExecution.dtype,toFloat]
    using native_seed b read u p i.1 i.2.1
theorem backward_incoming (b : Buffers c s .backward seed) (read : Nat → KernelCarrier.Word)
    (p : Program c) (u : State KernelCarrier.Node) (t : Fin c.N) (i : Lane) :
    (regional b read p u).incoming t.val i=(logical b read p).incoming t.val i := by
  have h := native_dense b read u "DS" (by decide) (KernelProgram.address c p t.val i.1.val i.2.1.val)
    (by rw [matrix_count b "DS" (by decide) (by decide) (by decide) (by decide)]; exact KernelIndex.matrix_inside c p t i.1 i.2.1)
  simpa [regional,KernelWords.inputs,KernelReference.read,logical,rawCell,KernelProgram.raw,KernelExecution.dtype,toFloat] using h
theorem backward_saved (b : Buffers c s .backward seed) (read : Nat → KernelCarrier.Word)
    (p : Program c) (u : State KernelCarrier.Node) (t : Fin c.N) (i : Lane) :
    (regional b read p u).saved t.val i=(logical b read p).saved t.val i := by
  have h := native_dense b read u "S" (by decide) (KernelProgram.address c p t.val i.1.val i.2.1.val)
    (by rw [matrix_count b "S" (by decide) (by decide) (by decide) (by decide)]; exact KernelIndex.matrix_inside c p t i.1 i.2.1)
  simpa [regional,KernelWords.inputs,KernelReference.read,logical,rawCell,KernelProgram.raw,KernelExecution.dtype,toFloat] using h

#axiomsClean gate
#axiomsClean forward_source
#axiomsClean forward_seed
#axiomsClean backward_incoming
#axiomsClean backward_saved
end
end VeriTile.Bench.Repairs.ChunkGateRecurrence.KernelTensorInputs
