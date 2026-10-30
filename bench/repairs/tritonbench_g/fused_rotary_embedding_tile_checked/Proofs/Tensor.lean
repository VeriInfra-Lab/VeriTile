import bench.repairs.tritonbench_g.fused_rotary_embedding_tile_checked.Contract
import bench.repairs.tritonbench_g.fused_rotary_embedding.Proofs.NativeHost
import VeriTile.Torch.FP32

namespace VeriTile.Bench.Repairs.FusedRotaryEmbeddingTileChecked.TensorObservation
open VeriTile FusedRotaryEmbedding
noncomputable section
set_option maxHeartbeats 2000000
set_option linter.unusedSimpArgs false

/-- API descriptor and storage laws only: no execution or output equation. -/
structure Binding (observed : Checked.Tensor) (actual : Torch.FP32.Tensor) : Prop where
  laws : Torch.FP32.TensorLaws actual
  object : actual.object=observed.object
  shape : actual.metadata.shape=observed.shape
  device : actual.metadata.device=⟨"cuda",observed.device⟩
  contiguous : actual.metadata.isContiguous=observed.contiguous
  negative : actual.metadata.isNeg=observed.negative
  offset : actual.storageOffset=observed.offset
  base : actual.storage.base=observed.base
  pointer : actual.dataPtr=BitVec.ofNat 64 observed.base

def payloadRegion (r : Region) : Prop := r∈Checked.regions ∧ r≠.lengths ∧ r≠.table
@[simp] theorem output_payload (kind : Kind) : payloadRegion kind.output := by
  cases kind <;> simp [payloadRegion,Checked.regions,Kind.output]

structure Family (c : Checked.Call) where
  tensor : Region → Torch.FP32.Tensor
  binding : ∀ r,payloadRegion r → Binding (c.tensor r) (tensor r)

def read (m : Arena.Memory) (a : Nat) : Torch.FP32.Word :=
  ⟨Physical.decode (fun i=>m ⟨(a+i.val)%(2^64),Nat.mod_lt _ (by decide)⟩)⟩

def logical (c : Checked.Call) (family : Family c) (m : Arena.Memory)
    (r : Region) (hr : payloadRegion r) (i : Nat) : Native.Word :=
  (Torch.FP32.logicalValue (read m) (family.tensor r) (family.binding r hr).laws i).bits

theorem logical_read (c : Checked.Call) (family : Family c) (m : Arena.Memory)
    (metadata : Physical.Bytes) (admitted : Checked.check c metadata=Bool.true)
    (r : Region) (hr : payloadRegion r) (a : Nat) (ha : a<Checked.extent c r) :
    logical c family m r hr a=FusedRotaryEmbedding.Binding.read c m r a := by
  have observed:=Checked.ordinary_checked c metadata admitted r hr.1
  have contiguous : (c.tensor r).contiguous=Bool.true := observed.2.2.2.1
  have negative : (c.tensor r).negative=Bool.false := observed.2.2.2.2.2.1
  have offset : (c.tensor r).offset=0 := observed.2.2.2.2.2.2.1
  have bound : a<(family.tensor r).metadata.numel := by
    simpa [Torch.TensorMetadata.numel,(family.binding r hr).shape,Checked.extent,hr.1] using ha
  have address := (family.binding r hr).laws.contiguous
    ((family.binding r hr).contiguous.trans contiguous) a bound
  simp only [logical,Torch.FP32.logicalValue,(family.binding r hr).negative,negative,
    Bool.false_eq_true,ite_false,address,(family.binding r hr).offset,offset,Nat.zero_add,
    Torch.FP32.Storage.address,(family.binding r hr).base,read,
    FusedRotaryEmbedding.Binding.read,Arena.address,Nat.add_assoc]

theorem mirror_read (c : Checked.Call) (m : Arena.Memory) (r : Region) (a : Nat)
    (ha : a<Checked.extent c r) :
    Physical.load (Arena.mirror c m) r a=FusedRotaryEmbedding.Binding.read c m r a := by
  unfold Physical.load FusedRotaryEmbedding.Binding.read
  congr 1
  funext i
  simp only [Arena.mirror,show 4*a+i.val<4*Checked.extent c r by omega,ite_true]

theorem inputs (c : Checked.Call) (family : Family c) (m : Arena.Memory)
    (admitted : inputChecks c (Arena.mirror c m)=Bool.true)
    (r : Region) (hr : payloadRegion r) (a : Nat) (ha : a<Checked.extent c r) :
    logical c family m r hr a=Physical.load (Arena.mirror c m) r a := by
  rw [logical_read c family m _ (original_checks c _ admitted) r hr a ha,mirror_read c m r a ha]

theorem outputs (c : Checked.Call) (family : Family c) (m : Arena.Memory)
    (admitted : inputChecks c (Arena.mirror c m)=Bool.true) (callerDevice : Nat)
    (p : Native.Instructions) (result : FusedRotaryEmbedding.Binding.Result)
    (properties : FusedRotaryEmbedding.Binding.NativeResult c callerDevice p p m result)
    (kind : Kind) (t h d : Nat) (ht : t<c.shape.tokens) (hh : h<c.shape.heads kind)
    (hd : d<c.shape.width) :
    logical c family result.memory kind.output (output_payload kind)
      (logicalOutput (Checked.host c (Arena.mirror c m)).config kind t h d)=
        wordFormula c.shape p (Physical.load (Arena.mirror c m)) kind t h d := by
  have checked:=original_checks c _ admitted
  rw [logical_read c family result.memory _ checked _ _ _
    (FusedRotaryEmbedding.Binding.logical_extent c _ checked kind t h d ht hh hd)]
  exact properties.values kind t h d ht hh hd

theorem frame (c : Checked.Call) (family : Family c) (m : Arena.Memory)
    (admitted : inputChecks c (Arena.mirror c m)=Bool.true) (callerDevice : Nat)
    (p : Native.Instructions) (result : FusedRotaryEmbedding.Binding.Result)
    (properties : FusedRotaryEmbedding.Binding.NativeResult c callerDevice p p m result)
    (r : Region) (hr : payloadRegion r) (a : Nat) (ha : a<Checked.extent c r)
    (outside : ∀ i : Fin 4,Arena.Outside c (Arena.mirror c m) (Arena.address c r (4*a+i.val))) :
    logical c family result.memory r hr a=logical c family m r hr a := by
  rw [logical_read c family result.memory _ (original_checks c _ admitted) r hr a ha,
    logical_read c family m _ (original_checks c _ admitted) r hr a ha]
  unfold FusedRotaryEmbedding.Binding.read
  congr 1
  funext i
  exact properties.frame _ (outside i)

#axiomsClean logical_read
#axiomsClean inputs
#axiomsClean outputs
#axiomsClean frame
#specNonCircular Binding avoiding [actualSource,Arena.run]
end
end VeriTile.Bench.Repairs.FusedRotaryEmbeddingTileChecked.TensorObservation
