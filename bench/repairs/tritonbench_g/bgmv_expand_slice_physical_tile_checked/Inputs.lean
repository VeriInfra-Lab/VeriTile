import bench.repairs.tritonbench_g.bgmv_expand_slice_physical_tile_checked.Native
namespace VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked
open VeriTile Triton
noncomputable section

def logicalWeight (r : Request) (heap : ByteMemory) (j : Nat) : Word :=
  half heap (r.w.storageBase+2*r.weights.physicalIndex j)

def logicalIndex (r : Request) (heap : ByteMemory) (j : Nat) : Int :=
  let p:=r.i.storageBase+indexBytes r.i*r.metadata.physicalIndex j
  if r.i.dtype=.other "int64" then (load32 heap (p+4) ++ load32 heap p).toInt
  else (load32 heap p).toInt

theorem index_input (r : Request) (heap : ByteMemory) (h : Admitted r heap)
    (j : Nat) (hj : j<r.i.numel) : indices r heap j=logicalIndex r heap j := by
  unfold logicalIndex
  rw [raw_pointer r.metadata (checked r heap h).dense.2.2.2.1 j hj]
  rfl

theorem weight_input (r : Request) (heap : ByteMemory) (h : Admitted r heap)
    (ops : Native.Instructions) (j : Nat) (hj : j<r.w.numel) :
    ((Native.inputs r heap).weights j).eval ops=logicalWeight r heap j := by
  unfold logicalWeight
  rw [raw_pointer r.weights (checked r heap h).dense.2.1.1 j hj]
  rfl

theorem fp32_input (r : Request) (heap : ByteMemory) (h : Admitted r heap)
    (ops : Native.Instructions) (j : Nat) (hj : j<r.x.numel) :
    Torch.FP32.logicalValue (fun a => ⟨load32 heap a⟩) r.input.tensor r.input.laws j=
      ⟨((Native.inputs r heap).input j).eval ops⟩ := by
  rw [fp32_logical r.input heap (checked r heap h).dense.1.2.1,
    logical_direct r.input _ (checked r heap h).dense.1.1 j hj]
  rfl

theorem old_output_input (r : Request) (heap : ByteMemory) (h : Admitted r heap)
    (ops : Native.Instructions) (j : Nat) (hj : j<r.y.numel) :
    Torch.FP32.logicalValue (fun a => ⟨load32 heap a⟩) r.output.tensor r.output.laws j=
      ⟨((Native.inputs r heap).output j).eval ops⟩ := by
  rw [fp32_logical r.output heap (checked r heap h).dense.2.2.1.2.1,
    logical_direct r.output _ (checked r heap h).dense.2.2.1.1 j hj]
  rfl

structure NativeReads (r : Request) (heap : ByteMemory) (ops : Native.Instructions) : Prop where
  input : ∀ j,j<r.x.numel → Torch.FP32.logicalValue (fun a => ⟨load32 heap a⟩)
    r.input.tensor r.input.laws j=⟨((Native.inputs r heap).input j).eval ops⟩
  weight : ∀ j,j<r.w.numel → ((Native.inputs r heap).weights j).eval ops=logicalWeight r heap j
  oldOutput : ∀ j,j<r.y.numel → Torch.FP32.logicalValue (fun a => ⟨load32 heap a⟩)
    r.output.tensor r.output.laws j=⟨((Native.inputs r heap).output j).eval ops⟩
  metadata : ∀ j,j<r.i.numel → (Native.inputs r heap).indices j=logicalIndex r heap j

theorem native_reads (r : Request) (heap : ByteMemory) (h : Admitted r heap) (ops : Native.Instructions) :
    NativeReads r heap ops :=
  ⟨fp32_input r heap h ops,weight_input r heap h ops,old_output_input r heap h ops,index_input r heap h⟩

#axiomsClean native_reads
#specNonCircular NativeReads avoiding [Native.execute,Native.reference]
end
end VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked
