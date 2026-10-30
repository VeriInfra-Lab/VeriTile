import bench.repairs.tritonbench_g.int8_matmul_quantization.Proofs.DirectData
namespace VeriTile.Bench.Repairs.Int8MatmulQuantization.Direct
noncomputable section

/-- The checked original data_ptr identifies every physical element of that
same storage slice. Logical row/column views are related by address lemmas. -/
def InputPointerBinding (base : Nat → Nat) (t : TensorView) : Prop :=
  ∀ delta,delta<t.info.metadata.numel → nativePointer t.info.metadata.dataPtr (bytes t.info) delta=
    base t.info.storage+bytes t.info*(t.info.offset+delta)
theorem input_pointer_binding (base : Nat → Nat) (t : TensorView) (laws : TensorLaws base t) : InputPointerBinding base t :=
  original_pointer base t laws

#axiomsClean input_pointer_binding
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization.Direct
