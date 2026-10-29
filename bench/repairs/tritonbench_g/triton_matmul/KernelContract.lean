import bench.repairs.tritonbench_g.triton_matmul.Proofs.InputAdapter
namespace VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.KernelContract
open VeriTile.Triton FP.Structural FPValue
open Contract TensorBinding InputAdapter
noncomputable section

/-- The already existing destination passed to the selected kernel. These are
ordinary metadata, pointer-window and physical separation obligations. There
is no allocator, returned-object identity, numerical result or execution premise. -/
structure Output (a b c : Tensor) : Prop where
  metadata : Metadata a b c
  tensor : Laws (storage a.metadata) c
  window : Contract.Window c.metadata
  separateA : MatmulTritonAutotune.Whole.byteSpansDisjoint c.metadata.dataPtr.toNat
    (width (storage a.metadata)*c.metadata.numel) a.metadata.dataPtr.toNat
    (width (storage a.metadata)*a.metadata.numel)
  separateB : MatmulTritonAutotune.Whole.byteSpansDisjoint c.metadata.dataPtr.toNat
    (width (storage a.metadata)*c.metadata.numel) b.metadata.dataPtr.toNat
    (width (storage a.metadata)*b.metadata.numel)

/-- One shared physical reader admits overlapping A/B views and storage
offsets. The reader is unrestricted for Real and reads encoded words for native. -/
structure Inputs {α : Type} (read : Storage → Nat → α) (a b : Tensor) (s : State α) : Prop where
  left : ∀ i,i<a.metadata.numel → s.mem TensorBinding.inputA i=
    stored (storage a.metadata) (read (storage a.metadata)
      (a.metadata.dataPtr.toNat+width (storage a.metadata)*i))
  right : ∀ i,i<b.metadata.numel → s.mem TensorBinding.inputB i=
    stored (storage b.metadata) (read (storage b.metadata)
      (b.metadata.dataPtr.toNat+width (storage b.metadata)*i))

#specNonCircular Output avoiding [Source.typed,FP.Structural.exec]
#specNonCircular Inputs avoiding [Source.typed,FP.Structural.exec]
end
end VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.KernelContract
