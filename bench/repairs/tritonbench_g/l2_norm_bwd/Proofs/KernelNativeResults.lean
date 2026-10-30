import bench.repairs.tritonbench_g.l2_norm_bwd.Proofs.RawKernel
namespace VeriTile.Bench.Repairs.L2NormBwd.KernelNativeResults
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.SwigluTriton.Memory
open VeriTile.Bench.Repairs.SwigluTriton
noncomputable section

theorem kernel_native_raw {M β : Type} (ops : Ops M Native.Word β) (instructions : Native.Instructions)
    (layout : RawKernel.Layout) (valid : layout.Valid) (eps : Native.Word) (before : M) (row : Nat) (hr : row<layout.physical.x.rows) :
    ∃ after, FP.Structural.exec (Native.algebra instructions layout.physical.block) (source layout.physical.strides layout.physical.x.cols layout.physical.block)
      (Tile.rowState row (Native.atom eps) (Bridge.sourceState ops Native.atom layout.physical before))=some after ∧
      (∀ col, col<layout.physical.x.cols → ((after.mem "DX" (row*layout.physical.dx.rowStride+col*layout.physical.dx.colStride)).read .real).value (col%128) =
        Native.row instructions layout.physical.x.cols layout.physical.block eps (Physical.inputs ops before layout.physical.x row) (Physical.inputs ops before layout.physical.dy row) col) ∧
      Kernel.RowFrame layout.physical row (Bridge.sourceState ops Native.atom layout.physical before) after :=
  RawKernel.native_row ops instructions layout.physical (layout.valid valid) eps before row hr

theorem grid_native_raw {M : Type} (ops : Ops M Native.Word Words.Byte) (codec : Words.Codec ops)
    (instructions : Native.Instructions) (layout : RawKernel.Layout) (valid : layout.Valid) (eps : Native.Word) (before : M) :
    Kernel.Certificates ops (Native.algebra instructions layout.physical.block) Native.atom (fun c n => n.value (c%128))
      layout.physical (Native.atom eps) (List.range layout.physical.x.rows) before ∧
      let after := Bridge.sourceRows ops (Native.algebra instructions layout.physical.block) Native.atom (fun c n => n.value (c%128))
        layout.physical (Native.atom eps) (List.range layout.physical.x.rows) before
      (∀ row col, row<layout.physical.x.rows → col<layout.physical.x.cols → ops.read after (layout.physical.dx.key row col) =
        Native.row instructions layout.physical.x.cols layout.physical.block eps (Physical.inputs ops before layout.physical.x row) (Physical.inputs ops before layout.physical.dy row) col ∧
        ∀ byte, byte<4 → ops.observe after (layout.physical.dx.key row col).owner ((layout.physical.dx.key row col).address+byte) =
          (Native.row instructions layout.physical.x.cols layout.physical.block eps (Physical.inputs ops before layout.physical.x row) (Physical.inputs ops before layout.physical.dy row) col).extractLsb' (8*byte) 8) ∧
      Kernel.ByteFrame ops layout.physical before after :=
  RawKernel.native_grid ops codec instructions layout.physical (layout.valid valid) eps before

#axiomsClean kernel_native_raw
#axiomsClean grid_native_raw
end
end VeriTile.Bench.Repairs.L2NormBwd.KernelNativeResults
