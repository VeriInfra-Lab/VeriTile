import bench.repairs.tritonbench_g.l2_norm_bwd.Proofs.RawKernel
namespace VeriTile.Bench.Repairs.L2NormBwd.KernelRealResults
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.SwigluTriton.Memory
open VeriTile.Bench.Repairs.SwigluTriton
noncomputable section

theorem kernel_correctness_raw {M β : Type} (ops : Ops M ℝ β)
    (layout : RawKernel.Layout) (valid : layout.Valid) (eps : ℝ) (before : M) (row : Nat) (hr : row<layout.physical.x.rows) :
    ∃ after, FP.Structural.exec Math.algebra (source layout.physical.strides layout.physical.x.cols layout.physical.block)
      (Tile.rowState row eps (Bridge.sourceState ops id layout.physical before))=some after ∧
      (∀ col, col<layout.physical.x.cols → (after.mem "DX" (row*layout.physical.dx.rowStride+col*layout.physical.dx.colStride)).read .real =
        Math.derivative layout.physical.x.cols eps (Physical.inputs ops before layout.physical.x row) (Physical.inputs ops before layout.physical.dy row) col) ∧
      Kernel.RowFrame layout.physical row (Bridge.sourceState ops id layout.physical before) after :=
  RawKernel.real_row ops layout.physical (layout.valid valid) eps before row hr

theorem grid_correctness_raw {M β : Type} (ops : Ops M ℝ β)
    (layout : RawKernel.Layout) (valid : layout.Valid) (eps : ℝ) (before : M) :
    Kernel.Certificates ops Math.algebra id (fun _ => id) layout.physical eps (List.range layout.physical.x.rows) before ∧
      let after := Bridge.sourceRows ops Math.algebra id (fun _ => id) layout.physical eps (List.range layout.physical.x.rows) before
      (∀ row col, row<layout.physical.x.rows → col<layout.physical.x.cols → ops.read after (layout.physical.dx.key row col) =
        Math.derivative layout.physical.x.cols eps (Physical.inputs ops before layout.physical.x row) (Physical.inputs ops before layout.physical.dy row) col) ∧ Kernel.ByteFrame ops layout.physical before after :=
  RawKernel.real_grid ops layout.physical (layout.valid valid) eps before

#axiomsClean kernel_correctness_raw
#axiomsClean grid_correctness_raw
end
end VeriTile.Bench.Repairs.L2NormBwd.KernelRealResults
