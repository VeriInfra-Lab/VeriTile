import bench.repairs.tritonbench_g.l2_norm_bwd.Proofs.IOGrid
import bench.repairs.tritonbench_g.l2_norm_bwd.Proofs.RawKernel

/-! The existing physical/byte results and public regional IO share the exact
same Tile.sourceRun certificate. Its output-injectivity condition follows from
the already admitted physical layout; no new Tensor assertion is needed. -/
namespace VeriTile.Bench.Repairs.L2NormBwd.PublicIO
open VeriTile.Triton FP.Structural
noncomputable section

theorem physical_unique (layout : RawKernel.Layout) (valid : layout.Valid) :
    OutputUnique layout.physical.strides layout.x.rows layout.x.cols := by
  intro row col row' col' hr hc hr' hc' equal
  apply valid.dxUnique row col row' col'
    (by rw [valid.dxRows];exact hr) (by rw [valid.dxCols];exact hc)
    (by rw [valid.dxRows];exact hr') (by rw [valid.dxCols];exact hc')
  change row*layout.dx.rowStride+col*layout.dx.colStride=
    row'*layout.dx.rowStride+col'*layout.dx.colStride at equal
  simp only [RawKernel.Matrix.key,equal,Nat.add_assoc]

theorem physical_invocation {A : Type} [Inhabited A] (M : Algebra A)
    (layout : RawKernel.Layout) (row : Nat) (eps : A) (s : State A) :
    (invocation M layout.physical.strides layout.x.cols layout.block row eps).run s =
      FP.Structural.exec M (source layout.physical.strides layout.x.cols layout.block)
        (Tile.rowState row eps s) := rfl

#axiomsClean physical_unique
#axiomsClean physical_invocation
end
end VeriTile.Bench.Repairs.L2NormBwd.PublicIO
