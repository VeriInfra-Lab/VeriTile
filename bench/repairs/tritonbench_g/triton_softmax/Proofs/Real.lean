import bench.repairs.tritonbench_g.triton_softmax.Proofs.KernelReal
import bench.repairs.tritonbench_g.softmax_triton1.Proofs.Storage

namespace VeriTile.Bench.Repairs.TritonSoftmax.RealModel
open VeriTile.Triton
open VeriTile.Bench.Repairs.SoftmaxTriton1 hiding softmax_kernel numWarps
noncomputable section
set_option maxHeartbeats 4000000
theorem source_safe (layout : Layout) (x y : Tensor) (B p : Nat)
    (hx : layout.registered x) (hy : layout.registered y)
    (shape : SameShape x y) (hB : 0<B) (hp : p<x.rows) (s : BlockState) :
    Kernel.TraceSafe (fun r => (layout.buffer r).capacity/4)
      (softmax_kernel y.region x.region x.offset y.offset x.rowStride x.colStride
        y.rowStride y.colStride x.cols B).toAlgKernel (launchState x p s) := by
  have hin : ∀ i : Fin B,i.val<x.cols →
      x.offset+p*x.rowStride+i.val*x.colStride<(layout.buffer x.region).capacity/4 := by
    intro i hi
    have h := (layout.legal x hx).2.2.2.2.2.2.2 p i.val hp hi
    dsimp [Tensor.index] at h
    omega
  have hout : ∀ i : Fin B,i.val<x.cols →
      y.offset+p*y.rowStride+i.val*y.colStride<(layout.buffer y.region).capacity/4 := by
    intro i hi
    have h := (layout.legal y hy).2.2.2.2.2.2.2 p i.val (by rwa [shape.rows]) (by rwa [shape.cols])
    dsimp [Tensor.index] at h
    omega
  obtain ⟨k,rfl⟩ := Nat.exists_eq_succ_of_ne_zero hB.ne'
  unfold Kernel.TraceSafe
  simp [softmax_kernel,ComputeKernel.toAlgKernel,ComputeKernel.surfaceBody,ComputeOp.toAlgorithm?,ComputeExpr.toAlgorithm?,
    Stmt.TraceSafeList,Stmt.TraceSafe,Op.SafeAt.eq_def,MaskOpt.SafeAt,MemAccess.SafeAt,
    MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MaskOpt.Active,
    Op.PointerAddressesSafeOn,Op.MemorySafe,BlockState.setReg,
    stepStmt,evalOp.eq_def,Tile.bop,Tile.cop,Tile.ptrAdd,Tile.uop,
    Tile.reduceMax,Tile.reduceMaxDrop,Tile.reduceSum,Tile.reduceSumDrop,
    TileShape.axisDim,TileShape.eraseAxis,TileShape.insertAxisIndex,
    NumericDType.add,NumericDType.mul,NumericDType.sub,NumericDType.div,ComparableDType.lt,
    launchState,launchState,← Int.natCast_mul,← Int.natCast_add,-Nat.cast_mul,-Nat.cast_add]
  exact ⟨fun a ha => hin a ha,fun a ha => hout a ha⟩

#specNonCircular formula avoiding [evaluate,softmax_kernel]
#axiomsClean source_memory
#axiomsClean source
#axiomsClean terminates
#axiomsClean source_safe
end
end VeriTile.Bench.Repairs.TritonSoftmax.RealModel
