import bench.repairs.tritonbench_g.bgmv_expand_slice_physical_tile_checked.IO
import bench.repairs.tritonbench_g.bgmv_expand_slice_physical_tile_checked.Real
import bench.repairs.tritonbench_g.bgmv_expand_slice_physical_tile_checked.Native

namespace VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked
open VeriTile.Triton FP.Structural Direct Direct.FPValues
open scoped BigOperators
noncomputable section

namespace PublicMath
/-- GEMV on logical input and gathered weight coordinates, with the original
optional in-place output addition. Signed skip rows are outside the write mask. -/
def formula (h : Call) (xs : PublicIO.Inputs ℝ h) :=
  PublicIO.result h (fun b n =>
    let total := ∑ k : Fin h.K,PublicIO.input xs b k.val*PublicIO.weight xs b n k.val
    if h.add then total+PublicIO.previous xs b n else total)
end PublicMath

namespace PublicNative
/-- Native observation of the full padded BK product/reduction tree, including
FP32-to-FP16 narrowing, FP16 product/sum, widening and optional FP32 addition.
Inputs remain arbitrary expressions; no leaf or finite-value restriction. -/
def formula (h : Call) (plans : Native.Plans) (ops : Native.Instructions)
    (xs : PublicIO.Inputs Native.Expr h) :=
  PublicIO.result h (fun b n => Native.Expr.bits
    ((PublicIO.localValue Native.algebra (Native.reduction plans) h
      (PublicIO.input xs b) (PublicIO.weight xs b n) (PublicIO.previous xs b n)).eval ops))
end PublicNative

#specNonCircular PublicMath.formula avoiding [Signed.Grid.run,FP.Structural.exec,SourceGrid.fold,Surface.source]
#specNonCircular PublicNative.formula avoiding [Signed.Grid.run,FP.Structural.exec,SourceGrid.fold,Surface.source]
end
end VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked
