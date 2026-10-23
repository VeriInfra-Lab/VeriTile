import bench.repairs.tritonbench_g.bgmv_expand_slice_physical_tile_checked.Proofs.IOImplementation
import bench.repairs.tritonbench_g.bgmv_expand_slice_physical_tile_checked.Targets

/-! Retained single-source native correctness. This module declares ordinary
theorems only; a two-program FP equivalence is not established here. -/

namespace VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked
open VeriTile.Triton FP.Structural Direct Direct.FPValues
noncomputable section

/-- The same complete source grid over arbitrary native expressions, observed
as FP words. The reduction plan fixes topology; it is not a numerical law. -/
def nativeIO (h : Call) (plans : Native.Plans) (actual : Native.Instructions) :
    ProgramIO Native.Expr Native.Expr :=
  PublicIO.sourceIO Native.algebra (Native.reduction plans)
    (fun value => .bits (value.eval actual)) h

/-- Source-to-native observation under the five existing primitive instruction
laws. The full padded BK graph and all signed skip/add cases are retained.
There is no candidate transformation or native-equals-Real assertion. -/
theorem fp_equiv (h : Call) (plans : Native.Plans)
    (actual selected : Native.Instructions) (numeric : Native.LocalConformance actual selected) :
    (nativeIO h plans actual).ImplementsOn PublicIO.Valid (PublicNative.formula h plans selected) := by
  rw [Native.instructions_eq actual selected numeric]
  exact PublicIO.source_implements Native.algebra (Native.reduction plans)
    (fun value => .bits (value.eval selected)) h

#axiomsClean fp_equiv
#stmtSurfaceSubset fp_equiv ⊆
  [nativeIO, ProgramIO.ImplementsOn, PublicIO.Valid, PublicNative.formula, Call, Native.Plans, Native.Instructions, Native.LocalConformance, Native.Expr, Native.instInhabitedExpr]
#specNonCircular PublicNative.formula avoiding [Signed.Grid.run,FP.Structural.exec,FPGrid.fold,Surface.source]
#specNonCircular Native.LocalConformance avoiding [PublicNative.formula,Native.Expr.eval]
end
end VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked
