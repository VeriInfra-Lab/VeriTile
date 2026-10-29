import bench.repairs.tritonbench_g.l2_norm_bwd.Proofs.IOGrid
import bench.repairs.tritonbench_g.l2_norm_bwd.Proofs.Native

/-! Retained single-source native correctness. These ordinary theorems keep
the original names, full contracts and proofs; they do not assert an
implementation-pair floating-point equivalence. -/

namespace VeriTile.Bench.Repairs.L2NormBwd.FPEquiv
open VeriTile.Triton FP.Structural
noncomputable section

/-- All FP32 input words are admitted. The existing source backend keeps its
reduction/fusion provenance; each output is observed at its actual thread lane. -/
def nativeIO (instructions : Native.Instructions) (layout : Strides)
    (rows N block : Nat) (eps : Native.Word) : ProgramIO Native.Word Native.Node :=
  PublicIO.sourceIO (Native.algebra instructions block) Native.atom
    (fun col value => value.value (col%128)) layout rows N block (Native.atom eps)

/-- Independent ordered native graph for the complete source grid, including
both reductions, sqrt.approx.ftz, div.full and the final neg/FMA. This is a
source-to-native observation, with no candidate or numerical rewrite premise. -/
theorem grid_native (instructions : Native.Instructions) (layout : Strides)
    (rows N block : Nat) (cover : N≤block) (unique : PublicIO.OutputUnique layout rows N)
    (eps : Native.Word) :
    (nativeIO instructions layout rows N block eps).Implements
      (fun xs => PublicIO.result rows N (fun row col =>
        Native.row instructions N block eps (PublicIO.rowValues xs Bool.false row)
          (PublicIO.rowValues xs Bool.true row) col.val)) := by
  apply PublicIO.implements _ _ _ _ _ _ _ cover unique
  intro xs ys col _
  exact Native.source_graph_lowering instructions N block eps xs ys col

#specNonCircular Native.row avoiding [source,Tile.scalar,FP.Structural.exec]
#axiomsClean grid_native
end
end VeriTile.Bench.Repairs.L2NormBwd.FPEquiv
