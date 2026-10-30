import bench.repairs.tritonbench_g.l2_norm_bwd.Proofs.LogicalIO

namespace VeriTile.Bench.Repairs.L2NormBwd.Correct
open VeriTile.Triton FP.Structural
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section

/-- With q=sum(x²)+eps and dot=sum(dy*x), every output is
`dy/sqrt(q) - dot*x/(q*sqrt(q))`, in the original independent formula.
The complete strided output and exact exterior frame follow for all rows. -/
specification grid_correctness (layout : Strides) (rows N block : Nat)
    (cover : N≤block) (unique : PublicIO.OutputUnique layout rows N) (eps : ℝ) :
    Spec.Real (LogicalIO.io Math.algebra layout rows N block eps ⊨ fun a row col =>
      let q := (∑ j : Fin N, a.x row j*a.x row j)+eps
      let dot := ∑ j : Fin N, a.dy row j*a.x row j
      a.dy row col*(1/Real.sqrt q)-dot*(1/q)*(1/Real.sqrt q)*a.x row col) :=
  LogicalIO.implements layout rows N block cover unique eps

#specNonCircular Math.derivative avoiding [source,Tile.scalar,FP.Structural.exec]
#axiomsClean grid_correctness
end
end VeriTile.Bench.Repairs.L2NormBwd.Correct
