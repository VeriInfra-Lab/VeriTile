import bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.IOResults

namespace VeriTile.Bench.Repairs.LayerNormWelfold.FPEquiv
open VeriTile.Triton FP.Structural Reuse
noncomputable section

/-- Independent ordered two-pass graph, retaining arbitrary logical input
terms and the original topology-only reduction tree. -/
def graphFormula (D : Nat) (c : Configuration) (x w bias : Nat → NativeKernel.Term)
    (row : Nat) : NativeKernel.Term × NativeKernel.Term × (Nat → NativeKernel.Term) :=
  let inputs := KernelGraph.logicalInputs NativeKernel.algebra x w bias D row
  (Native.eval NativeKernel.constructors inputs (Native.meanExpr D c.block c.tree),
   Native.eval NativeKernel.constructors inputs (Native.inverseExpr D c.block c.tree),
   fun col => Native.eval NativeKernel.constructors inputs (Native.outputExpr D c.block col c.tree))

abbrev graphIO (Y Mean Rstd : RegionName) (S D : Nat) (c : Configuration) :=
  IO.layerNormIO NativeKernel.algebra id Y Mean Rstd S D c.block c.tree

/-- The exact source graph is retained as a supporting public logical result. -/
theorem kernel_graph (S D : Nat) (Y Mean Rstd : RegionName) (c : Configuration)
    (sep : Separation "X" "W" "Bias" Y Mean Rstd) :
    (graphIO Y Mean Rstd S D c).Implements (IO.result Y Mean Rstd S D (graphFormula D c)) :=
  IOResults.graph Y Mean Rstd S D c sep

#axiomsClean kernel_graph
#stmtSurfaceSubset kernel_graph ⊆
  [RegionName,Region.mk,TileDType.real,Configuration,graphIO,graphFormula,IO.result,ProgramIO.Implements,
   NativeKernel.Term,VeriTile.Bench.TritonBenchG.FusedLayernormTriton.Whole.Separation,
   VeriTile.Bench.Repairs.SoftmaxFlaggems.Native.instInhabitedTerm]
end
end VeriTile.Bench.Repairs.LayerNormWelfold.FPEquiv
