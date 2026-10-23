import bench.repairs.tritonbench_g.layer_norm_welfold.Proofs.KernelGraphResult

namespace VeriTile.Bench.Repairs.LayerNormWelfold.FPEquiv
open VeriTile.Triton FP.Structural Reuse
noncomputable section

/-- Native words are embedded injectively as .bits terms, preserving the
original unrestricted graph-term inputs and every Word-valued observation. -/
abbrev nativeIO (I : NativeKernel.Instructions) (P : NativeKernel.Profile)
    (Y Mean Rstd : RegionName) (S D : Nat) (c : Configuration) :=
  IO.layerNormIO NativeKernel.algebra (IOResults.observed I P) Y Mean Rstd S D c.block c.tree

/-- Independent native recurrence under the retained primitive and compiler
profile. No reassociation, native=Real identity or new numerical law is used. -/
def nativeFormula (I : NativeKernel.Instructions) (P : NativeKernel.Profile)
    (D : Nat) (c : Configuration) (x w bias : Nat → NativeKernel.Term) (row : Nat) :
    NativeKernel.Term × NativeKernel.Term × (Nat → NativeKernel.Term) :=
  let ref := NativeKernel.reference I P x w bias D c.block row c.tree
  (.bits ref.mean,.bits ref.inverse,fun col => .bits (ref.output col))

theorem kernel_native (I : NativeKernel.Instructions) (P : NativeKernel.Profile)
    (S D : Nat) (Y Mean Rstd : RegionName) (c : Configuration)
    (sep : Separation "X" "W" "Bias" Y Mean Rstd) :
    (nativeIO I P Y Mean Rstd S D c).Implements
      (IO.result Y Mean Rstd S D (nativeFormula I P D c)) :=
  IOResults.native I P Y Mean Rstd S D c sep

#specNonCircular nativeFormula avoiding [Source.rowKernel,Source.launch,meanValue,inverseValue,rowOutput]
#axiomsClean kernel_native
#stmtSurfaceSubset kernel_native ⊆
  [RegionName,Region.mk,TileDType.real,Configuration,nativeIO,nativeFormula,IO.result,ProgramIO.Implements,
   NativeKernel.Term,NativeKernel.Instructions,NativeKernel.Profile,
   VeriTile.Bench.TritonBenchG.FusedLayernormTriton.Whole.Separation,
   VeriTile.Bench.Repairs.SoftmaxFlaggems.Native.instInhabitedTerm]
end
end VeriTile.Bench.Repairs.LayerNormWelfold.FPEquiv
