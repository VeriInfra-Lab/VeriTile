import bench.tritonbench_g.max_reduction.IO
import bench.tritonbench_g.max_reduction.Proofs.Real

namespace VeriTile.Bench.Repairs.MaxReduction.DimensionMath
open VeriTile.Triton FP.Structural
noncomputable section

abbrev Inputs (inp : RegionName) (BM BN : Nat) := ChannelValues ℝ [⟨.real,[BM,BN],inp⟩]
abbrev Outputs (ov oi : RegionName) (BM : Nat) := ChannelValues ℝ [⟨.real,[BM],ov⟩,⟨.int,[BM],oi⟩]

/-- Logical original-axis input, without physical memory or a source reducer. -/
def input {inp : RegionName} {BM BN : Nat} (xs : Inputs inp BM BN) (i : Fin BM) (n : Nat) : ℝ :=
  if h : n<BN then xs 0 (i,⟨n,h⟩,PUnit.unit) else 0

/-- The nonempty axis returns an attained maximum and the least tied original
index. The existing empty-axis projection returns value zero and index zero. -/
def result (inp ov oi : RegionName) (M N BM BN p : Nat)
    (xs : Inputs inp BM BN) (ys : Outputs ov oi BM) : Prop :=
  ∀ i : Fin BM,p*BM+i.val<M →
    if N=0 then ys 0 (i,PUnit.unit)=(0 : ℝ) ∧ ys 1 (i,PUnit.unit)=(0 : Int)
    else ∃ index : Nat,ys 1 (i,PUnit.unit)=(index : Int) ∧
      RealModel.IsLeastArgmax (input xs i) N (ys 0 (i,PUnit.unit)) index

#specNonCircular result avoiding [MaxReduction.dimension,Source.Engine.algebra,Source.rowValue,FP.Structural.exec]
end
end VeriTile.Bench.Repairs.MaxReduction.DimensionMath
