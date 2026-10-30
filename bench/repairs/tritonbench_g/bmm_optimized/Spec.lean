import bench.repairs.tritonbench_g.bmm_optimized.Kernels

namespace VeriTile.Bench.Repairs.BmmOptimized
open scoped BigOperators

/-- Independent initial-array batched matrix product, without tiling, grid,
mask flags, execution, or an output-valued assumption. -/
noncomputable def mathematical (s : Shape) (m : Memory ℝ) (x : Cell) : ℝ :=
  ∑ k ∈ Finset.range s.k, m .a (s.a x.batch x.row k) * m .b (s.b x.batch k x.column)

/-- Independent flat ordered scalar chain. It preserves every padded zero
operation through the final 32-wide tile; no exceptional-word padding identity
is assumed for native arithmetic. -/
def reference {α : Type} (s : Shape) (p : Scalar α) (m : Memory α) (x : Cell) : Nat → α
  | 0 => p.zero
  | k+1 => p.fma
      (if k<s.k then m .a (s.a x.batch x.row k) else p.zero)
      (if k<s.k then m .b (s.b x.batch k x.column) else p.zero)
      (reference s p m x k)
def ordered {α : Type} (s : Shape) (p : Scalar α) (m : Memory α) (x : Cell) :=
  reference s p m x (((s.k+31)/32)*32)

#specNonCircular mathematical avoiding [source,sourceValue,sourceProgram,blockLoop,laneLoop]
#specNonCircular ordered avoiding [source,sourceValue,sourceProgram,blockLoop,laneLoop]
end VeriTile.Bench.Repairs.BmmOptimized
