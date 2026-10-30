import bench.tritonbench_g.nested_loops_processing.Proofs.Shared.Address

namespace VeriTile.Bench.Repairs.NestedLoopsProcessing.Shared.Execution
open Address
open VeriTile.Bench.Repairs.LayernormFwdTriton.Memory
def Disjoint (a : Dense p) (b : Dense q) : Prop :=
  a.base+p*4≤b.base ∨ b.base+q*4≤a.base

theorem disjoint_words (a : Dense p) (b : Dense q) (h : Disjoint a b)
    (i j : Nat) (hi : i<p) (hj : j<q) : Apart (a.address i) (b.address j) := by
  unfold Disjoint at h
  unfold Apart Dense.address
  omega


end VeriTile.Bench.Repairs.NestedLoopsProcessing.Shared.Execution
