import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.Address
import bench.repairs.tritonbench_g.layernorm_fwd_triton.Proofs.Memory

namespace VeriTile.Bench.Repairs.NestedLoopsProcessing.Shared.Address
open VeriTile.Bench.Repairs.LayernormFwdTriton.Memory
structure Dense (count : Nat) where
  base : Nat
  finite : base+count*4≤2^64
  aligned : base%4=0

def Dense.address (a : Dense count) (i : Nat) : Nat := a.base+i*4

def Dense.contains (a : Dense count) (i : Nat) : Prop := a.base ≤ i ∧ i < a.base+count*4

theorem count_bound (a : Dense count) : count≤2^62 := by
  have h:=a.finite
  omega

theorem address_bytes (a : Dense count) (i : Nat) (hi : i<count) (b : Nat) (hb : b<4) :
    a.contains (a.address i+b) ∧ a.address i+b<2^64 := by
  have h:=a.finite
  simp only [Dense.contains,Dense.address]
  omega

theorem address_apart (a : Dense count) {i j : Nat} (h : i≠j) :
    Apart (a.address i) (a.address j) := by
  unfold Apart Dense.address
  omega


end VeriTile.Bench.Repairs.NestedLoopsProcessing.Shared.Address
