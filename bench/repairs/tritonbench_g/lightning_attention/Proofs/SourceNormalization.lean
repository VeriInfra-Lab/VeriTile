import bench.repairs.tritonbench_g.lightning_attention.Proofs.AddressBounds
namespace VeriTile.Bench.Repairs.LightningAttention.SourceNormalization
open VeriTile Triton
/-- Signed source subtraction and direct comparison have the same mask. -/
theorem signed_causal (i j : Int) : (i-j≥0) ↔ i≥j := by omega

/-- Every source descending loop visits exactly the reverse of range(n). -/
theorem reverse_range (n : Nat) : (List.range n).reverse=(List.range n).map (fun r => n-1-r) := by
  have h := @List.reverse_range' 0 n
  simpa only [Nat.zero_add,← List.range_eq_range'] using h

theorem reverse_index (n r : Nat) (hr : r<n) : n-1-r<n ∧ n-1-(n-1-r)=r := by omega
/-- The only use of the deleted descending outer iterator is its skip test. -/
theorem reverse_skip (n r : Nat) (hr : r<n) : (n-1-r<n-1) ↔ r>0 := by omega
/-- The initial pointer is decremented before each subblock. Its row therefore
corresponds to the source descending i,j order, including the final subblock. -/
theorem reverse_row (n r sub : Nat) (hr : r<n) (hs : sub<2) :
    n*64-(r*2+sub+1)*32=(n-1-r)*64+(1-sub)*32 := by omega

/-- Reindexing preserves source-order iteration, not just a set of indices. -/
theorem reverse_fold {α : Type} (n : Nat) (step : α → Nat → α) (init : α) :
    (List.range n).reverse.foldl step init=
      (List.range n).foldl (fun state r => step state (n-1-r)) init := by
  rw [reverse_range,List.foldl_map]
#axiomsClean signed_causal
#axiomsClean reverse_range
#axiomsClean reverse_index
#axiomsClean reverse_skip
#axiomsClean reverse_row
#axiomsClean reverse_fold
end VeriTile.Bench.Repairs.LightningAttention.SourceNormalization
