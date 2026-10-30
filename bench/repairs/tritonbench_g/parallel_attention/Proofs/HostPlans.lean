import bench.repairs.tritonbench_g.parallel_attention.Proofs.Graph

namespace VeriTile.Bench.Repairs.ParallelAttention
noncomputable section

/-- Exact zero-seeded left-to-right host shard addition syntax. This is a
candidate lowering for the explicit host-loop proposal, not a claim about
arbitrary torch.sum implementations. -/
def serialSum : Nat → SumPlan
  | 0 => .zero
  | n+1 => .add (serialSum n) (.input n)

theorem serialSum_leaves (n : Nat) : (serialSum n).leaves=List.range n := by
  induction n with
  | zero => rfl
  | succ n ih => simp [serialSum,SumPlan.leaves,ih,List.range_succ]

theorem serialSum_complete (n : Nat) : (serialSum n).Covers n := by
  rw [SumPlan.Covers,serialSum_leaves]

theorem serialSum_eval (a : Ops α) (x : Nat → α) (n : Nat) :
    (serialSum n).eval a x=foldN (fun i carry => a.add carry (x i)) n a.zero := by
  induction n with
  | zero => rfl
  | succ n ih => simp only [serialSum,SumPlan.eval,foldN,ih]

#axiomsClean serialSum_complete
#axiomsClean serialSum_eval
end
end VeriTile.Bench.Repairs.ParallelAttention
