import bench.repairs.tritonbench_g.block_sparse_attn.KernelOnly.Compatibility

namespace VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs Bool.false

/-- The finite MMA chain selects each operand at its physical key owner. -/
theorem chain_selected (a : BlockSparseAttn.Ops α) (literal argument : ℝ → α)
    (left : Fin 16 → Replicas α) (right : Nat → α) (seed : α) :
    chain (Arithmetic.ofLegacy a literal argument) left (fun i : Fin 16 => broadcast (right i.val)) seed=
      BlockSparseAttn.dot a .mma16n8k8 16
        (fun j => if h : j<16 then left ⟨j,h⟩ (owner j) else a.zero) right seed := by
  have zeroBound (i : Fin 8) : i.val<16 := by omega
  have oneBound (i : Fin 8) : 8+i.val<16 := by omega
  simp [chain,fragment,broadcast,Arithmetic.ofLegacy,BlockSparseAttn.dot,
    List.range_succ,zeroBound,oneBound]
  delta fragment
  simp [broadcast,zeroBound,oneBound]

theorem reduceLine_selected (a : BlockSparseAttn.Ops α) (literal argument : ℝ → α)
    (op : α → α → α) (x : Fin 16 → Replicas α) (o : Fin 4) :
    reduceLine (Arithmetic.ofLegacy a literal argument) op x o=
      (BlockSparseAttn.SM90B16.tree o).eval op (fun j => x j (owner j.val)) := by
  simp only [reduceLine,dite_true]
  rw [Tree.eval_legacy,selected_tree]

#axiomsClean chain_selected
#axiomsClean reduceLine_selected
end
end VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly
