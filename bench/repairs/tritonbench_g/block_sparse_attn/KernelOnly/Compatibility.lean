import bench.repairs.tritonbench_g.block_sparse_attn.KernelOnly.Proofs.Reference
import bench.repairs.tritonbench_g.block_sparse_attn.Proofs.Reduction

namespace VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 6000000
set_option linter.unusedSimpArgs Bool.false

/-- Reuse the already proved independent row mathematics with only its scalar
and fixed-width instruction parameters. No SourceBinding object is involved. -/
def Arithmetic.ofLegacy (a : BlockSparseAttn.Ops α) (literal argument : ℝ → α) : Arithmetic α where
  zero := a.zero
  negInf := a.negInf
  add := a.add
  sub := a.sub
  mul := a.mul
  fma := a.fma
  max := a.max
  exp := a.exp
  div := a.div
  mma8 := fun _ => a.mma8 .mma16n8k8
  literal := literal
  argument := argument

def Tree.toLegacy : Tree n → BlockSparseAttn.Tree n
  | .leaf i => .leaf i
  | .merge l r => .merge l.toLegacy r.toLegacy
theorem Tree.eval_legacy (tree : Tree n) (op : α → α → α) (values : Fin n → α) :
    tree.eval op values = tree.toLegacy.eval op values := by
  induction tree with
  | leaf i => rfl
  | merge l r hl hr => simp only [Tree.eval,Tree.toLegacy,BlockSparseAttn.Tree.eval,hl,hr]
theorem selected_tree (o : Fin 4) :
    (SM90B16.tree o).toLegacy=BlockSparseAttn.SM90B16.tree o := by rfl
theorem selected_owner (column : Nat) : owner column=BlockSparseAttn.owner column := rfl

theorem chain_broadcast (a : BlockSparseAttn.Ops α) (literal argument : ℝ → α)
    (left right : Nat → α) (seed : α) :
    chain (Arithmetic.ofLegacy a literal argument)
      (fun i : Fin 16 => broadcast (left i.val))
      (fun i : Fin 16 => broadcast (right i.val)) seed =
      BlockSparseAttn.dot a .mma16n8k8 16 left right seed := by
  have zeroBound (i : Fin 8) : i.val<16 := by omega
  have oneBound (i : Fin 8) : 8+i.val<16 := by omega
  simp [chain,fragment,broadcast,Arithmetic.ofLegacy,BlockSparseAttn.dot,
    List.range_succ,zeroBound,oneBound]
  delta fragment
  simp [broadcast,zeroBound,oneBound]

theorem owner_channel (block : Nat) (column : Fin 16) :
    owner (block*16+column.val)=owner column.val := by
  apply Fin.ext
  simp [owner,Nat.add_mod,Nat.mul_mod]

#axiomsClean Tree.eval_legacy
#axiomsClean selected_tree
#axiomsClean selected_owner
#axiomsClean chain_broadcast
#axiomsClean owner_channel
end
end VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly
