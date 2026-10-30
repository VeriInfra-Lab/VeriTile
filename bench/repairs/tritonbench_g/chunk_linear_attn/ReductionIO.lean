import bench.repairs.tritonbench_g.chunk_linear_attn.HostSum
import VeriTile.Triton.Float.ReductionProgramIO

namespace VeriTile.Bench.Repairs.ChunkLinearAttn.LibraryReduction
open VeriTile Triton FP.Structural
noncomputable section

def tree {n : Nat} : LibrarySum.Tree n → FP32AddTree n
  | .zero => .zero
  | .input i => .input i
  | .add a b => .add (tree a) (tree b)

def plan {n : Nat} (p : LibrarySum.Plan n) : FP32AddPlan n :=
  ⟨tree p.tree, by
    have leaves : ∀ {n : Nat} (t : LibrarySum.Tree n), (tree t).leaves = t.leaves := by
      intro n t
      induction t with
      | zero => rfl
      | input i => rfl
      | add a b ha hb => simp only [tree, FP32AddTree.leaves, LibrarySum.Tree.leaves, ha, hb]
    rw [leaves]
    exact p.coverage⟩

/-- The original finite library sum, with the same input snapshot, finite
tree, FP32 load/add/cast boundaries and erased Real cell storage tag. -/
def sumStage {α : Type} (M : Algebra α) (p : Parameters)
    (selection : LibrarySelection p) : FP32SumStage α where
  backend := M
  inputCount := keyBlocks p
  outputCount := valueCount p
  read := fun o k => ("valuePartials", k.val * valueCount p + o.val)
  write := fun o => ("valueGradient", o.val)
  plans := fun o => plan (selection.plans o)

end
end VeriTile.Bench.Repairs.ChunkLinearAttn.LibraryReduction
