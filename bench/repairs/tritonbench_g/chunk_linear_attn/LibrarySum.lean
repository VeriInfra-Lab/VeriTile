import Mathlib.Algebra.BigOperators.Fin
import bench.repairs.tritonbench_g.chunk_linear_attn.RealVJP
import bench.repairs.tritonbench_g.chunk_linear_attn.Model
namespace VeriTile.Bench.Repairs.ChunkLinearAttn.LibrarySum
open scoped BigOperators
open VeriTile Triton FP.Structural
noncomputable section
set_option maxHeartbeats 6000000

/-- Explicit finite operation plan supplied by the ordinary library lowering.
Zeros retain accumulator initialization; each join is exactly one scalar add.
This data structure contains no tensor-valued numerical callback. -/
inductive Tree (n : Nat) where
  | zero : Tree n
  | input : Fin n → Tree n
  | add : Tree n → Tree n → Tree n

def Tree.leaves {n : Nat} : Tree n → List (Fin n)
  | .zero => []
  | .input i => [i]
  | .add a b => a.leaves ++ b.leaves

def Tree.eval {n : Nat} {α : Type} (zero : α) (add : α → α → α) (x : Fin n → α) : Tree n → α
  | .zero => zero
  | .input i => x i
  | .add a b => add (a.eval zero add x) (b.eval zero add x)

structure Plan (n : Nat) where
  tree : Tree n
  coverage : tree.leaves.Perm (List.ofFn (fun i : Fin n => i))

/-- The actual tree is allowed to depend on layout, device and library profile.
Coverage is a structural leaf obligation, never a hypothesis about its answer. -/
def evaluate {n : Nat} {α : Type} (M : Algebra α) (plan : Plan n) (x : Fin n → α) : α :=
  plan.tree.eval (M.literal (some .fp32) .real 0)
    (M.binary (some .fp32) .real .add) (fun i => M.fp32Load (x i))

theorem tree_real {n : Nat} (tree : Tree n) (x : Fin n → ℝ) :
    tree.eval 0 (·+·) x=(tree.leaves.map x).sum := by
  induction tree with
  | zero => rfl
  | input i => simp [Tree.eval,Tree.leaves]
  | add a b ha hb => simp [Tree.eval,Tree.leaves,ha,hb]

theorem real_complete {n : Nat} (plan : Plan n) (x : Fin n → ℝ) :
    plan.tree.eval 0 (·+·) x=∑ i : Fin n, x i := by
  rw [tree_real,(plan.coverage.map x).sum_eq]
  simp [List.sum_ofFn,Function.comp_def]

theorem transport {n : Nat} {α β : Type} (f : α → β)
    (za : α) (zb : β) (aa : α → α → α) (ab : β → β → β)
    (xa : Fin n → α) (xb : Fin n → β)
    (hz : f za=zb) (ha : ∀ x y, f (aa x y)=ab (f x) (f y))
    (hx : ∀ i, f (xa i)=xb i) (tree : Tree n) :
    f (tree.eval za aa xa)=tree.eval zb ab xb := by
  induction tree with
  | zero => exact hz
  | input i => exact hx i
  | add a b h1 h2 => simpa only [Tree.eval,ha,h1,h2]

def linearTree (n : Nat) : Tree n :=
  (List.ofFn (fun i : Fin n => i)).foldl (fun a i => .add a (.input i)) .zero

theorem linear_leaves (n : Nat) : (linearTree n).leaves=List.ofFn (fun i : Fin n => i) := by
  have aux : ∀ (xs : List (Fin n)) (a : Tree n),
      (xs.foldl (fun a i => Tree.add a (.input i)) a).leaves=a.leaves++xs := by
    intro xs
    induction xs with
    | nil => intro a; simp
    | cons i xs ih => intro a; simpa [List.foldl_cons,Tree.leaves,List.append_assoc] using ih (.add a (.input i))
  simpa [linearTree,Tree.leaves] using aux (List.ofFn (fun i : Fin n => i)) .zero

def linearPlan (n : Nat) : Plan n := ⟨linearTree n,by rw [linear_leaves]⟩

#axiomsClean real_complete
#axiomsClean transport
#axiomsClean linear_leaves
#specNonCircular Plan avoiding [evaluate,Tree.eval]
end
end VeriTile.Bench.Repairs.ChunkLinearAttn.LibrarySum
