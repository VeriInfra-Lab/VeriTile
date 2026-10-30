import bench.repairs.tritonbench_g.attention_llama.Proofs.SourceGraph

/-! Fixed SM80 schedules for the recorded 64x64 score tile. No numerical
reduction law is an input: every ordered binary node is evaluated explicitly.
Four-warp profile: lane low bits distribute columns 2*l,2*l+1 modulo 8;
each thread left-folds sixteen columns, then XOR2 and XOR1 (local first).
Eight-warp D128: each warp owns eight columns, folds its two registers and
XOR2/XOR1; lane low2=0 writes shared memory. The cross-warp shuffle uses
XOR4/XOR2/XOR1, and lane low3=0 stores the canonical broadcast result.
The artifact audit binds these schedules to both dtypes and causal branches. -/
namespace VeriTile.Bench.Repairs.AttentionLlama.NativeSchedule

inductive Tree where
  | leaf (column : Fin 64)
  | node (left right : Tree)

def column (n : Nat) : Fin 64 := ⟨n % 64, Nat.mod_lt _ (by decide)⟩
def lane (n : Nat) : Fin 4 := ⟨n / 2 % 4, Nat.mod_lt _ (by decide)⟩

def local16 (l : Nat) : Nat → Tree
  | 0 => .leaf (column (2*l))
  | n+1 => .node (local16 l n) (.leaf (column (8*((n+1)/2)+2*l+(n+1)%2)))

def butterfly (x : Nat → Tree) : List Nat → Nat → Tree
  | [], l => x l
  | mask :: rest, l => butterfly (fun k => .node (x k) (x (Nat.xor k mask))) rest l

def fourWarp (l : Fin 4) : Tree := butterfly (fun k => local16 k 15) [2,1] l.val

def warp8 (w : Nat) : Tree :=
  butterfly (fun l => .node (.leaf (column (8*w+2*l)))
    (.leaf (column (8*w+2*l+1)))) [2,1] 0

def eightWarp : Tree := butterfly warp8 [4,2,1] 0

def selected (D : Nat) (l : Fin 4) : Tree := if D = 128 then eightWarp else fourWarp l

def Tree.eval {α : Type} (op : α → α → α) (x : Fin 64 → α) : Tree → α
  | .leaf i => x i
  | .node a b => op (a.eval op x) (b.eval op x)

theorem Tree.eval_map {α β : Type} (op : α → α → α) (op' : β → β → β)
    (f : α → β) (h : ∀ x y, f (op x y) = op' (f x) (f y))
    (x : Fin 64 → α) (t : Tree) :
    f (t.eval op x) = t.eval op' (fun i => f (x i)) := by
  induction t with
  | leaf i => rfl
  | node a b ia ib => simp only [Tree.eval,h,ia,ib]

/-- Hardware operand-index to logical K coordinate. E5M2 selects kWidth=4
even after conversion to FP16. This interleaves half-pairs across each K32
span; D16 has one K16 with a different within-instruction permutation. -/
def mmaCoordinate (fp8 : Bool) (K group : Nat) (i : Fin 16) : Nat :=
  if fp8 then
    if K = 16 then 4*((i.val%8)/2) + i.val%2 + 2*(i.val/8)
    else 32*(group/2) + 2*(group%2) + 4*((i.val%8)/2) + i.val%2 + 16*(i.val/8)
  else 16*group+i.val

/-- Ordered K16 composition with explicit actual fragment coordinates.
An ISA port has exactly 16 FP16 operands on each side and one FP32 seed.
Hardware internal rounding is not fabricated. -/
def mmaFold {α : Type} (mma16 : (Fin 16 → α) → (Fin 16 → α) → α → α)
    (coordinate : Nat → Fin 16 → Nat) (x y : Nat → α) (seed : α) : Nat → α
  | 0 => seed
  | n+1 => mma16 (fun i => x (coordinate n i)) (fun i => y (coordinate n i))
      (mmaFold mma16 coordinate x y seed n)

theorem mmaFold_map {α β : Type}
    (a : (Fin 16 → α) → (Fin 16 → α) → α → α)
    (b : (Fin 16 → β) → (Fin 16 → β) → β → β) (f : α → β)
    (h : ∀ x y z, f (a x y z) = b (fun i => f (x i)) (fun i => f (y i)) (f z))
    (coordinate : Nat → Fin 16 → Nat) (x y : Nat → α) (seed : α) (n : Nat) :
    f (mmaFold a coordinate x y seed n) =
      mmaFold b coordinate (fun i => f (x i)) (fun i => f (y i)) (f seed) n := by
  induction n with
  | zero => rfl
  | succ n ih => simp only [mmaFold,h,ih]

end VeriTile.Bench.Repairs.AttentionLlama.NativeSchedule
