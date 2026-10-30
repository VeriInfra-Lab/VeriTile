import bench.tritonbench_g.lora_expand_gemv.Proofs.Single
namespace VeriTile.Bench.Repairs.LoraExpandGemv.Native
open VeriTile.Triton FP.Structural FPValue
open FP.Equational (ReductionPlan)
noncomputable section
abbrev Bits := BitVec 32
inductive Shape where
  | zero | leaf
  | cast (src dst : OutputFormat) (x : Shape)
  | mul (fmt : OutputFormat) (x y : Shape)
  | add (fmt : OutputFormat) (x y : Shape)
  deriving DecidableEq

def shape : Word → Shape
  | .zero => .zero | .leaf _ _ => .leaf
  | .cast a b x => .cast a b (shape x)
  | .mul f x y => .mul f (shape x) (shape y)
  | .add f x y => .add f (shape x) (shape y)

inductive Fusion where | split | left | right deriving DecidableEq
structure Schedule where
  fusion : Shape → Fusion
  swap : Shape → Bool

/-- Six actual scalar families: no reduction, tensor, or result callbacks.
Half values occupy the low 16 bits. Binary32 output occupies all 32 bits. -/
structure ScalarOps where
  narrow : Bits → Bits
  widen : Bits → Bits
  mul16 : Bits → Bits → Bits
  add16 : Bits → Bits → Bits
  fma16 : Bits → Bits → Bits → Bits
  add32 : Bits → Bits → Bits

def binary (S : Schedule) (sh : Shape) (op : Bits → Bits → Bits) (x y : Bits) :=
  if S.swap sh then op y x else op x y

def fused (A : ScalarOps) (S : Schedule) (a b : Word) (x y z : Bits) :=
  if S.swap (.mul .half (shape a) (shape b)) then A.fma16 y x z else A.fma16 x y z

/-- The default compiler may contract either adjacent half multiplication.
The output widening is retained before FP32 ADD, so it is a contraction boundary.
Selectors inspect erased graph shape, never floating words or tensor contents. -/
def eval (A : ScalarOps) (S : Schedule) (bits : RegionName → Nat → Bits) : Word → Bits
  | .zero => 0
  | .leaf r o => bits r o
  | .cast .single .half x => A.narrow (eval A S bits x)
  | .cast .half .single x => A.widen (eval A S bits x)
  | .cast _ _ _ => 0
  | .mul .half x y => binary S (.mul .half (shape x) (shape y)) A.mul16 (eval A S bits x) (eval A S bits y)
  | .mul _ _ _ => 0
  | .add .half a b =>
      match S.fusion (.add .half (shape a) (shape b)),a,b with
      | .left,.mul .half x y,c => fused A S x y (eval A S bits x) (eval A S bits y) (eval A S bits c)
      | .right,c,.mul .half x y => fused A S x y (eval A S bits x) (eval A S bits y) (eval A S bits c)
      | _,a,b => binary S (.add .half (shape a) (shape b)) A.add16 (eval A S bits a) (eval A S bits b)
  | .add .single a b => binary S (.add .single (shape a) (shape b)) A.add32 (eval A S bits a) (eval A S bits b)
  | .add _ _ _ => 0

def supported : Word → Prop
  | .zero | .leaf _ _ => True
  | .cast .single .half x | .cast .half .single x => supported x
  | .mul .half x y | .add .half x y | .add .single x y => supported x ∧ supported y
  | _ => False

theorem narrow_supported (x : Word) (h : supported x) : supported (castWord .ieeeValue .single .half x) := by
  cases x <;> simpa [castWord,supported] using h

theorem widen_supported (x : Word) (h : supported x) : supported (castWord .ieeeValue .half .single x) := by
  cases x <;> simpa [castWord,supported] using h

theorem tree_supported (xs : Fin n → Word) (h : ∀ i,supported (xs i))
    (tree : FP.Equational.ReductionTree n) : supported (treeValue .ieeeValue .half xs tree) := by
  induction tree with
  | input i => exact h i
  | zero => trivial
  | add a b ia ib => exact ⟨ia,ib⟩

theorem value_supported (plans : ∀ n, ReductionPlan n) (g : Geometry) (raw : Nat → Int) (b n : Nat) :
    supported (value plans g raw b n) := by
  have h : ∀ k : Fin g.block,supported (product g raw b n k) := by
    intro k
    by_cases hk : k.val<g.K <;> simp [product,hk,castWord,supported]
  have ht := widen_supported _ (tree_supported _ h (plans g.block).tree)
  unfold value
  cases g.add <;> simpa [supported] using ht

structure ScalarLaws (actual reference : ScalarOps) : Prop where
  narrow : ∀ x,actual.narrow x=reference.narrow x
  widen : ∀ x,actual.widen x=reference.widen x
  mul16 : ∀ x y,actual.mul16 x y=reference.mul16 x y
  add16 : ∀ x y,actual.add16 x y=reference.add16 x y
  fma16 : ∀ x y z,actual.fma16 x y z=reference.fma16 x y z
  add32 : ∀ x y,actual.add32 x y=reference.add32 x y

theorem operations_eq (a b : ScalarOps) (h : ScalarLaws a b) : a=b := by
  cases a; cases b
  simp only [ScalarOps.mk.injEq]
  exact ⟨funext h.narrow,funext h.widen,funext (fun x => funext (h.mul16 x)),
    funext (fun x => funext (h.add16 x)),funext (fun x => funext (fun y => funext (h.fma16 x y))),
    funext (fun x => funext (h.add32 x))⟩

#axiomsClean narrow_supported
#axiomsClean widen_supported
#axiomsClean tree_supported
#axiomsClean value_supported
#axiomsClean operations_eq
#specNonCircular ScalarLaws avoiding [Branches.source,FP.Structural.exec,value]
end
end VeriTile.Bench.Repairs.LoraExpandGemv.Native
