import VeriTile.Triton.DSL.Expansion.Common

/-! Transparent source expansion for an explicitly selected ordered reduction
tree. These definitions only build the existing typed scalar operations; they
introduce neither a numerical oracle nor an associativity assumption. -/
namespace VeriTile.Triton.DSL.WelfordSource
open VeriTile.Triton

inductive Tree (N : Nat) where
  | leaf : Fin N → Tree N
  | node : Tree N → Tree N → Tree N

def Tree.leaves {N : Nat} : Tree N → List (Fin N)
  | .leaf i => [i]
  | .node l r => l.leaves ++ r.leaves

def Tree.Complete {N : Nat} (t : Tree N) : Prop :=
  t.leaves.Perm (List.finRange N)

structure Triple (α : Type) where
  mean : α
  m2 : α
  weight : α

def combine (left right : Triple (Op .real [])) : Triple (Op .real []) :=
  let zero : Op .real [] := .const 0.0
  let delta := Op.where (.eq .real .nil left.mean right.mean) zero
    (.sub .real .nil right.mean left.mean)
  let weight := Op.add .real .nil left.weight right.weight
  let ratio := Op.where (.eq .real .nil weight zero) zero
    (.div .real .nil right.weight weight)
  { mean := .add .real .nil left.mean (.mul .real .nil delta ratio)
    m2 := .add .real .nil (.add .real .nil left.m2 right.m2)
      (.mul .real .nil (.mul .real .nil (.mul .real .nil delta delta) left.weight) ratio)
    weight := weight }

/-- The supported helper call reduces axis 1 of a single-row tile. Each leaf
is selected explicitly, including padded lanes, in the supplied tree order. -/
def expressions {N : Nat} (mean m2 weight : Op .real [1,N]) :
    Tree N → Triple (Op .real [])
  | .leaf i =>
    { mean := .remap (shape := [1,N]) [] (fun _ => ⟨⟨0,by decide⟩,i,()⟩) mean
      m2 := .remap (shape := [1,N]) [] (fun _ => ⟨⟨0,by decide⟩,i,()⟩) m2
      weight := .remap (shape := [1,N]) [] (fun _ => ⟨⟨0,by decide⟩,i,()⟩) weight }
  | .node l r => combine (expressions mean m2 weight l) (expressions mean m2 weight r)

def sumExpression {N : Nat} (input : Op .real [1,N]) : Tree N → Op .real []
  | .leaf i => .remap (shape := [1,N]) [] (fun _ => ⟨⟨0,by decide⟩,i,()⟩) input
  | .node l r => .add .real .nil (sumExpression input l) (sumExpression input r)

end VeriTile.Triton.DSL.WelfordSource
