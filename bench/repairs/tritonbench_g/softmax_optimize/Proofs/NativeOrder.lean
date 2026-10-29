import bench.repairs.tritonbench_g.softmax_flaggems.Proofs.Native

namespace VeriTile.Bench.Repairs.SoftmaxOptimize.NativeOrder
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.SoftmaxFlaggems.Native
noncomputable section

/-- Compiler-selected addition operand order. Decisions inspect source shapes,
never leaf words. This is syntax selection, not an FP commutativity law. -/
abbrev AddOrder := Shape → Bool

def arrange (order : AddOrder) : Term → Term
  | .binary p d .add a b =>
      let left := arrange order a
      let right := arrange order b
      if order (.binary p d .add (shape a) (shape b)) then .binary p d .add right left
      else .binary p d .add left right
  | .binary p d op a b => .binary p d op (arrange order a) (arrange order b)
  | .unary p op a => .unary p op (arrange order a)
  | .cast p src dst a => .cast p src dst (arrange order a)
  | .load a => .load (arrange order a)
  | term => term

/-- Reuse the existing native primitive/FMA evaluator after selecting operand
order. The shared fusion schedule sees the selected syntax. -/
def word (I : Instructions) (S : Schedule) (order : AddOrder) (term : Term) : Word :=
  eval I S (arrange order term)

def observe (I : Instructions) (S : Schedule) (order : AddOrder) (s : State Term)
    (r : RegionName) (i : Nat) : Word := word I S order ((s.mem r i).read .real)

@[simp] theorem leaf (I : Instructions) (S : Schedule) (O : AddOrder)
    (r : RegionName) (address : Nat) (w : Word) : word I S O (.leaf r address w)=w := rfl

#specNonCircular word avoiding [FP.Structural.exec]
end
end VeriTile.Bench.Repairs.SoftmaxOptimize.NativeOrder
