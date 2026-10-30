import bench.tritonbench_g.layer_norm_triton.IO
import bench.tritonbench_g.fast_layernorm.Proofs.Real
import bench.tritonbench_g.fast_layernorm.Proofs.Layout

/-! Ordered default-FP32 graph. Zero leaves, per-output-lane reduction roots,
FMA edges and first/subsequent group contributions remain explicit. Forward
inverse standard deviation is division by sqrt, not an rsqrt substitution. -/
namespace VeriTile.Bench.Repairs.LayerNormTriton
namespace Reuse
abbrev Storage := VeriTile.Bench.Repairs.FastLayernorm.Storage.Tensor
abbrev Dense := VeriTile.Bench.Repairs.FastLayernorm.Storage.Dense
abbrev Statistics := VeriTile.Bench.Repairs.FastLayernorm.Statistics
end Reuse
open VeriTile.Bench.Repairs.FastLayernorm (MulAdd)
noncomputable section

structure Ops (α : Type) where
  zero : α
  one : α
  count : Nat → α
  (add sub mul div : α → α → α)
  fma : α → α → α → α
  sqrt : α → α

def ordered (op : α → α → α) (swap : Bool) (a b : α) : α :=
  if swap then op b a else op a b

def mulAdd (ops : Ops α) : MulAdd → α → α → α → α
  | .separate sm sa,a,b,c => ordered ops.add sa (ordered ops.mul sm a b) c
  | .fused sm,a,b,c => if sm then ops.fma b a c else ops.fma a b c

inductive SumTree (B : Nat) where
  | zero
  | leaf (j : Fin B)
  | add (left right : SumTree B)

def SumTree.leaves : SumTree B → List (Fin B)
  | .zero => []
  | .leaf j => [j]
  | .add a b => a.leaves ++ b.leaves

def SumTree.eval (ops : Ops α) (values : Fin B → α) : SumTree B → α
  | .zero => ops.zero
  | .leaf j => values j
  | .add a b => ops.add (a.eval ops values) (b.eval ops values)

inductive ProductTree (B : Nat) where
  | zero
  | product (j : Fin B) (swap : Bool)
  | add (left right : ProductTree B)
  | fma (j : Fin B) (swap : Bool) (rest : ProductTree B)

def ProductTree.leaves : ProductTree B → List (Fin B)
  | .zero => []
  | .product j _ => [j]
  | .add a b => a.leaves ++ b.leaves
  | .fma j _ rest => j :: rest.leaves

def ProductTree.eval (ops : Ops α) (x y : Fin B → α) : ProductTree B → α
  | .zero => ops.zero
  | .product j swap => ordered ops.mul swap (x j) (y j)
  | .add a b => ops.add (a.eval ops x y) (b.eval ops x y)
  | .fma j swap rest => if swap then ops.fma (y j) (x j) (rest.eval ops x y)
      else ops.fma (x j) (y j) (rest.eval ops x y)

structure Plan (N B : Nat) where
  meanTree : Nat → SumTree B
  varianceTree : Nat → ProductTree B
  firstTree : Nat → ProductTree B
  secondTree : Nat → SumTree B
  meanComplete : ∀ lane,(meanTree lane).leaves.Perm (List.finRange B)
  varianceComplete : ∀ lane,(varianceTree lane).leaves.Perm (List.finRange B)
  firstComplete : ∀ lane,(firstTree lane).leaves.Perm (List.finRange B)
  secondComplete : ∀ lane,(secondTree lane).leaves.Perm (List.finRange B)
  (outputKind correctionKind groupKind : MulAdd)
  (normalSwap weightDySwap finalSwap epsilonSwap dbSwap groupFirstSwap : Bool)
  eraseUnitDivision : Bool
  unitOnly : eraseUnitDivision=Bool.true → N=1
  covers : N≤B

def divideCount (ops : Ops α) (plan : Plan N B) (value : α) : α :=
  if plan.eraseUnitDivision then value else ops.div value (ops.count N)

def masked (ops : Ops α) (N : Nat) (values : Nat → α) (j : Fin B) : α :=
  if j.val<N then values j.val else ops.zero

def mean (ops : Ops α) (plan : Plan N B) (x : Nat → α) (lane : Nat) : α :=
  divideCount ops plan ((plan.meanTree lane).eval ops (masked ops N x))

def centered (ops : Ops α) (plan : Plan N B) (x : Nat → α) (j : Fin B) : α :=
  if j.val<N then ops.sub (x j.val) (mean ops plan x j.val) else ops.zero

def variance (ops : Ops α) (plan : Plan N B) (x : Nat → α) (lane : Nat) : α :=
  divideCount ops plan ((plan.varianceTree lane).eval ops (centered ops plan x) (centered ops plan x))

def statistics (ops : Ops α) (plan : Plan N B) (x : Nat → α) (eps : α) (lane : Nat) : Reuse.Statistics α :=
  ⟨mean ops plan x lane,
   ops.div ops.one (ops.sqrt (ordered ops.add plan.epsilonSwap (variance ops plan x lane) eps))⟩

def normed (ops : Ops α) (plan : Plan N B) (stats : Reuse.Statistics α) (x : α) : α :=
  ordered ops.mul plan.normalSwap (ops.sub x stats.mean) stats.inverse

def forwardValue (ops : Ops α) (plan : Plan N B) (x w bias : Nat → α) (eps : α) (column : Nat) : α :=
  mulAdd ops plan.outputKind (normed ops plan (statistics ops plan x eps column) (x column))
    (w column) (bias column)

def weightedDy (ops : Ops α) (plan : Plan N B) (dy w : Nat → α) (column : Nat) : α :=
  ordered ops.mul plan.weightDySwap (w column) (dy column)

def backwardValue (ops : Ops α) (plan : Plan N B) (stats : Reuse.Statistics α)
    (x dy w : Nat → α) (column : Nat) : α :=
  let zs := masked ops N (fun j => normed ops plan stats (x j))
  let qs := masked ops N (weightedDy ops plan dy w)
  let c1 := divideCount ops plan ((plan.firstTree column).eval ops zs qs)
  let c2 := divideCount ops plan ((plan.secondTree column).eval ops qs)
  let correction := mulAdd ops plan.correctionKind (normed ops plan stats (x column)) c1 c2
  ordered ops.mul plan.finalSwap (ops.sub (weightedDy ops plan dy w column) correction) stats.inverse

def groupStep (ops : Ops α) (plan : Plan N B) (x dy : Nat → Nat → α)
    (stats : Nat → Reuse.Statistics α) (column row : Nat) : Option (α × α) → Option (α × α)
  | none => some (ordered ops.mul plan.groupFirstSwap (dy row column) (normed ops plan (stats row) (x row column)),dy row column)
  | some previous => some
      (mulAdd ops plan.groupKind (dy row column) (normed ops plan (stats row) (x row column)) previous.1,
       ordered ops.add plan.dbSwap (dy row column) previous.2)

def groupValue (ops : Ops α) (plan : Plan N B) (M G group : Nat)
    (x dy : Nat → Nat → α) (stats : Nat → Reuse.Statistics α) (column : Nat) : α × α :=
  ((groupRows M G group).foldl (fun acc row => groupStep ops plan x dy stats column row acc) none).getD
    (ops.zero,ops.zero)

structure FinalPlan where
  tree : Nat → SumTree 32
  complete : ∀ lane,(tree lane).leaves.Perm (List.finRange 32)
  streamSwap : Bool

def finalLane (ops : Ops α) (plan : FinalPlan) (groups : Nat) (partials : Nat → α) (lane : Fin 32) : α :=
  (List.range ((groups+31)/32)).foldl
    (fun acc iteration => ordered ops.add plan.streamSwap acc
      (if iteration*32+lane.val<groups then partials (iteration*32+lane.val) else ops.zero)) ops.zero

def finalValue (ops : Ops α) (plan : FinalPlan) (groups : Nat) (partials : Nat → α) (column : Nat) : α :=
  (plan.tree column).eval ops (finalLane ops plan groups partials)

/-- Local scalar interpretation obligations only. No index, memory, protocol,
successful execution or whole-kernel output appears in this record. -/
structure LocalExact (a b : Ops α) : Prop where
  zero : a.zero=b.zero
  one : a.one=b.one
  count : ∀ n,a.count n=b.count n
  add : ∀ x y,a.add x y=b.add x y
  sub : ∀ x y,a.sub x y=b.sub x y
  mul : ∀ x y,a.mul x y=b.mul x y
  div : ∀ x y,a.div x y=b.div x y
  fma : ∀ x y z,a.fma x y z=b.fma x y z
  sqrt : ∀ x,a.sqrt x=b.sqrt x

theorem LocalExact.operations {a b : Ops α} (h : LocalExact a b) : a=b := by
  cases a
  cases b
  cases h
  congr <;> funext <;> first | assumption | aesop

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormTriton
