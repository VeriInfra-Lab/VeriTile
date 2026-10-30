import bench.tritonbench_g.layer_norm_fwd.Proofs.Mathematical_WholeReduction
import bench.tritonbench_g.layernorm_fwd_triton.Proofs.Memory
import bench.tritonbench_g.layer_norm_fwd.Proofs.WrapperAction

/-! Selected default-FP32 arithmetic. Reduction roots are indexed by the
consumer lane: native butterfly order need not be identified on opaque words.
Saved statistics use lane zero. Contraction and operand order are data. -/
namespace VeriTile.Bench.Repairs.FastLayernorm
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.LayerNormFwd.Whole
noncomputable section

structure Ops (α : Type) where
  zero : α
  count : Nat → α
  (add sub mul div : α → α → α)
  neg : α → α
  fma : α → α → α → α
  rsqrt : α → α

def ordered (op : α → α → α) (swap : Bool) (a b : α) : α := if swap then op b a else op a b

inductive MulAdd where
  | separate (swapMul swapAdd : Bool)
  | fused (swapMul : Bool)
  deriving DecidableEq

def mulAdd (ops : Ops α) : MulAdd → α → α → α → α
  | .separate sm sa,a,b,c => ordered ops.add sa (ordered ops.mul sm a b) c
  | .fused sm,a,b,c => if sm then ops.fma b a c else ops.fma a b c

inductive ProductTree (B : Nat) where
  | product (j : Fin B) (swap : Bool)
  | add (left right : ProductTree B)
  | fused (j : Fin B) (swap : Bool) (rest : ProductTree B)

def ProductTree.leaves : ProductTree B → List (Fin B)
  | .product j _ => [j]
  | .add a b => a.leaves++b.leaves
  | .fused j _ rest => j::rest.leaves

def ProductTree.eval (ops : Ops α) (x y : Fin B → α) : ProductTree B → α
  | .product j swap => ordered ops.mul swap (x j) (y j)
  | .add a b => ops.add (a.eval ops x y) (b.eval ops x y)
  | .fused j swap rest => if swap then ops.fma (y j) (x j) (rest.eval ops x y)
      else ops.fma (x j) (y j) (rest.eval ops x y)

inductive MulSub where
  | separate (swap : Bool)
  | fused (swap : Bool)
  deriving DecidableEq

def mulSub (ops : Ops α) : MulSub → α → α → α → α
  | .separate swap,a,b,c => ops.sub (ordered ops.mul swap a b) c
  | .fused swap,a,b,c => if swap then ops.fma b a (ops.neg c) else ops.fma a b (ops.neg c)

structure Plan (N B : Nat) where
  meanTree : Nat → SumTree B
  squareTree : Nat → SquareTree B
  backwardFirst : Nat → ProductTree B
  backwardSecond : Nat → SumTree B
  meanComplete : ∀ lane,(meanTree lane).leaves.Perm (List.finRange B)
  squareComplete : ∀ lane,(squareTree lane).leaves.Perm (List.finRange B)
  firstComplete : ∀ lane,(backwardFirst lane).leaves.Perm (List.finRange B)
  secondComplete : ∀ lane,(backwardSecond lane).leaves.Perm (List.finRange B)
  (outputFMA affineFMA : MulAdd)
  eraseInactiveAffine : Bool
  firstSubtract : MulSub
  (normedSwap productSwap correctionSwap finalSwap epsilonSwap : Bool)
  unitCorrectionFMA : Bool
  unitCorrectionValid : unitCorrectionFMA=Bool.true → N=1
  unitVarianceFMA : Bool
  unitVarianceValid : unitVarianceFMA=Bool.true → N=1
  eraseUnitDivision : Bool
  eraseUnitValid : eraseUnitDivision=Bool.true → N=1
  covers : N≤B

/-- FP32-to-FP32 source casts are representation identities. -/
def cast32 (value : α) : α := value

def divideCount (ops : Ops α) (plan : Plan N B) (value : α) : α :=
  if plan.eraseUnitDivision then value else ops.div value (ops.count N)

def masked (ops : Ops α) (N : Nat) (x : Nat → α) (j : Fin B) : α :=
  if j.val<N then cast32 (x j.val) else ops.zero

def mean (ops : Ops α) (plan : Plan N B) (x : Nat → α) (lane : Nat) : α :=
  divideCount ops plan ((plan.meanTree lane).eval ops.add (masked ops N x))

def centered (ops : Ops α) (plan : Plan N B) (x : Nat → α) (j : Fin B) : α :=
  if j.val<N then ops.sub (cast32 (x j.val)) (mean ops plan x j.val) else ops.zero

def variance (ops : Ops α) (plan : Plan N B) (x : Nat → α) (lane : Nat) : α :=
  divideCount ops plan ((plan.squareTree lane).eval ops.mul ops.add ops.fma (centered ops plan x))

structure Statistics (α : Type) where
  mean : α
  inverse : α

def statistics (ops : Ops α) (plan : Plan N B) (x : Nat → α) (eps : α) (lane : Nat) : Statistics α :=
  let v := if plan.unitVarianceFMA then
    let z := ops.sub (cast32 (x 0)) (mean ops plan x 0)
    ops.fma z z eps
    else ordered ops.add plan.epsilonSwap (variance ops plan x lane) eps
  ⟨mean ops plan x lane,ops.rsqrt v⟩

def normed (ops : Ops α) (plan : Plan N B) (stats : Statistics α) (x : α) : α :=
  ordered ops.mul plan.normedSwap (ops.sub x stats.mean) stats.inverse

def forwardValue (ops : Ops α) (plan : Plan N B) (stats : Statistics α)
    (x w bias : α) : α := cast32 (mulAdd ops plan.outputFMA (normed ops plan stats (cast32 x)) (cast32 w) (cast32 bias))

def forwardReference (ops : Ops α) (plan : Plan N B) (x w bias : Nat → α) (eps : α) (col : Nat) : α :=
  forwardValue ops plan (statistics ops plan x eps col) (x col) (w col) (bias col)

def dyw (ops : Ops α) (N : Nat) (dy w : Nat → α) (j : Fin B) : α :=
  ops.mul (masked ops N dy j) (masked ops N w j)

/-- Active selection excludes exceptional inactive 0*NaN/Inf. -/
def backwardProduct (ops : Ops α) (plan : Plan N B) (stats : Statistics α)
    (x dy w : Nat → α) (j : Fin B) : α :=
  if j.val<N then ordered ops.mul plan.productSwap (dyw ops N dy w j)
    (normed ops plan stats (masked ops N x j)) else ops.zero

def correctionSubtract (ops : Ops α) (plan : Plan N B) (first z second : α) : α :=
  if plan.unitCorrectionFMA then
    if plan.correctionSwap then ops.fma second (ops.neg z) first else ops.fma (ops.neg z) second first
  else ops.sub first (divideCount ops plan (ordered ops.mul plan.correctionSwap z second))

def backwardReference (ops : Ops α) (plan : Plan N B) (stats : Statistics α)
    (x dy w : Nat → α) (col : Nat) : α :=
  let first := divideCount ops plan ((plan.backwardFirst col).eval ops (masked ops N dy) (masked ops N w))
  let second := (plan.backwardSecond col).eval ops.add (backwardProduct ops plan stats x dy w)
  let left := mulSub ops plan.firstSubtract (cast32 (dy col)) (cast32 (w col)) first
  cast32 (ordered ops.mul plan.finalSwap
    (correctionSubtract ops plan left (normed ops plan stats (cast32 (x col))) second) stats.inverse)

/-- Increasing int64 row loop. Affine tails receive exact zero for every row. -/
def affineStep (ops : Ops α) (plan : Plan N B) (x dy : Nat → Nat → α)
    (stats : Nat → Statistics α) (col row : Nat) (acc : α × α) : α × α :=
  let active := col<N
  let d := if active then cast32 (dy row col) else ops.zero
  let z := normed ops plan (stats row) (if active then cast32 (x row col) else ops.zero)
  let dw := if active then mulAdd ops plan.affineFMA d z acc.1
    else if plan.eraseInactiveAffine then ops.zero else ops.add acc.1 ops.zero
  (dw,ops.add acc.2 d)

def affineReference (ops : Ops α) (plan : Plan N B) (rows : Nat)
    (x dy : Nat → Nat → α) (stats : Nat → Statistics α) (col : Nat) : α × α :=
  (List.range rows).foldl (fun acc row => affineStep ops plan x dy stats col row acc) (ops.zero,ops.zero)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FastLayernorm
