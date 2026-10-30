import bench.repairs.tritonbench_g.cross_entropy2.Kernels

/-! Independent ordered scalar expressions, with actual runtime scalar values.
No reference executes the kernel or reads interpreter memory/registers. -/
namespace VeriTile.Bench.Repairs.CrossEntropy2.Reference
open VeriTile.Triton FP.Structural
noncomputable section
variable {α : Type}

def logits (M : Algebra α) (p : Parameters) (args : RuntimeArgs.Values α)
    (x : Nat → α) (block : Nat) : Values α .real [p.block] := fun j =>
  if block*p.block+j.1.val<p.cols then
    M.binary (some .fp32) .real .mul
      (M.cast (some .fp32) .real .real (x (block*p.block+j.1.val))) args.logitScale
  else M.negInf

def maximum (M : Algebra α) (p : Parameters) (args : RuntimeArgs.Values α)
    (x : Nat → α) (block : Nat) : α :=
  M.reduceMax (some .fp32) ⟨0,by simp⟩ Bool.false (logits M p args x block) PUnit.unit

def lse (M : Algebra α) (p : Parameters) (args : RuntimeArgs.Values α)
    (x : Nat → α) (block : Nat) : α :=
  M.binary (some .fp32) .real .add
    (M.unary (some .fp32) .log
      (M.reduceSum (some .fp32) ⟨0,by simp⟩ Bool.false
        (fun j => M.unary (some .fp32) .exp
          (M.binary (some .fp32) .real .sub (logits M p args x block j)
            (maximum M p args x block))) PUnit.unit))
    (maximum M p args x block)

def sum (M : Algebra α) (p : Parameters) (args : RuntimeArgs.Values α)
    (x : Nat → α) (block : Nat) : α :=
  M.reduceSum (some .fp32) (shape:=[p.block]) ⟨0,by simp⟩ Bool.false
    (fun j => if block*p.block+j.1.val<p.cols then logits M p args x block j
      else M.literal (some .fp32) .real 0.0) PUnit.unit

def owns (p : Parameters) (block : Nat) (label : Int) : Prop :=
  (block : Int)*p.block≤label-p.classStart ∧
    label-p.classStart<min (p.cols : Int) (((block : Int)+1)*p.block)

def coreLoss (M : Algebra α) (p : Parameters) (args : RuntimeArgs.Values α)
    (x : Nat → α) (block : Nat) (label : Int) : α := by
  classical
  let L := if p.split then M.literal (some .fp32) .real 0.0 else lse M p args x block
  let s := args.smoothing
  let C := M.fromInt (some .fp32) (p.totalClasses : Int)
  let sub := M.binary (some .fp32) .real .sub
  let mul := M.binary (some .fp32) .real .mul
  let div := M.binary (some .fp32) .real .div
  let gathered := mul (x (label-p.classStart).toNat) args.logitScale
  exact if owns p block label then
    if p.hasSmoothing then
      sub (sub L (div (mul s (sum M p args x block)) C))
        (mul (sub (M.literal (some .fp32) .real 1) s) gathered)
    else sub L gathered
  else if p.hasSmoothing then
    let original := mul s (sub L (div (sum M p args x block) C))
    if p.split then original else
      M.binary (some .fp32) .real .add original
        (mul (sub (M.literal (some .fp32) .real 1) s) (lse M p args x block))
  else L

def zLoss (M : Algebra α) (p : Parameters) (args : RuntimeArgs.Values α)
    (x : Nat → α) (block : Nat) (label : Int) : α :=
  if label=p.ignored then M.literal (some .fp32) .real 0.0 else
    if p.split then M.literal (some .fp32) .real 0.0 else
      M.binary (some .fp32) .real .mul
        (M.binary (some .fp32) .real .mul args.squareScale (lse M p args x block))
        (lse M p args x block)

def loss (M : Algebra α) (p : Parameters) (args : RuntimeArgs.Values α)
    (x : Nat → α) (block : Nat) (label : Int) : α :=
  if label=p.ignored then M.literal (some .fp32) .real 0.0 else
    if p.split then coreLoss M p args x block label else
      M.binary (some .fp32) .real .add (coreLoss M p args x block label)
        (zLoss M p args x block label)

def forward (M : Algebra α) (p : Parameters) (args : RuntimeArgs.Values α)
    (block : Nat) (x : Nat → α) (label : Int) : α × α × α :=
  (lse M p args x block,loss M p args x block label,zLoss M p args x block label)

def backward (M : Algebra α) (p : Parameters) (args : RuntimeArgs.Values α)
    (block : Nat) (x : Nat → α) (label : Int) (L upstream : α) (lane : Fin p.block) : α :=
  let lit := M.literal (some .fp32) .real
  let op := M.binary (some .fp32) .real
  let lse := M.fp32Load L
  let probability := M.unary (some .fp32) .exp (op .sub (logits M p args x block (lane,PUnit.unit)) lse)
  let corrected := op .add probability (op .mul (op .mul (op .mul (lit 2.0) args.squareScale) lse) probability)
  let owner := p.classStart≤label ∧ label<p.classStart+(p.cols : Int)
  let shifted := (if owner then label else p.classStart)-p.classStart
  let hit := owner ∧ ((block*p.block+lane.val : Nat) : Int)=shifted
  let adjusted := if p.hasSmoothing then
      op .sub (if hit then op .sub corrected (op .sub (lit 1) args.smoothing) else corrected)
        (op .div args.smoothing (M.fromInt (some .fp32) (p.totalClasses : Int)))
    else if hit then op .sub corrected (lit 1.0) else corrected
  if label=p.ignored then lit 0.0 else
    op .mul (op .mul (M.fp32Load upstream) args.logitScale) adjusted

#specNonCircular forward avoiding [forwardSyntax,backwardSyntax,FP.Structural.exec]
#specNonCircular backward avoiding [forwardSyntax,backwardSyntax,FP.Structural.exec]
end
end VeriTile.Bench.Repairs.CrossEntropy2.Reference
