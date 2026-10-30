import bench.tritonbench_g.cross_entropy_ops.Kernels
import VeriTile.Triton.Float.ExecutionProfile

/-! Independent, source-ordered arithmetic on logical logits and scalar ABI
values. These definitions neither construct memory nor execute a program. -/
namespace VeriTile.Bench.Repairs.CrossEntropyOps.Ordered
open VeriTile.Triton FP.Structural
noncomputable section

def logits (M : Algebra A) (p : Parameters) (v : RuntimeArgs.Values A)
    (block : Nat) (x : Nat → A) : Values A .real [p.block] :=
  fun i => if block * p.block + i.1.val < p.cols then
    M.binary (some .fp32) .real .mul
      (M.cast (some .fp32) .real .real (x (block * p.block + i.1.val))) v.logitScale
    else M.negInf
def maximum (M : Algebra A) (p : Parameters) (v : RuntimeArgs.Values A)
    (block : Nat) (x : Nat → A) : A :=
  M.reduceMax (some .fp32) ⟨0, by simp⟩ Bool.false (logits M p v block x) PUnit.unit
def sum (M : Algebra A) (p : Parameters) (v : RuntimeArgs.Values A)
    (block : Nat) (x : Nat → A) : A :=
  M.reduceSum (some .fp32) (shape := [p.block]) ⟨0, by simp⟩ Bool.false
    (fun (i, _) => if block * p.block + i.val < p.cols then
      logits M p v block x (i, PUnit.unit) else M.literal (some .fp32) .real 0.0) PUnit.unit
def lse (M : Algebra A) (p : Parameters) (v : RuntimeArgs.Values A)
    (block : Nat) (x : Nat → A) : A :=
  M.binary (some .fp32) .real .add
    (M.unary (some .fp32) .log
      (M.reduceSum (some .fp32) ⟨0, by simp⟩ Bool.false
        (fun i => M.unary (some .fp32) .exp
          (M.binary (some .fp32) .real .sub (logits M p v block x i)
            (maximum M p v block x))) PUnit.unit))
    (maximum M p v block x)
def owns (p : Parameters) (block : Nat) (label : Int) : Prop :=
  (block : Int) * p.block ≤ label - p.classStart ∧
    label - p.classStart < min (p.cols : Int) (((block : Int) + 1) * p.block)
def coreLoss (M : Algebra A) (p : Parameters) (v : RuntimeArgs.Values A)
    (block : Nat) (x : Nat → A) (label : Int) : A := by
  classical
  let L := if p.split then M.literal (some .fp32) .real 0.0 else lse M p v block x
  let C := M.fromInt (some .fp32) (p.totalClasses : Int)
  let sub := M.binary (some .fp32) .real .sub
  let mul := M.binary (some .fp32) .real .mul
  let div := M.binary (some .fp32) .real .div
  let gathered := mul (x (label - p.classStart).toNat) v.logitScale
  exact if owns p block label then
    if p.hasSmoothing then
      sub (sub L (div (mul v.smoothing (sum M p v block x)) C))
        (mul (sub (M.literal (some .fp32) .real 1) v.smoothing) gathered)
    else sub L gathered
  else if p.hasSmoothing then mul v.smoothing (sub L (div (sum M p v block x) C))
  else M.literal (some .fp32) .real 0.0
def loss (M : Algebra A) (p : Parameters) (v : RuntimeArgs.Values A)
    (block : Nat) (x : Nat → A) (label : Int) : A :=
  if label = p.ignored then M.literal (some .fp32) .real 0.0 else
    if p.split then coreLoss M p v block x label else
      M.binary (some .fp32) .real .add (coreLoss M p v block x label)
        (M.binary (some .fp32) .real .mul
          (M.binary (some .fp32) .real .mul v.squareScale (lse M p v block x))
          (lse M p v block x))
def zloss (M : Algebra A) (p : Parameters) (v : RuntimeArgs.Values A)
    (block : Nat) (x : Nat → A) (label : Int) : A :=
  if label = p.ignored then M.literal (some .fp32) .real 0.0 else
    M.binary (some .fp32) .real .mul
      (M.binary (some .fp32) .real .mul v.squareScale (lse M p v block x)) (lse M p v block x)
def gradient (M : Algebra A) (p : Parameters) (v : RuntimeArgs.Values A)
    (block : Nat) (x : Nat → A) (label : Int) (savedLSE upstream : A) (lane : Fin p.block) : A :=
  let lit := M.literal (some .fp32) .real
  let op := M.binary (some .fp32) .real
  let L := M.fp32Load savedLSE
  let probability := M.unary (some .fp32) .exp (op .sub (logits M p v block x (lane, PUnit.unit)) L)
  let corrected := op .add probability (op .mul (op .mul (op .mul (lit 2.0) v.squareScale) L) probability)
  let shifted := (if label = p.ignored then p.classStart else label) - p.classStart
  let hit := ((block * p.block + lane.val : Nat) : Int) = shifted
  let adjusted := if p.hasSmoothing then
      op .sub (if hit then op .sub corrected (op .sub (lit 1) v.smoothing) else corrected)
        (op .div v.smoothing (M.fromInt (some .fp32) (p.totalClasses : Int)))
    else if hit then op .sub corrected (lit 1.0) else corrected
  if label = p.ignored then lit 0.0 else
    op .mul (op .mul (M.fp32Load upstream) v.logitScale) adjusted

#specNonCircular lse avoiding [forwardSyntax, backwardSyntax, exec, run]
#specNonCircular loss avoiding [forwardSyntax, backwardSyntax, exec, run]
#specNonCircular gradient avoiding [forwardSyntax, backwardSyntax, exec, run]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropyOps.Ordered
