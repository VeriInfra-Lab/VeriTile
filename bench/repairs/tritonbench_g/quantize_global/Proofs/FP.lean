import bench.repairs.tritonbench_g.quantize_global.Proofs.Host
import bench.tritonbench_g.quantize_global.Proofs.RealScalar
/-! Optional software-reference/model-conformance results preserved for the
prior host theorem. Primitive ports below are not deferred numerical
transformation experiments and are not imported by the public kernel entries. -/
namespace QuantizeGlobalRepair.Floating
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 24000
abbrev Word := FP.Value .fp32
instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩
def zero : Word := FP.Value.ofBits .fp32 0
def one : Word := FP.Value.ofBits .fp32 0x3f800000
def hi : Word := FP.Value.ofBits .fp32 0x42fe0000
def lo : Word := FP.Value.ofBits .fp32 0xc2fe0000
def finite (x : Word) : Prop := x.toRat?.isSome
def eq (x y : Word) : Bool := match x.toRat?,y.toRat? with
  | some a,some b => decide (a=b) | _,_ => Bool.false
def lt (x y : Word) : Bool := match x.toRat?,y.toRat? with
  | some a,some b => decide (a<b) | _,_ => Bool.false
def maximum (x y : Word) : Word := if lt x y then y else x
def divisor (m : Word) : Word := if eq m zero then one else m
def quotient (cfg : FP.Config) (x m : Word) : Word := FP.div cfg x (divisor m)
def product (cfg : FP.Config) (x m : Word) : Word := FP.mul cfg hi (quotient cfg x m)
def lower (cfg : FP.Config) (x m : Word) : Word :=
  if lt (product cfg x m) lo then lo else product cfg x m
def clamped (cfg : FP.Config) (x m : Word) : Word :=
  if lt hi (lower cfg x m) then hi else lower cfg x m
/-- Defined finite libdevice llrintf reference. The dummy branch is unreachable
under the explicit local conversion-domain obligation below. -/
def rounded (x : Word) : BitVec 64 := match x.toRat? with
  | some q => BitVec.ofInt 64 (QuantizationReal.nearestEven (q : ℝ))
  | none => 0
def output (cfg : FP.Config) (x m : Word) : BitVec 8 :=
  BitVec.ofInt 8 (rounded (clamped cfg x m)).toInt

/-- Each field is one local scalar primitive obligation, at the exact operand
words of this candidate. No tensor result, scalar quantizer result, execution,
index, mask, launch, or memory equivalence is assumed. `defined` makes the
conversion domain explicit and forbids certifying the reference dummy branch.
The selected production profile is ieeeValue: RNE and gradual underflow.
-/
structure LocalPorts (a : Scalar Word) (cfg : FP.Config) (x m : Word) : Prop where
  literal0 : a.literal 0 = zero
  literal1 : a.literal 1 = one
  literal127 : a.literal 127 = hi
  literalNeg127 : a.literal (-127) = lo
  equalZero : a.eq m zero = eq m zero
  divideRNE : a.div x (divisor m) = quotient cfg x m
  multiply : a.mul hi (quotient cfg x m) = product cfg x m
  lowerCompare : a.lt (product cfg x m) lo = lt (product cfg x m) lo
  upperCompare : a.lt hi (lower cfg x m) = lt hi (lower cfg x m)
  defined : ∃ q, (clamped cfg x m).toRat? = some q ∧ -127 ≤ q ∧ q ≤ 127
  llrint64 : a.round64 (clamped cfg x m) = rounded (clamped cfg x m)

theorem scalar_equivalent (a : Scalar Word) (cfg : FP.Config) (x m : Word)
    (h : LocalPorts a cfg x m) : scalar a x m = output cfg x m := by
  have hd : denominator a m = divisor m := by
    simp [denominator,divisor,h.literal0,h.literal1,h.equalZero]
  have hs : scaled a x m = product cfg x m := by
    simp [scaled,hd,h.literal127,h.divideRNE,h.multiply]
  have hb : bounded a (scaled a x m) = clamped cfg x m := by
    simp only [bounded,hs,h.literalNeg127,h.literal127,h.lowerCompare]
    change (if a.lt hi (lower cfg x m) then hi else lower cfg x m) = _
    rw [h.upperCompare]
    rfl
  simp [scalar,hb,h.llrint64,output]

/-- Host numerical obligations also remain scalar: one abs at each leaf and
one max at each internal node. The reduction tree may have any shape/order. -/
def HostPorts (absolute : Word → Word) (maxOp : Word → Word → Word)
    (xs : Nat → Word) : MaxTree → Prop
  | .empty => True
  | .leaf i => absolute (xs i) = (xs i).abs
  | .node l r => HostPorts absolute maxOp xs l ∧ HostPorts absolute maxOp xs r ∧
      maxOp (l.eval zero FP.Value.abs maximum xs) (r.eval zero FP.Value.abs maximum xs) =
        maximum (l.eval zero FP.Value.abs maximum xs) (r.eval zero FP.Value.abs maximum xs)

theorem maximum_equivalent (absolute : Word → Word) (maxOp : Word → Word → Word)
    (xs : Nat → Word) (tree : MaxTree) (h : HostPorts absolute maxOp xs tree) :
    tree.eval zero absolute maxOp xs = tree.eval zero FP.Value.abs maximum xs := by
  induction tree with
  | empty => rfl
  | leaf i => exact h
  | node l r ihl ihr =>
    change maxOp _ _ = _
    rw [ihl h.1,ihr h.2.1]
    exact h.2.2

#specNonCircular output avoiding [scalar,scaled,bounded,source,kernel,execute,hostRun]
#axiomsClean scalar_equivalent
#axiomsClean maximum_equivalent
end
end QuantizeGlobalRepair.Floating
