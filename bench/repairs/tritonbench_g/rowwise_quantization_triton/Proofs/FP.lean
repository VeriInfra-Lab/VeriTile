import VeriTile.Meta.StatementAudit
import bench.repairs.tritonbench_g.rowwise_quantization_triton.Proofs.Public
namespace VeriTile.Bench.Repairs.RowwiseQuantizationTriton
set_option autoImplicit false

/-- Native-dependent ports are strictly scalar. In particular, Triton's default
FP32 division is not silently identified with correctly rounded software div.
The half port returns a 16-bit binary16 payload, never a bfloat16 value. -/
structure NativeScalars where
  maximum : Word → Word → Word
  divide : Word → Word → Word
  multiply : Word → Word → Word
  llrint : Word → Int
  toHalf : Word → Half

def nativeOperations (native : NativeScalars) : Operations Word where
  zero := VeriTile.Triton.FP.Value.ofBits .fp32 0
  one := VeriTile.Triton.FP.Value.ofBits .fp32 1065353216
  scale := VeriTile.Triton.FP.Value.ofBits .fp32 1123942400
  abs := VeriTile.Triton.FP.Value.abs
  maximum := native.maximum
  isZero := VeriTile.Triton.FP.Value.isZero
  divide := native.divide
  multiply := native.multiply
  llrint := native.llrint
  toHalf := native.toHalf

/-- Local primitive conformance, not a row-output or host-output premise. -/
structure PrimitiveAgreement (actual reference : Operations Word) : Prop where
  zero : actual.zero = reference.zero
  one : actual.one = reference.one
  scale : actual.scale = reference.scale
  abs : ∀ x, actual.abs x = reference.abs x
  maximum : ∀ x y, actual.maximum x y = reference.maximum x y
  isZero : ∀ x, actual.isZero x = reference.isZero x
  divide : ∀ x y, actual.divide x y = reference.divide x y
  multiply : ∀ x y, actual.multiply x y = reference.multiply x y
  llrint : ∀ x, actual.llrint x = reference.llrint x
  toHalf : ∀ x, actual.toHalf x = reference.toHalf x

theorem PrimitiveAgreement.operations {actual reference : Operations Word}
    (agreement : PrimitiveAgreement actual reference) : actual = reference := by
  have ha := funext agreement.abs
  have hm := funext (fun x => funext (agreement.maximum x))
  have hz := funext agreement.isZero
  have hd := funext (fun x => funext (agreement.divide x))
  have hp := funext (fun x => funext (agreement.multiply x))
  have hl := funext agreement.llrint
  have hh := funext agreement.toHalf
  cases actual
  cases reference
  cases agreement
  simp_all

theorem FPEquivalence (api : Services) (actual reference : Operations Word)
    (primitive : PrimitiveAgreement actual reference) (p : Launch) (schedule : Schedule p)
    (input : Input p.rows p.columns) (before : State) (q m : Tensor) (after : Heap)
    (run : forward api actual p schedule input before = some (q, m, after)) :
    ResultCorrect reference p input before q m after := by
  rw [primitive.operations] at run
  exact Correct api reference p schedule input before q m after run

/-- Only six local zero identities are needed for the repaired zero denominator
branch; there is no assumed result for a row or for a complete task. -/
structure ZeroLaws (ops : Operations Word) : Prop where
  abs : ops.abs ops.zero = ops.zero
  maximum : ops.maximum ops.zero ops.zero = ops.zero
  isZero : ops.isZero ops.zero = Bool.true
  divide : ops.divide ops.zero ops.one = ops.zero
  multiply : ops.multiply ops.scale ops.zero = ops.zero
  llrint : ops.llrint ops.zero = 0

theorem Tree.evaluate_zero (ops : Operations Word) (laws : ZeroLaws ops)
    (tree : Tree) (values : Nat → Word) (zero : ∀ i, values i = ops.zero) :
    tree.evaluate ops.maximum values = ops.zero := by
  induction tree with
  | leaf i => exact zero i
  | node a b ha hb => simpa only [Tree.evaluate, ha, hb] using laws.maximum

theorem native_zero_row (ops : Operations Word) (laws : ZeroLaws ops)
    (p : Launch) (values : Nat → Word) (zero : ∀ i, i < p.columns → values i = ops.zero)
    (c : Nat) (hc : c < p.columns) :
    rowMaximum ops p values = ops.zero ∧
    rowQuantized ops (values c) (rowMaximum ops p values) = narrow8 0 := by
  have hm : rowMaximum ops p values = ops.zero := by
    apply Tree.evaluate_zero ops laws
    intro i
    by_cases hi : i < p.columns
    · simp [hi, zero i hi, laws.abs]
    · simp [hi]
  refine ⟨hm, ?_⟩
  simp [rowQuantized, rowScaled, hm, zero c hc, laws.isZero,
    laws.divide, laws.multiply, laws.llrint]

#axiomsClean PrimitiveAgreement.operations
#axiomsClean FPEquivalence
#axiomsClean native_zero_row
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.RowwiseQuantizationTriton
