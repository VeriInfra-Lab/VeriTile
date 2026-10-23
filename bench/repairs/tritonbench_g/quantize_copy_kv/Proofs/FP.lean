import VeriTile.Meta.StatementAudit
import bench.repairs.tritonbench_g.quantize_copy_kv.Proofs.Memory
import bench.repairs.tritonbench_g.rowwise_quantization_triton.Proofs.Encoding
import VeriTile.Triton.Float.IntegerConversionTotal
import bench.tritonbench_g.matmul_dequant_int4.Proofs.Binary16

namespace VeriTile.Bench.Repairs.QuantizeCopyKv
set_option autoImplicit false
set_option maxHeartbeats 2000000
open VeriTile.Triton
abbrev Word := FP.Value .fp32
abbrev Half := VeriTile.Bench.TritonBenchG.MatmulDequantInt4.Binary16.Value

def narrow16 (x : Word) : Half :=
  match x.decode with
  | .nan .. => TritonBenchG.MatmulDequantInt4.Binary16.Value.qNaN
  | .infinity negative => TritonBenchG.MatmulDequantInt4.Binary16.Value.infinity negative
  | .finite negative magnitude =>
    TritonBenchG.MatmulDequantInt4.Binary16.round .ieeeValue (if negative then -magnitude else magnitude) negative

def widen16 (x : Half) : Word :=
  match x.decode with
  | .nan .. => FP.Value.qNaN .fp32
  | .infinity negative => FP.Value.infinity .fp32 negative
  | .finite negative magnitude =>
    FP.round .fp32 .ieeeValue (if negative then -magnitude else magnitude) negative

def decodeWord (bytes : Fin 4 → Byte) : Word :=
  FP.Value.ofBits .fp32 ((bytes ⟨0,by decide⟩).val + 256*(bytes ⟨1,by decide⟩).val +
    65536*(bytes ⟨2,by decide⟩).val + 16777216*(bytes ⟨3,by decide⟩).val)
def halfByte (x : Half) (b : Fin 2) : Byte :=
  ⟨x.bits.toNat / 256^b.val % 256, Nat.mod_lt _ (by decide)⟩

def fp32Maximum (a b : Word) : Word :=
  if a.isNaN then (if b.isNaN then FP.Value.qNaN .fp32 else b)
  else if b.isNaN then a
  else if a.isZero && b.isZero then FP.Value.zero .fp32 (a.sign && b.sign)
  else if FP.lt .ieeeValue a b then b else a

/-- Concrete binary32 arithmetic, correctly rounded binary16 conversion and
exact binary16 promotion. Native division and max-node behavior are local
correspondences below, never silently assumed from offline compilation. -/
def softwareOps (undefinedCast : Word → Int) : Operations Word Half where
  one := FP.Value.ofBits .fp32 0x3f800000
  levels := FP.Value.ofBits .fp32 0x42fe0000
  abs := FP.Value.abs
  maximum := fp32Maximum
  divide := FP.div .ieeeValue
  isZero := FP.Value.isZero
  toHalf := narrow16
  fromHalf := widen16
  int8 := fun x => VeriTile.Bench.Repairs.RowwiseQuantizationTriton.narrow8 (FP.IntegerConversion.totalSigned undefinedCast 8 x)

def Tree.PrimitiveAgreement {α : Type} (actual reference : α → α → α) (x : Nat → α) : Tree → Prop
  | .leaf _ => True
  | .node left right => Tree.PrimitiveAgreement actual reference x left ∧
      Tree.PrimitiveAgreement actual reference x right ∧
      actual (left.evaluate actual x) (right.evaluate actual x) =
        reference (left.evaluate actual x) (right.evaluate actual x)

theorem tree_primitive_equivalence {α : Type} (tree : Tree) (a b : α → α → α) (x : Nat → α)
    (laws : Tree.PrimitiveAgreement a b x tree) : tree.evaluate a x = tree.evaluate b x := by
  induction tree with
  | leaf c => rfl
  | node left right ihl ihr =>
    obtain ⟨hl,hr,hop⟩ := laws
    simp only [RowwiseQuantizationTriton.Tree.evaluate]
    rw [hop,ihl hl,ihr hr]

/-- These fields mention only the single primitive's actual source operands. -/
structure ScalarNumerics {α η : Type} (a b : Operations α η) (maximum : α) (x : Nat → α) (n : Nat) : Prop where
  scale_literal : a.levels = b.levels
  one_literal : a.one = b.one
  scale_division : a.divide maximum a.levels = b.divide maximum a.levels
  half_cast : a.toHalf (a.divide maximum a.levels) = b.toHalf (a.divide maximum a.levels)
  zero_comparison : a.isZero maximum = b.isZero maximum
  half_promotion : a.fromHalf (rowScale a maximum) = b.fromHalf (rowScale a maximum)
  lane_division : ∀ c, c < n → a.divide (x c) (denominator a maximum) = b.divide (x c) (denominator a maximum)
  signed_conversion : ∀ c, c < n →
    a.int8 (a.divide (x c) (denominator a maximum)) = b.int8 (a.divide (x c) (denominator a maximum))

theorem ScalarNumerics.scale_eq {α η : Type} {a b : Operations α η} {maximum : α} {x : Nat → α} {n : Nat}
    (laws : ScalarNumerics a b maximum x n) : rowScale a maximum = rowScale b maximum := by
  unfold rowScale
  rw [laws.half_cast,laws.scale_division,laws.scale_literal]

theorem ScalarNumerics.denominator_eq {α η : Type} {a b : Operations α η} {maximum : α} {x : Nat → α} {n : Nat}
    (laws : ScalarNumerics a b maximum x n) : denominator a maximum = denominator b maximum := by
  unfold denominator
  rw [laws.zero_comparison,laws.one_literal,laws.half_promotion,laws.scale_eq]

theorem ScalarNumerics.code_eq {α η : Type} {a b : Operations α η} {maximum : α} {x : Nat → α} {n : Nat}
    (laws : ScalarNumerics a b maximum x n) (c : Nat) (hc : c < n) :
    rowCode a (x c) maximum = rowCode b (x c) maximum := by
  unfold rowCode
  rw [laws.signed_conversion c hc,laws.lane_division c hc,laws.denominator_eq]

structure RowNumerics {α η : Type} (a b : Operations α η) (tree : Tree) (x : Nat → α) (n : Nat) : Prop where
  absolute : ∀ c, c < n → a.abs (x c) = b.abs (x c)
  maximum : Tree.PrimitiveAgreement a.maximum b.maximum (fun c => a.abs (x c)) tree
  scalar : ScalarNumerics a b (rowMaximum a tree x) x n

theorem row_numerics {α η : Type} (a b : Operations α η) (p : Request) (plan : Plan p) (x : Nat → α)
    (laws : RowNumerics a b plan.tree x p.columns) :
    (∀ c, c < p.columns → (sourceRegisters a plan.tree x).1 c = (sourceRegisters b plan.tree x).1 c) ∧
    (sourceRegisters a plan.tree x).2 = (sourceRegisters b plan.tree x).2 := by
  have hm : rowMaximum a plan.tree x = rowMaximum b plan.tree x := by
    unfold rowMaximum
    rw [tree_primitive_equivalence plan.tree a.maximum b.maximum _ laws.maximum]
    apply RowwiseQuantizationTriton.Tree.evaluate_congr
    intro c hc
    exact laws.absolute c (List.mem_range.mp (plan.covers.mem_iff.mp hc))
  change (∀ c, c < p.columns → rowCode a (x c) (rowMaximum a plan.tree x) = rowCode b (x c) (rowMaximum b plan.tree x)) ∧
    rowScale a (rowMaximum a plan.tree x) = rowScale b (rowMaximum b plan.tree x)
  exact ⟨fun c hc => by rw [laws.scalar.code_eq c hc,hm], by rw [laws.scalar.scale_eq,hm]⟩

def TaskNumerics (a b : Operations Word Half) (p : Request) (plan : Plan p) (v : Layout) (heap : Heap) : Prop :=
  ∀ s h, s < p.tokens → h < p.heads →
    RowNumerics a b plan.tree (loadInput decodeWord p v heap s h) p.columns

def quotient (ops : Operations Word Half) (tree : Tree) (x : Nat → Word) (c : Nat) : Word :=
  ops.divide (x c) (denominator ops (rowMaximum ops tree x))

/-- Precisely the observed conversion trace. Nonzero half-scale underflow,
nonfinite arithmetic and out-of-range conversions are not assigned a GPU byte. -/
def TaskCastDomain (ops : Operations Word Half) (p : Request) (plan : Plan p) (v : Layout) (heap : Heap) : Prop :=
  ∀ s h c, s < p.tokens → h < p.heads → c < p.columns →
    (FP.IntegerConversion.toSigned 8 (quotient ops plan.tree (loadInput decodeWord p v heap s h) c)).isSome

theorem reference_code_defined (undefinedCast : Word → Int) (tree : Tree) (x : Nat → Word) (c : Nat)
    (defined : (FP.IntegerConversion.toSigned 8 (quotient (softwareOps undefinedCast) tree x c)).isSome) :
    ∃ q z, (quotient (softwareOps undefinedCast) tree x c).toRat? = some q ∧
      FP.IntegerConversion.TowardZero q z ∧ FP.IntegerConversion.SignedRange 8 z ∧
      RowwiseQuantizationTriton.signedByte ((sourceRegisters (softwareOps undefinedCast) tree x).1 c) = z := by
  cases hc : FP.IntegerConversion.toSigned 8 (quotient (softwareOps undefinedCast) tree x c) with
  | none => simp [hc] at defined
  | some z =>
    obtain ⟨q,hq,ht,hr⟩ := (FP.IntegerConversion.toSigned_spec 8 _ z).mp hc
    refine ⟨q,z,hq,ht,hr,?_⟩
    change RowwiseQuantizationTriton.signedByte (RowwiseQuantizationTriton.narrow8
      (FP.IntegerConversion.totalSigned undefinedCast 8 (quotient (softwareOps undefinedCast) tree x c))) = z
    rw [FP.IntegerConversion.totalSigned_defined _ _ _ _ hc]
    apply RowwiseQuantizationTriton.narrow8_signed
    · simpa [FP.IntegerConversion.SignedRange] using hr.2.1
    · simpa [FP.IntegerConversion.SignedRange] using hr.2.2

theorem half_bytes_exact (x : Half) :
    (halfByte x ⟨0,by decide⟩).val + 256*(halfByte x ⟨1,by decide⟩).val = x.bits.toNat := by
  have hx := x.bits.isLt
  norm_num at hx
  simp only [halfByte]
  norm_num
  omega

def zeroWord : Word := FP.Value.zero .fp32
def zeroHalf : Half := TritonBenchG.MatmulDequantInt4.Binary16.Value.zero

/-- Closed calculations of the zero primitives, including defined conversion.
The arbitrary undefined-conversion function is never consulted at zero. -/
theorem reference_zero_primitives (undefinedCast : Word → Int) :
    (softwareOps undefinedCast).abs zeroWord = zeroWord ∧
    (softwareOps undefinedCast).maximum zeroWord zeroWord = zeroWord ∧
    rowScale (softwareOps undefinedCast) zeroWord = zeroHalf ∧
    denominator (softwareOps undefinedCast) zeroWord = (softwareOps undefinedCast).one ∧
    FP.IntegerConversion.toSigned 8
      ((softwareOps undefinedCast).divide zeroWord (softwareOps undefinedCast).one) = some 0 ∧
    rowCode (softwareOps undefinedCast) zeroWord zeroWord = RowwiseQuantizationTriton.narrow8 0 := by
  have hz : FP.IntegerConversion.toSigned 8
      (FP.div .ieeeValue zeroWord (FP.Value.ofBits .fp32 0x3f800000)) = some 0 := by decide +kernel
  have hs : narrow16 (FP.div .ieeeValue zeroWord (FP.Value.ofBits .fp32 0x42fe0000)) = zeroHalf := by decide +kernel
  refine ⟨?_,?_,hs,?_,hz,?_⟩
  · change FP.Value.abs zeroWord = zeroWord
    decide +kernel
  · change fp32Maximum zeroWord zeroWord = zeroWord
    decide +kernel
  · rfl
  · change RowwiseQuantizationTriton.narrow8
      (FP.IntegerConversion.totalSigned undefinedCast 8
        ((softwareOps undefinedCast).divide zeroWord (softwareOps undefinedCast).one)) = _
    have hz' : FP.IntegerConversion.toSigned 8
        ((softwareOps undefinedCast).divide zeroWord (softwareOps undefinedCast).one) = some 0 := hz
    rw [FP.IntegerConversion.totalSigned_defined undefinedCast 8 _ 0 hz']

theorem reference_zero_maximum (undefinedCast : Word → Int) (tree : Tree) (x : Nat → Word)
    (zero : ∀ c ∈ tree.leaves, x c = zeroWord) :
    rowMaximum (softwareOps undefinedCast) tree x = zeroWord := by
  induction tree with
  | leaf c =>
    change (softwareOps undefinedCast).abs (x c) = zeroWord
    rw [zero c (by simp [RowwiseQuantizationTriton.Tree.leaves])]
    exact (reference_zero_primitives undefinedCast).1
  | node left right ihl ihr =>
    have hl := ihl (fun c hc => zero c (by simp [RowwiseQuantizationTriton.Tree.leaves,hc]))
    have hr := ihr (fun c hc => zero c (by simp [RowwiseQuantizationTriton.Tree.leaves,hc]))
    change (softwareOps undefinedCast).maximum
      (rowMaximum (softwareOps undefinedCast) left x) (rowMaximum (softwareOps undefinedCast) right x) = zeroWord
    rw [hl,hr]
    exact (reference_zero_primitives undefinedCast).2.1

/-- Every positive-width zero row has scale bits 0000, int8 codes 00, and a
defined signed conversion, for every source-selected reduction tree. -/
theorem reference_zero_row (undefinedCast : Word → Int) (p : Request) (plan : Plan p) (x : Nat → Word)
    (zero : ∀ c, c < p.columns → x c = zeroWord) :
    (∀ c, c < p.columns → (sourceRegisters (softwareOps undefinedCast) plan.tree x).1 c = RowwiseQuantizationTriton.narrow8 0) ∧
    (sourceRegisters (softwareOps undefinedCast) plan.tree x).2 = zeroHalf ∧
    (∀ c, c < p.columns → FP.IntegerConversion.toSigned 8 (quotient (softwareOps undefinedCast) plan.tree x c) = some 0) := by
  have hm := reference_zero_maximum undefinedCast plan.tree x
    (fun c hc => zero c (List.mem_range.mp (plan.covers.mem_iff.mp hc)))
  have laws := reference_zero_primitives undefinedCast
  refine ⟨?_,?_,?_⟩
  · intro c hc
    change rowCode (softwareOps undefinedCast) (x c) (rowMaximum (softwareOps undefinedCast) plan.tree x) = _
    rw [hm,zero c hc]
    exact laws.2.2.2.2.2
  · change rowScale (softwareOps undefinedCast) (rowMaximum (softwareOps undefinedCast) plan.tree x) = _
    rw [hm]
    exact laws.2.2.1
  · intro c hc
    unfold quotient
    rw [hm,zero c hc,laws.2.2.2.1]
    exact laws.2.2.2.2.1

end VeriTile.Bench.Repairs.QuantizeCopyKv

#guard_msgs (drop info) in
#auditModuleAxioms
