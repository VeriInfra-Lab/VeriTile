import bench.repairs.tritonbench_g.kv_cache_filling.Proofs.Store
import bench.repairs.tritonbench_g.kv_cache_filling.Proofs.Memory
import VeriTile.Triton.Float.IntegerConversionTotal

namespace VeriTile.Bench.Repairs.KvCacheFilling
open VeriTile.Triton
abbrev Word := FP.Value .fp32

/-- The native source uses 0 - lo for unary minus. The offline sm80 profile
retains div.full.f32 and two separate add.f32 instructions, without FMA. -/
def fp32Minimum (a b : Word) : Word :=
  if a.isNaN || b.isNaN then FP.Value.qNaN .fp32
  else if a.isZero && b.isZero then FP.Value.zero .fp32 (a.sign || b.sign)
  else if FP.lt .ieeeValue a b then a else b

def fp32Maximum (a b : Word) : Word :=
  if a.isNaN || b.isNaN then FP.Value.qNaN .fp32
  else if a.isZero && b.isZero then FP.Value.zero .fp32 (a.sign && b.sign)
  else if FP.lt .ieeeValue b a then a else b

def softwareOps (undefinedCast : Word → Nat) : Operations Word Byte where
  minimum := fp32Minimum
  maximum := fp32Maximum
  sub := FP.sub .ieeeValue
  div := FP.div .ieeeValue
  add := FP.add .ieeeValue
  neg := fun a => FP.sub .ieeeValue (FP.Value.zero .fp32) a
  equal := FP.eq .ieeeValue
  literal := fun n => FP.round .fp32 .ieeeValue (Rat.ofInt n)
  half := FP.Value.ofBits .fp32 0x3f000000
  uint8 := fun x => BitVec.ofNat 8 (FP.IntegerConversion.totalUnsigned undefinedCast 8 x)
  pack := fun a b => a+b*16

def wordByte (x : Word) (b : Nat) : Byte := BitVec.ofNat 8 (x.bits.toNat / 2^(8*b))

/-- Exact local numeric operands at each binary reduction node. Leaves have
no numerical obligation. This is not a whole-reduction result assumption. -/
def Tree.PrimitiveAgreement {α : Type} (a b : α → α → α) (x : Nat → α) : Tree → Prop
  | .leaf _ => True
  | .node l r => l.PrimitiveAgreement a b x ∧ r.PrimitiveAgreement a b x ∧
      a (l.eval a x) (r.eval a x) = b (l.eval a x) (r.eval a x)

theorem Tree.eval_eq_of_primitives {α : Type} (t : Tree) (a b : α → α → α) (x : Nat → α)
    (h : t.PrimitiveAgreement a b x) : t.eval a x = t.eval b x := by
  induction t with
  | leaf i => rfl
  | node l r il ir =>
    obtain ⟨hl,hr,hop⟩ := h
    simp only [Tree.eval]
    rw [hop,il hl,ir hr]

/-- Intermediates name the actual ordered source operations. -/
def scaleAt {α β : Type} (ops : Operations α β) (levels : Nat) (lo hi : α) : α :=
  if ops.equal hi lo then ops.literal 1 else ops.div (ops.sub hi lo) (ops.literal levels)
def zeroAt {α β : Type} (ops : Operations α β) (levels : Nat) (lo hi : α) : α :=
  ops.div (ops.neg lo) (scaleAt ops levels lo hi)
def shiftedAt {α β : Type} (ops : Operations α β) (levels : Nat) (lo hi x : α) : α :=
  ops.add (ops.div x (scaleAt ops levels lo hi)) (zeroAt ops levels lo hi)
def roundedAt {α β : Type} (ops : Operations α β) (levels : Nat) (lo hi x : α) : α :=
  ops.add (shiftedAt ops levels lo hi x) ops.half

/-- Only primitive result equalities on the source's actual scalar operands.
No row, output, address, memory, protocol or successful-run equality is a field. -/
structure ScalarNumerics {α β : Type} (a b : Operations α β) (levels : Nat) (lo hi : α) (x : Nat → α) (n : Nat) : Prop where
  levels_literal : a.literal levels = b.literal levels
  one_literal : a.literal 1 = b.literal 1
  half_literal : a.half = b.half
  comparison : a.equal hi lo = b.equal hi lo
  difference : a.sub hi lo = b.sub hi lo
  range_division : a.div (a.sub hi lo) (a.literal levels) = b.div (a.sub hi lo) (a.literal levels)
  negation : a.neg lo = b.neg lo
  zero_division : a.div (a.neg lo) (scaleAt a levels lo hi) = b.div (a.neg lo) (scaleAt a levels lo hi)
  lane_division : ∀ i, i < n → a.div (x i) (scaleAt a levels lo hi) = b.div (x i) (scaleAt a levels lo hi)
  shift_addition : ∀ i, i < n →
    a.add (a.div (x i) (scaleAt a levels lo hi)) (zeroAt a levels lo hi) =
    b.add (a.div (x i) (scaleAt a levels lo hi)) (zeroAt a levels lo hi)
  half_addition : ∀ i, i < n → a.add (shiftedAt a levels lo hi (x i)) a.half = b.add (shiftedAt a levels lo hi (x i)) a.half
  conversion : ∀ i, i < n → a.uint8 (roundedAt a levels lo hi (x i)) = b.uint8 (roundedAt a levels lo hi (x i))

theorem ScalarNumerics.scale_eq {α β : Type} {a b : Operations α β} {levels n : Nat} {lo hi : α} {x : Nat → α}
    (h : ScalarNumerics a b levels lo hi x n) : scaleAt a levels lo hi = scaleAt b levels lo hi := by
  simp only [scaleAt]
  rw [h.range_division,h.difference,h.levels_literal,h.comparison,h.one_literal]

theorem ScalarNumerics.zero_eq {α β : Type} {a b : Operations α β} {levels n : Nat} {lo hi : α} {x : Nat → α}
    (h : ScalarNumerics a b levels lo hi x n) : zeroAt a levels lo hi = zeroAt b levels lo hi := by
  simp only [zeroAt]
  rw [h.zero_division,h.negation,h.scale_eq]

theorem ScalarNumerics.code_eq {α β : Type} {a b : Operations α β} {levels n : Nat} {lo hi : α} {x : Nat → α}
    (h : ScalarNumerics a b levels lo hi x n) (i : Nat) (hi' : i < n) :
    a.uint8 (roundedAt a levels lo hi (x i)) = b.uint8 (roundedAt b levels lo hi (x i)) := by
  rw [h.conversion i hi']
  congr 1
  unfold roundedAt
  rw [h.half_addition i hi',h.half_literal]
  congr 1
  unfold shiftedAt
  rw [h.shift_addition i hi',h.lane_division i hi',h.scale_eq,h.zero_eq]

structure RowNumerics {α β : Type} (a b : Operations α β) (levels n : Nat) (x : Nat → α) (low high : Tree) : Prop where
  minimum : low.PrimitiveAgreement a.minimum b.minimum x
  maximum : high.PrimitiveAgreement a.maximum b.maximum x
  scalar : ScalarNumerics a b levels (low.eval a.minimum x) (high.eval a.maximum x) x n

theorem quantize_fp {α β : Type} (a b : Operations α β) (levels n : Nat) (x : Nat → α) (low high : Tree)
    (h : RowNumerics a b levels n x low high) :
    (∀ i, i < n → (quantize a levels x low high).code i = (quantize b levels x low high).code i) ∧
    (quantize a levels x low high).scale = (quantize b levels x low high).scale ∧
    (quantize a levels x low high).zero = (quantize b levels x low high).zero := by
  have hl := low.eval_eq_of_primitives a.minimum b.minimum x h.minimum
  have hh := high.eval_eq_of_primitives a.maximum b.maximum x h.maximum
  constructor
  · intro i hi
    change a.uint8 (roundedAt a levels (low.eval a.minimum x) (high.eval a.maximum x) (x i)) = _
    rw [h.scalar.code_eq i hi]
    rw [hl,hh]
    rfl
  · constructor
    · change scaleAt a levels (low.eval a.minimum x) (high.eval a.maximum x) = _
      rw [h.scalar.scale_eq,hl,hh]
      rfl
    · change zeroAt a levels (low.eval a.minimum x) (high.eval a.maximum x) = _
      rw [h.scalar.zero_eq,hl,hh]
      rfl

/-- Defined-domain conversion is separate from numerical conformance. Undefined
cast behavior is not chosen as a hardware fact. -/
def CastDomain (ops : Operations Word Byte) (levels n : Nat) (x : Nat → Word) (low high : Tree) : Prop :=
  ∀ i, i < n → (FP.IntegerConversion.toUnsigned 8
    (roundedAt ops levels (low.eval ops.minimum x) (high.eval ops.maximum x) (x i))).isSome


def PolicyNumerics {α β : Type} (a b : Operations α β) (policy : Policy) (width : Nat)
    (x : Nat → α) (low high : Tree) : Prop :=
  match policy with
  | .copy => True
  | .int8 => RowNumerics a b 255 width x low high
  | .int4 => RowNumerics a b 15 (2*width) x low high ∧
      ∀ i, i < width →
        a.pack ((quantize a 15 x low high).code i) ((quantize a 15 x low high).code (i+width)) =
        b.pack ((quantize a 15 x low high).code i) ((quantize a 15 x low high).code (i+width))

theorem rowKernel_fp_value {α β : Type} (a b : Operations α β) (policy : Policy) (width : Nat)
    (x : Nat → α) (low high : Tree) (h : PolicyNumerics a b policy width x low high)
    (i : Nat) (hi : i < width) :
    (rowKernel a policy width x low high).value i = (rowKernel b policy width x low high).value i := by
  cases policy with
  | copy => rfl
  | int8 => exact congrArg Cell.uint8 ((quantize_fp a b 255 width x low high h).1 i hi)
  | int4 =>
    have q := quantize_fp a b 15 (2*width) x low high h.1
    simp only [rowKernel]
    rw [h.2 i hi,q.1 i (by omega),q.1 (i+width) (by omega)]

theorem rowKernel_fp_metadata {α β : Type} (a b : Operations α β) (policy : Policy) (width : Nat)
    (x : Nat → α) (low high : Tree) (h : PolicyNumerics a b policy width x low high) (hp : policy ≠ .copy) :
    (rowKernel a policy width x low high).scale = (rowKernel b policy width x low high).scale ∧
    (rowKernel a policy width x low high).zero = (rowKernel b policy width x low high).zero := by
  cases policy with
  | copy => exact False.elim (hp rfl)
  | int8 => exact (quantize_fp a b 255 width x low high h).2
  | int4 => exact (quantize_fp a b 15 (2*width) x low high h.1).2

theorem taskValue_fp {α β : Type} (r : Request) (ops ref : Operations α β)
    (kp : Plan r.dk) (vp : Plan r.dv) (k v : Nat → Nat → α) (j : Job) (a : Address)
    (ha : a ∈ rowAddresses r j)
    (hk : PolicyNumerics ops ref r.policy r.ck (k (sourceRow r j)) kp.low kp.high)
    (hv : PolicyNumerics ops ref r.policy r.cv (v (sourceRow r j)) vp.low vp.high) :
    taskValue r ops kp vp k v j a = taskValue r ref kp vp k v j a := by
  obtain ⟨_,hc,har⟩ := (mem_rowAddresses r j a).mp ha
  rcases a with ⟨role,row,col⟩
  cases role with
  | key => exact rowKernel_fp_value ops ref _ _ _ _ _ hk col hc
  | value => exact rowKernel_fp_value ops ref _ _ _ _ _ hv col hc
  | keyMeta =>
    have hp : r.policy ≠ .copy := by simpa [activeRole] using har
    have h := rowKernel_fp_metadata ops ref _ _ _ _ _ hk hp
    simp only [taskValue,rowValue,h.1,h.2]
  | valueMeta =>
    have hp : r.policy ≠ .copy := by simpa [activeRole] using har
    have h := rowKernel_fp_metadata ops ref _ _ _ _ _ hv hp
    simp only [taskValue,rowValue,h.1,h.2]

/-- All real token/head jobs; K and V have distinct primitive trace obligations. -/
def TaskNumerics {α β : Type} (r : Request) (ops ref : Operations α β)
    (kp : Plan r.dk) (vp : Plan r.dv) (k v : Nat → Nat → α) : Prop :=
  ∀ j ∈ jobs r,
    PolicyNumerics ops ref r.policy r.ck (k (sourceRow r j)) kp.low kp.high ∧
    PolicyNumerics ops ref r.policy r.cv (v (sourceRow r j)) vp.low vp.high

theorem whole_fp_bytes (r : Request) (d : Domain r) (l : Layout r) (ops : Operations Word Byte)
    (kp : Plan r.dk) (vp : Plan r.dv) (k v : Nat → Nat → Word) (m : Nat → Byte)
    (undefinedCast : Word → Nat) (h : TaskNumerics r ops (softwareOps undefinedCast) kp vp k v)
    (j : Job) (hj : j ∈ jobs r) (a : Address) (ha : a ∈ rowAddresses r j)
    (b : Nat) (hb : b < elementBytes r a.role) :
    executeBytes r l ops kp vp k v wordByte m (bytePtr r l a b) =
      cellByte wordByte (taskValue r (softwareOps undefinedCast) kp vp k v j a) b := by
  rw [whole_bytes_readback r d l ops kp vp k v wordByte m j hj a ha b hb]
  rw [taskValue_fp r ops (softwareOps undefinedCast) kp vp k v j a ha (h j hj).1 (h j hj).2]


def decodeWord (bytes : Fin 4 → Byte) : Word :=
  FP.Value.ofBits .fp32 ((bytes ⟨0,by omega⟩).toNat +
    (bytes ⟨1,by omega⟩).toNat*256 + (bytes ⟨2,by omega⟩).toNat*65536 + (bytes ⟨3,by omega⟩).toNat*16777216)

def TaskCastDomain (r : Request) (ops : Operations Word Byte) (kp : Plan r.dk) (vp : Plan r.dv)
    (k v : Nat → Nat → Word) : Prop :=
  ∀ j ∈ jobs r,
    match r.policy with
    | .copy => True
    | .int8 => CastDomain ops 255 r.dk (k (sourceRow r j)) kp.low kp.high ∧
        CastDomain ops 255 r.dv (v (sourceRow r j)) vp.low vp.high
    | .int4 => CastDomain ops 15 r.dk (k (sourceRow r j)) kp.low kp.high ∧
        CastDomain ops 15 r.dv (v (sourceRow r j)) vp.low vp.high

theorem software_cast_defined (undefinedCast : Word → Nat) (x : Word)
    (h : (FP.IntegerConversion.toUnsigned 8 x).isSome) :
    (softwareOps undefinedCast).uint8 x = BitVec.ofNat 8 ((FP.IntegerConversion.toUnsigned 8 x).getD 0) := by
  cases hx : FP.IntegerConversion.toUnsigned 8 x with
  | none => simp [hx] at h
  | some z => simp [softwareOps,FP.IntegerConversion.totalUnsigned,hx]

theorem software_cast_spec (x : Word) (z : Nat) (h : FP.IntegerConversion.toUnsigned 8 x = some z) :
    ∃ q, x.toRat? = some q ∧ FP.IntegerConversion.TowardZero q z ∧ FP.IntegerConversion.UnsignedRange 8 z :=
  (FP.IntegerConversion.toUnsigned_spec 8 x z).mp h


theorem fp32_byte_codec (bytes : Fin 4 → Byte) (b : Fin 4) :
    wordByte (decodeWord bytes) b = bytes b := by
  have h0 := (bytes ⟨0,by omega⟩).isLt
  have h1 := (bytes ⟨1,by omega⟩).isLt
  have h2 := (bytes ⟨2,by omega⟩).isLt
  have h3 := (bytes ⟨3,by omega⟩).isLt
  rcases b with ⟨b,hb⟩
  have cases : b=0 ∨ b=1 ∨ b=2 ∨ b=3 := by omega
  rcases cases with he | he | he | he <;> subst b <;>
    apply BitVec.eq_of_toNat_eq <;>
    simp only [wordByte,decodeWord,FP.Value.ofBits,FP.Format.width,FP.Format.fractionBits,BitVec.toNat_ofNat] <;>
    norm_num at * <;> omega


theorem copy_has_no_numeric_obligations (r : Request) (hp : r.policy = .copy)
    (ops ref : Operations Word Byte) (kp : Plan r.dk) (vp : Plan r.dv) (k v : Nat → Nat → Word) :
    TaskNumerics r ops ref kp vp k v ∧ TaskCastDomain r ops kp vp k v := by
  simp [TaskNumerics,TaskCastDomain,PolicyNumerics,hp]

end VeriTile.Bench.Repairs.KvCacheFilling
