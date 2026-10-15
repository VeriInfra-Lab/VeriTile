# Spec sheet — `bench/tritonbench_g/kv_cache_filling/FPCorrect.lean`

**Python source:** `bench/tritonbench_g/kv_cache_filling/kv_cache_filling.py`

## Public theorem: `kv_metadata_typed_correctness`

<details><summary>docstring</summary>

```
/-- Typed scale/zero metadata writeback only. The quantizer producer and the
K/V value conversion are covered by separate public contracts below. -/
```
</details>

**Statement:**
```lean
specification kv_metadata_typed_correctness {α : Type} [Inhabited α]
    (rest : Algebra α) (d : Storage) (p : TypedMetadata.Parameters) (s : State α)
    (hinj : ∀ a b, TypedMetadata.active p a → TypedMetadata.active p b →
      TypedMetadata.outputOffset p s a = TypedMetadata.outputOffset p s b → a = b) :
    ∃ t, FP.Structural.exec (copyBackend rest) (TypedMetadata.source d p) s = some t ∧
      (∀ i : TileIndex [p.width], TypedMetadata.active p i →
        t.mem p.output (TypedMetadata.outputOffset p s i) =
          .mk d.tag ((s.mem p.input i.1.val).read d.tag)) ∧
      (∀ r o, (r ≠ p.output ∨ ∀ i : TileIndex [p.width], TypedMetadata.active p i →
          o ≠ TypedMetadata.outputOffset p s i) → t.mem r o = s.mem r o)
```

**Assumptions / layout contracts:**
- `hinj : ∀ a b, TypedMetadata.active p a → TypedMetadata.active p b →
      TypedMetadata.outputOffset p s a = TypedMetadata.outputOffset p s b → a = b`

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.

## Public theorem: `kv_k_uint8_typed_correctness`

<details><summary>docstring</summary>

```
/-- Actual fp32 state-to-uint8 cache K tile. Active inputs must decode to a
finite rational with a representable truncation toward zero. Dimensions,
padding, strides and selected page/block remain symbolic. Undefined conversions
in inactive lanes have arbitrary policy and cannot change observable memory. -/
```
</details>

**Statement:**
```lean
specification kv_k_uint8_typed_correctness
    (rest : Algebra TypedUInt8.Word) (undefinedValue : Nat → TypedUInt8.Word → Nat)
    (p : TypedUInt8.Parameters) (s : State TypedUInt8.Word)
    (h : TypedUInt8.InputDefined p s)
    (hinj : ∀ a b, TypedUInt8.active p a → TypedUInt8.active p b →
      TypedUInt8.outputOffset p s a = TypedUInt8.outputOffset p s b → a = b) :
    ∃ t, FP.Structural.exec (TypedUInt8.backend rest undefinedValue)
        (TypedUInt8.source Bool.false p) s = some t ∧
      (∀ i : TileIndex [p.height, p.width], TypedUInt8.active p i →
        t.mem p.output (TypedUInt8.outputOffset p s i) =
          .mk .nat (TypedUInt8.expected p s i) ∧
        ∃ q, (TypedUInt8.inputValue p s i).toRat? = some q ∧
          FP.IntegerConversion.TowardZero q (TypedUInt8.expected p s i : Int) ∧
          FP.IntegerConversion.UnsignedRange 8 (TypedUInt8.expected p s i : Int)) ∧
      (∀ r o, (r ≠ p.output ∨ ∀ i : TileIndex [p.height, p.width],
          TypedUInt8.active p i → o ≠ TypedUInt8.outputOffset p s i) →
        t.mem r o = s.mem r o)
```

**Assumptions / layout contracts:**
- `undefinedValue : Nat → TypedUInt8.Word → Nat`
- `h : TypedUInt8.InputDefined p s`
- `hinj : ∀ a b, TypedUInt8.active p a → TypedUInt8.active p b →
      TypedUInt8.outputOffset p s a = TypedUInt8.outputOffset p s b → a = b`

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.

## Public theorem: `kv_v_uint8_typed_correctness`

<details><summary>docstring</summary>

```
/-- Actual fp32 state-to-uint8 cache V tile. Active inputs must decode to a
finite rational with a representable truncation toward zero. Dimensions,
padding, strides and selected page/block remain symbolic. Undefined conversions
in inactive lanes have arbitrary policy and cannot change observable memory. -/
```
</details>

**Statement:**
```lean
specification kv_v_uint8_typed_correctness
    (rest : Algebra TypedUInt8.Word) (undefinedValue : Nat → TypedUInt8.Word → Nat)
    (p : TypedUInt8.Parameters) (s : State TypedUInt8.Word)
    (h : TypedUInt8.InputDefined p s)
    (hinj : ∀ a b, TypedUInt8.active p a → TypedUInt8.active p b →
      TypedUInt8.outputOffset p s a = TypedUInt8.outputOffset p s b → a = b) :
    ∃ t, FP.Structural.exec (TypedUInt8.backend rest undefinedValue)
        (TypedUInt8.source Bool.true p) s = some t ∧
      (∀ i : TileIndex [p.height, p.width], TypedUInt8.active p i →
        t.mem p.output (TypedUInt8.outputOffset p s i) =
          .mk .nat (TypedUInt8.expected p s i) ∧
        ∃ q, (TypedUInt8.inputValue p s i).toRat? = some q ∧
          FP.IntegerConversion.TowardZero q (TypedUInt8.expected p s i : Int) ∧
          FP.IntegerConversion.UnsignedRange 8 (TypedUInt8.expected p s i : Int)) ∧
      (∀ r o, (r ≠ p.output ∨ ∀ i : TileIndex [p.height, p.width],
          TypedUInt8.active p i → o ≠ TypedUInt8.outputOffset p s i) →
        t.mem r o = s.mem r o)
```

**Assumptions / layout contracts:**
- `undefinedValue : Nat → TypedUInt8.Word → Nat`
- `h : TypedUInt8.InputDefined p s`
- `hinj : ∀ a b, TypedUInt8.active p a → TypedUInt8.active p b →
      TypedUInt8.outputOffset p s a = TypedUInt8.outputOffset p s b → a = b`

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.
