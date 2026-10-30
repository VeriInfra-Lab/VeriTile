# Spec sheet — `bench/tritonbench_g/fifth_order_sph_harmonics/FPCorrect.lean`

**Python source:** `bench/tritonbench_g/fifth_order_sph_harmonics/fifth_order_sph_harmonics.py`

## Public theorem: `spherical_forward_typed_correctness`

<details><summary>docstring</summary>

```
/-- Full eleven-channel forward source, evaluated in source operation order.
The value specification reads only the initial coordinate memory. -/
```
</details>

**Statement:**
```lean
specification spherical_forward_typed_correctness {α : Type} [Inhabited α]
    (M : Algebra α) (p : Parameters) (s : State α)
    (hstride : 10 < p.outputStride) (cover : InputCover p s) :
    ∃ t, FP.Structural.exec M (kernel p) s = some t ∧
      (∀ c i, outputOffset p s c i < p.outputNumel →
        t.mem p.output (outputOffset p s c i) = .mk .real (rawExpected M p s c i)) ∧
      (∀ r o, Outside p s r o → t.mem r o = s.mem r o)
```

**Assumptions / layout contracts:**
- `hstride : 10 < p.outputStride`

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.

## Public theorem: `spherical_forward_software_fp32`

<details><summary>docstring</summary>

```
/-- Every arithmetic operation uses the binary32 bit interpreter. Decimal
coefficient encodings are checked by `coefficient_words`; multiplication and
addition are separate operations in this profile. -/
```
</details>

**Statement:**
```lean
specification spherical_forward_software_fp32 [Inhabited Word]
    (rest : Algebra Word) (cfg : FP.Config) (p : Parameters) (s : State Word)
    (hstride : 10 < p.outputStride) (cover : InputCover p s) :
    ∃ t, FP.Structural.exec (software rest cfg) (kernel p) s = some t ∧
      (∀ c i, outputOffset p s c i < p.outputNumel →
        t.mem p.output (outputOffset p s c i) = .mk .real
          (bitChannels cfg ((s.mem p.input (coordOffset p s i)).read .real)
            ((s.mem p.input (coordOffset p s i + 1)).read .real)
            ((s.mem p.input (coordOffset p s i + 2)).read .real) c)) ∧
      (∀ r o, Outside p s r o → t.mem r o = s.mem r o)
```

**Assumptions / layout contracts:**
- `hstride : 10 < p.outputStride`

> ⚠ statement references **no local spec def** — spec may be inlined or stated against an opaque value.
