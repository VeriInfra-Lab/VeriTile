# Spec sheet — `bench/tritonbench_g/bgmv_expand_slice/FPEquiv.lean`

**Python source:** `bench/tritonbench_g/bgmv_expand_slice/bgmv_expand_slice.py`

## Public theorem: `bgmv_output_first_equiv`

**Statement:**
```lean
specification bgmv_output_first_equiv {α : Type} [Inhabited α]
    (M : Algebra α) (D : FP.Guarded.Domain α)
    (native_add : NativeFP.NativeAdditionCommutes M D
      (original input weight cast out_dtype x w out N K indices
        sx sk sw0 swn swk sy syn slice BN BK splits even add).kernel)
    (s : State α)
    (finite_operands : ∀ site ∈ (original input weight cast out_dtype x w out N K indices
      sx sk sw0 swn swk sy syn slice BN BK splits even add).domain, site.Holds M D s) :
    FP.Structural.exec (FP.ProfiledRewrite.engine M)
      (original input weight cast out_dtype x w out N K indices
        sx sk sw0 swn swk sy syn slice BN BK splits even add).kernel s =
    FP.Structural.exec (FP.ProfiledRewrite.engine M)
      (outputFirst input weight cast out_dtype x w out N K indices
        sx sk sw0 swn swk sy syn slice BN BK splits even add).kernel s
```

**Assumptions / layout contracts:**
- `finite_operands : ∀ site ∈ (original input weight cast out_dtype x w out N K indices
      sx sk sw0 swn swk sy syn slice BN BK splits even add).domain, site.Holds M D s`

**Closed-form spec defs (transitive):** `original`, `outputFirst`

<details><summary><code>original</code></summary>

```lean
abbrev original := FPEquivProofs.original
```
</details>

<details><summary><code>outputFirst</code></summary>

```lean
abbrev outputFirst := FPEquivProofs.outputFirst

section
variable (input : TypedSource.InputStorage) (weight : TypedSource.WeightStorage)
    (cast : Bool) (out_dtype : FloatDType)
    (x w out : RegionName) (N K : Nat) (indices : Region .int)
    (sx sk sw0 swn swk sy syn slice BN BK splits : Nat) (even add : Bool)
```
</details>
